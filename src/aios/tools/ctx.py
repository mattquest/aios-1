"""The ``ctx_*`` builtins — the model's window onto context variables.

Context variables (docs/rlm.md) hold state out of the prompt window: the
agent inspects them programmatically instead of ever holding them in
context. ``ctx_list`` is cheap metadata; ``ctx_peek``/``ctx_grep`` return
bounded slices/matches; ``ctx_write`` persists; ``ctx_eval`` runs Python
over staged variable content inside the session's own sandbox — the
mechanism by which the agent authors reusable helpers over its own state
(save the script as a ``kind='helper'`` variable, re-run it by name).

Read visibility is the session's own variables + its agent's agent-scoped
variables + anything granted at spawn (``rlm_query`` children). All caps
are enforced here in the dispatch path, never the prompt.
"""

from __future__ import annotations

import asyncio
import re
import shutil
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from aios.config import get_settings
from aios.harness import runtime
from aios.models.context_variables import (
    MAX_DESCRIPTION_CHARS,
    VariableKind,
    VariableName,
)
from aios.sandbox.spec import resolve_bash_timeout_ceiling
from aios.sandbox.volumes import ensure_owned_dir, ensure_session_attachments_dir, safe_filename
from aios.services import context_variables as ctx_service
from aios.services import sessions as sessions_service
from aios.tools.invoke import ToolBail, current_tool_call_id
from aios.tools.registry import registry

_MAX_GREP_MATCHES = 500
_GREP_LINE_CHARS = 500
_CTX_EVAL_SUBDIR = "ctx"


def _parse[M: BaseModel](model: type[M], arguments: dict[str, Any]) -> M:
    try:
        return model.model_validate(arguments)
    except PydanticValidationError as exc:
        raise ToolBail(f"invalid arguments: {exc}") from exc


async def _session_context(session_id: str) -> tuple[Any, str, str | None]:
    """``(pool, account_id, agent_id)`` for the executing session."""
    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    session = await sessions_service.get_session_basic(pool, session_id, account_id=account_id)
    return pool, account_id, session.agent_id


def _meta_entry(variable: Any) -> dict[str, Any]:
    return {
        "name": variable.name,
        "scope": variable.scope,
        "kind": variable.kind,
        "description": variable.description,
        "schema_tag": variable.schema_tag,
        "size_bytes": variable.content_size_bytes,
        "updated_at": variable.updated_at.isoformat(),
    }


# ─── ctx_list ────────────────────────────────────────────────────────────────


class _CtxListArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope: Literal["session", "agent"] | None = Field(
        default=None, description="Restrict to one scope; omitted lists both plus granted."
    )
    kind: VariableKind | None = Field(default=None, description="Restrict to one kind.")


async def ctx_list_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    args = _parse(_CtxListArgs, arguments)
    pool, account_id, agent_id = await _session_context(session_id)
    variables = await ctx_service.list_readable(
        pool,
        account_id=account_id,
        session_id=session_id,
        agent_id=agent_id,
        scope=args.scope,
        kind=args.kind,
    )
    return {"variables": [_meta_entry(v) for v in variables]}


# ─── ctx_peek ────────────────────────────────────────────────────────────────


class _CtxPeekArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: VariableName
    offset: int = Field(default=0, ge=0, description="Byte offset to start reading from.")
    length: int | None = Field(
        default=None,
        ge=1,
        description="Bytes to read; capped at the server's per-call peek limit.",
    )


async def ctx_peek_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    args = _parse(_CtxPeekArgs, arguments)
    pool, account_id, agent_id = await _session_context(session_id)
    variable = await ctx_service.resolve_readable(
        pool, account_id=account_id, session_id=session_id, agent_id=agent_id, name=args.name
    )
    cap = get_settings().ctx_peek_max_bytes
    length = min(args.length, cap) if args.length is not None else cap
    encoded = (variable.content or "").encode("utf-8")
    window = encoded[args.offset : args.offset + length]
    return {
        "name": variable.name,
        "scope": variable.scope,
        "offset": args.offset,
        "length": len(window),
        "total_size": variable.content_size_bytes,
        "truncated": args.offset + len(window) < variable.content_size_bytes,
        "content": window.decode("utf-8", errors="ignore"),
    }


# ─── ctx_grep ────────────────────────────────────────────────────────────────


class _CtxGrepArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: VariableName
    pattern: str = Field(min_length=1, description="Python regular expression, matched per line.")
    max_matches: int = Field(default=100, ge=1, le=_MAX_GREP_MATCHES)


async def ctx_grep_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    args = _parse(_CtxGrepArgs, arguments)
    try:
        pattern = re.compile(args.pattern)
    except re.error as exc:
        raise ToolBail(f"invalid pattern: {exc}") from exc
    pool, account_id, agent_id = await _session_context(session_id)
    variable = await ctx_service.resolve_readable(
        pool, account_id=account_id, session_id=session_id, agent_id=agent_id, name=args.name
    )

    def _scan() -> tuple[list[dict[str, Any]], int]:
        matches: list[dict[str, Any]] = []
        total = 0
        offset = 0
        for line_no, line in enumerate((variable.content or "").splitlines(), start=1):
            if pattern.search(line):
                total += 1
                if len(matches) < args.max_matches:
                    matches.append(
                        {
                            "line": line_no,
                            "byte_offset": offset,
                            "text": line[:_GREP_LINE_CHARS],
                        }
                    )
            offset += len(line.encode("utf-8")) + 1
        return matches, total

    # Line-at-a-time matching in a thread: an adversarial pattern can be slow,
    # but never blocks the event loop, and per-line inputs bound backtracking.
    matches, total = await asyncio.to_thread(_scan)
    return {
        "name": variable.name,
        "total_matches": total,
        "matches": matches,
        "truncated": total > len(matches),
    }


# ─── ctx_write ───────────────────────────────────────────────────────────────


class _CtxWriteArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: VariableName
    content: str
    scope: Literal["session", "agent"] = Field(
        default="session",
        description="'agent' persists across every session of this agent (the world model).",
    )
    kind: VariableKind = "data"
    description: str = Field(default="", max_length=MAX_DESCRIPTION_CHARS)
    schema_tag: str | None = Field(default=None, max_length=128)
    append: bool = Field(default=False, description="Concatenate onto the existing content.")


async def ctx_write_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    args = _parse(_CtxWriteArgs, arguments)
    pool, account_id, agent_id = await _session_context(session_id)
    if args.scope == "agent" and agent_id is None:
        raise ToolBail("this session has no agent; agent-scoped variables are unavailable")
    try:
        variable = await ctx_service.write_variable(
            pool,
            account_id=account_id,
            scope=args.scope,
            session_id=session_id if args.scope == "session" else None,
            agent_id=agent_id if args.scope == "agent" else None,
            name=args.name,
            content=args.content,
            kind=args.kind,
            description=args.description,
            schema_tag=args.schema_tag,
            metadata={"writer_session_id": session_id},
            append=args.append,
        )
    except ValueError as exc:
        raise ToolBail(str(exc)) from exc
    return _meta_entry(variable)


# ─── ctx_eval ────────────────────────────────────────────────────────────────


class _CtxEvalArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str | None = Field(
        default=None, description="Python source to run. Exactly one of code / helper."
    )
    helper: VariableName | None = Field(
        default=None,
        description="Name of a kind='helper' variable whose content is the script to run.",
    )
    vars: list[VariableName] = Field(
        default_factory=list,
        description="Variables staged read-only for the script (files under $CTX_VARS_DIR).",
    )
    timeout_seconds: int | None = Field(default=None, ge=1)


async def ctx_eval_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    args = _parse(_CtxEvalArgs, arguments)
    if (args.code is None) == (args.helper is None):
        raise ToolBail("exactly one of code / helper must be provided")
    pool, account_id, agent_id = await _session_context(session_id)

    code = args.code
    if args.helper is not None:
        helper = await ctx_service.resolve_readable(
            pool, account_id=account_id, session_id=session_id, agent_id=agent_id, name=args.helper
        )
        if helper.kind != "helper":
            raise ToolBail(f"variable {args.helper!r} is kind={helper.kind!r}, not a helper")
        code = helper.content or ""

    staged: list[tuple[str, str]] = []
    for name in dict.fromkeys(args.vars):  # de-dup, order-preserving
        variable = await ctx_service.resolve_readable(
            pool, account_id=account_id, session_id=session_id, agent_id=agent_id, name=name
        )
        staged.append((name, variable.content or ""))

    max_timeout = await resolve_bash_timeout_ceiling(session_id)
    timeout = min(args.timeout_seconds or max_timeout, max_timeout)

    sandbox = runtime.require_sandbox_registry()
    handle = await sandbox.get_or_provision(session_id, pool=pool)

    call_dir = safe_filename(current_tool_call_id() or "eval")
    host_dir = ensure_session_attachments_dir(session_id) / _CTX_EVAL_SUBDIR / call_dir
    sandbox_dir = f"/mnt/attachments/{_CTX_EVAL_SUBDIR}/{call_dir}"

    def _stage() -> None:
        ensure_owned_dir(host_dir)
        ensure_owned_dir(host_dir / "vars")
        assert code is not None
        (host_dir / "script.py").write_text(code, encoding="utf-8")
        for name, content in staged:
            (host_dir / "vars" / name).write_text(content, encoding="utf-8")

    await asyncio.to_thread(_stage)
    try:
        result = await sandbox.exec(
            handle,
            f"CTX_VARS_DIR={sandbox_dir}/vars python3 {sandbox_dir}/script.py",
            timeout_seconds=timeout,
            max_output_bytes=get_settings().bash_max_output_bytes,
        )
    finally:
        # Staged content is per-call scratch; reap it now rather than waiting
        # for the boot-time attachment GC.
        await asyncio.to_thread(shutil.rmtree, host_dir, True)

    return {
        "exit_code": result.exit_code,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "timed_out": result.timed_out,
        "truncated": result.truncated,
    }


# ─── descriptions + registration ─────────────────────────────────────────────

CTX_LIST_DESCRIPTION = (
    "List your context variables (metadata only: name, scope, kind, size, "
    "schema_tag, updated_at). Session-scoped variables are private to this "
    "session; agent-scoped variables persist across every session of your "
    "agent and are your durable memory. Content is never returned here — "
    "use ctx_peek / ctx_grep / ctx_eval to read."
)
CTX_PEEK_DESCRIPTION = (
    "Read a bounded slice of a context variable's content (byte offset + "
    "length, capped per call). Use repeated peeks, ctx_grep, or ctx_eval for "
    "anything larger than one slice — variables are meant to be inspected "
    "programmatically, not inlined."
)
CTX_GREP_DESCRIPTION = (
    "Search a context variable's content with a Python regular expression, "
    "matched line by line. Returns matching lines with line numbers and byte "
    "offsets (usable as ctx_peek offsets), capped at max_matches."
)
CTX_WRITE_DESCRIPTION = (
    "Create, overwrite, or append to a context variable. scope='session' is "
    "private scratch for this session; scope='agent' persists across all "
    "your sessions — use it for durable state (world model, digests, saved "
    "helpers). kind='helper' marks a Python script runnable via ctx_eval."
)
CTX_EVAL_DESCRIPTION = (
    "Run Python over your context variables inside your sandbox. Provide "
    "code inline, or helper=<name> to run a saved kind='helper' variable. "
    "Requested vars are staged read-only as files in $CTX_VARS_DIR/<name>. "
    "stdout is the result; write reusable scripts once with ctx_write and "
    "re-run them by name so competence compounds."
)


def _register() -> None:
    registry.register(
        name="ctx_list",
        description=CTX_LIST_DESCRIPTION,
        parameters_schema=_CtxListArgs.model_json_schema(),
        handler=ctx_list_handler,
        transport="both",
    )
    registry.register(
        name="ctx_peek",
        description=CTX_PEEK_DESCRIPTION,
        parameters_schema=_CtxPeekArgs.model_json_schema(),
        handler=ctx_peek_handler,
        transport="both",
    )
    registry.register(
        name="ctx_grep",
        description=CTX_GREP_DESCRIPTION,
        parameters_schema=_CtxGrepArgs.model_json_schema(),
        handler=ctx_grep_handler,
        transport="both",
    )
    registry.register(
        name="ctx_write",
        description=CTX_WRITE_DESCRIPTION,
        parameters_schema=_CtxWriteArgs.model_json_schema(),
        handler=ctx_write_handler,
        transport="both",
    )
    registry.register(
        name="ctx_eval",
        description=CTX_EVAL_DESCRIPTION,
        parameters_schema=_CtxEvalArgs.model_json_schema(),
        handler=ctx_eval_handler,
        transport="agent_tool",
        executes="sandbox",
    )


_register()
