"""The ``rlm_*`` builtins — recursive sub-queries over context variables.

``rlm_query`` spawns an attenuated generic child session whose context is
only the prompt plus granted variable handles, parks until the child
answers (``return``/``error``), and returns the answer with provenance.
``rlm_map`` fans one child out per chunk of a variable. ``rlm_verify`` is
the stricter sibling: a verifier child limited to ``ctx_peek``/``ctx_grep``
on exactly the cited variables, forced onto a verdict schema.

Everything is composition (docs/rlm.md): the spawn is the ``stimulate``
spine's ``AskNewSession`` arm (same clamp + freeze + trusted edge as a
workflow ``agent()`` child, with a session caller); the park is
``invoke_session._park_and_resolve`` (so worker-crash recovery re-parks
from the durable edge — ``resumable=True``); budgets ride the edge and the
per-session ``rlm_ledgers`` row, enforced here in the dispatch path with
structured errors the model can react to. Children default to NO network,
NO connectors, no write tools: the declared surface is ctx read tools (+
``rlm_query``/``rlm_verify`` so recursion is possible; the down-counting
depth budget is what bounds it), clamped by the lattice meet against the
launcher — recursion never escalates authority.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from typing import Any, Literal

import asyncpg
from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from aios.config import get_settings
from aios.db import queries
from aios.harness import runtime
from aios.harness.model_tier import tier_model_string
from aios.ids import SESSION, make_id
from aios.jobs.app import defer_wake
from aios.logging import get_logger
from aios.models.agents import ToolSpec
from aios.models.attenuation import Surface, surface_of
from aios.models.context_variables import (
    MAX_CONTENT_BYTES,
    ContextVariable,
    VariableName,
)
from aios.services import agents as agents_service
from aios.services import attenuation as attenuation_service
from aios.services import context_variables as ctx_service
from aios.services import sessions as sessions_service
from aios.tools.invoke import ToolBail, current_tool_call_id
from aios.tools.invoke_session import _park_and_resolve
from aios.tools.registry import ToolResult, registry

# The child's declared tool surface. Read-only ctx access plus the recursion
# pair; the effective surface is the meet with the launcher's, so a child
# only ever narrows. No mcp servers, no http servers, no sandbox tools —
# an rlm child never touches the network or a container.
_QUERY_CHILD_TOOLS = ("ctx_list", "ctx_peek", "ctx_grep", "rlm_query", "rlm_verify")
_VERIFY_CHILD_TOOLS = ("ctx_peek", "ctx_grep")

_NAME_SAFE = re.compile(r"[^A-Za-z0-9_.-]")

log = get_logger(__name__)

VERIFY_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "supported": {"type": "boolean"},
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "var": {"type": "string"},
                    "quote": {"type": "string"},
                    "offset": {"type": "integer"},
                },
                "required": ["var", "quote"],
                "additionalProperties": False,
            },
        },
        "notes": {"type": "string"},
    },
    "required": ["supported", "evidence"],
    "additionalProperties": False,
}


def _parse[M: BaseModel](model: type[M], arguments: dict[str, Any]) -> M:
    try:
        return model.model_validate(arguments)
    except PydanticValidationError as exc:
        raise ToolBail(f"invalid arguments: {exc}") from exc


def rlm_child_session_id(
    parent_session_id: str, tool_call_id: str, ordinal: int | None = None
) -> str:
    """Deterministic child id from ``(parent, tool_call_id[, ordinal])`` —
    the ``workflows.child_id`` pattern, so a replayed spawn re-attaches."""
    key = f"rlm:{parent_session_id}:{tool_call_id}"
    if ordinal is not None:
        key = f"{key}#{ordinal}"
    seed = hashlib.sha256(key.encode()).digest()
    return make_id(SESSION, body=seed[:16])


def _require_tier(tier: str) -> str:
    """Validate a tier NAME against config; return the stampable ``tier:`` string.

    Tier names only — a raw model string is refused at this boundary so a
    solopreneur's unit economics stay in one config knob (docs/rlm.md).
    """
    tiers = get_settings().model_tiers
    if tier not in tiers:
        raise ToolBail(
            f"unknown model tier {tier!r}",
            detail={"configured_tiers": sorted(tiers)},
        )
    return tier_model_string(tier)


def _child_surface(tool_types: tuple[str, ...], launcher: Any) -> Surface:
    declared = Surface(
        tools=[ToolSpec(type=t) for t in tool_types],
        mcp_servers=[],
        http_servers=[],
    )
    return attenuation_service.clamp(declared, surface_of(launcher))


def _variable_manifest(variables: list[ContextVariable]) -> str:
    if not variables:
        return "No context variables were granted; answer from the task alone."
    lines = ["Context variables granted to you (read-only):"]
    for v in variables:
        desc = f" — {v.description}" if v.description else ""
        tag = f" [schema: {v.schema_tag}]" if v.schema_tag else ""
        lines.append(f"- {v.name} ({v.kind}, {v.content_size_bytes:,} bytes){tag}{desc}")
    lines.append(
        "Inspect them with ctx_peek (bounded slices) and ctx_grep (regex); "
        "never assume content you have not read."
    )
    return "\n".join(lines)


class _Budgets:
    """Resolved dispatch-path budget state for one spawning session."""

    def __init__(self, depth_remaining: int, token_budget: int, tokens_spent: int) -> None:
        self.depth_remaining = depth_remaining
        self.token_budget = token_budget
        self.tokens_spent = tokens_spent

    @property
    def tokens_remaining(self) -> int:
        return max(0, self.token_budget - self.tokens_spent)


async def _admit(
    pool: asyncpg.Pool[Any],
    *,
    session_id: str,
    account_id: str,
    tool_call_id: str,
    children: int = 1,
) -> tuple[_Budgets, int]:
    """Dispatch-path budget admission for ``children`` spawns.

    Raises a structured :class:`ToolBail` on exhaustion (depth, per-step
    children, per-turn child tokens). Returns the resolved budgets and the
    turn key the harvest must accrue against. Owns its own connection, and
    while holding it awaits only ``queries.*`` calls (the pooled-connection
    rule — the lint cannot see through local helper awaits).
    """
    settings = get_settings()
    async with pool.acquire() as conn:
        spawn = await queries.get_rlm_spawn_budget(conn, session_id, account_id=account_id)
        depth_remaining = spawn.depth if spawn is not None else settings.rlm_max_depth
        token_budget = (
            spawn.token_budget if spawn is not None else settings.rlm_max_total_child_tokens
        )
        if depth_remaining <= 0:
            raise ToolBail(
                "rlm recursion depth exhausted",
                detail={"kind": "rlm_depth_exhausted", "depth": depth_remaining},
            )
        step_key = await queries.get_tool_call_parent_seq(
            conn, session_id, tool_call_id, account_id=account_id
        )
        turn_key = await queries.get_session_last_user_seq(conn, session_id, account_id=account_id)
        budgets: _Budgets | None = None
        for _ in range(children):
            spawned, tokens_spent = await queries.admit_rlm_child(
                conn, session_id=session_id, step_key=step_key or 0, turn_key=turn_key
            )
            if spawned > settings.rlm_max_children_per_step:
                raise ToolBail(
                    "rlm children budget exhausted for this step",
                    detail={
                        "kind": "rlm_children_exhausted",
                        "max_children_per_step": settings.rlm_max_children_per_step,
                    },
                )
            if tokens_spent >= token_budget:
                raise ToolBail(
                    "rlm child-token budget exhausted for this turn",
                    detail={
                        "kind": "rlm_tokens_exhausted",
                        "token_budget": token_budget,
                        "tokens_spent": tokens_spent,
                    },
                )
            budgets = _Budgets(depth_remaining, token_budget, tokens_spent)
        assert budgets is not None
        return budgets, turn_key


async def _spawn_and_await(
    pool: asyncpg.Pool[Any],
    *,
    parent_session_id: str,
    account_id: str,
    environment_id: str,
    child_id: str,
    tool_call_id: str,
    model: str,
    surface: Surface,
    input_text: str,
    output_schema: dict[str, Any] | None,
    depth: int,
    token_budget: int,
    granted_variable_ids: list[str],
    turn_key: int,
    tier: str,
    vars_read: list[str],
) -> ToolResult:
    """Spawn (idempotently), wake, park, harvest — one child end to end."""
    stim = sessions_service.AskNewSession(
        session_id=child_id,
        agent_id=None,
        environment_id=environment_id,
        agent_version=None,
        model=model,
        parent_run_id=None,
        surface=surface,
        vault_ids=[],
        request_id=tool_call_id,
        input=input_text,
        output_schema=output_schema,
        depth=depth,
        caller={"kind": "session", "id": parent_session_id, "tool_call_id": tool_call_id},
        context_variable_ids=granted_variable_ids,
        rlm_token_budget=token_budget,
    )
    spawned = await sessions_service.stimulate(pool, stim, account_id=account_id)
    if spawned:
        # Wake only on first spawn (a replay re-attaches to the running or
        # archived child; waking an archived child would crash the append).
        await defer_wake(pool, child_id, account_id=account_id, cause="rlm_child_spawn")

    resolved = await _park_and_resolve(
        pool,
        servicer_kind="session",
        servicer_id=child_id,
        request_id=tool_call_id,
        account_id=account_id,
        output_schema=output_schema,
    )

    # Harvest: accrue the child's own usage onto the spawner's turn ledger and
    # attach provenance. The child's counters cover only its own inference;
    # its descendants accrued onto ITS ledger the same way (docs/rlm.md).
    async with pool.acquire() as conn:
        usage = await queries.get_session_usage(conn, child_id, account_id=account_id)
        ctx_calls = await queries.summarize_session_tool_calls(
            conn, child_id, account_id=account_id
        )
        if usage.total_tokens:
            await queries.add_rlm_child_tokens(
                conn, session_id=parent_session_id, turn_key=turn_key, tokens=usage.total_tokens
            )

    provenance: dict[str, Any] = {
        "child_session_id": child_id,
        "tier": tier,
        "vars_read": vars_read,
        "granted_variable_ids": granted_variable_ids,
        "tokens": usage.total_tokens,
        "cost_microusd": usage.cost_microusd,
        "ctx_calls": ctx_calls,
    }
    if isinstance(resolved, ToolResult):
        # The park's error shape ({error: …} / output_schema_violation),
        # enriched with provenance so a failed sub-query is still auditable.
        content = resolved.content
        if isinstance(content, dict):
            content = {**content, "child_session_id": child_id, "tier": tier}
        return ToolResult(content=content, metadata={"rlm": provenance}, is_error=True)
    return ToolResult(
        content={
            "ok": resolved["ok"],
            "child_session_id": child_id,
            "tier": tier,
            "vars_read": vars_read,
            "tokens": usage.total_tokens,
            "cost_usd": usage.cost_microusd / 1_000_000,
        },
        metadata={"rlm": provenance},
    )


async def _resolve_granted(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    session_id: str,
    agent_id: str | None,
    names: list[str],
) -> list[ContextVariable]:
    variables: list[ContextVariable] = []
    for name in dict.fromkeys(names):
        variables.append(
            await ctx_service.resolve_readable(
                pool,
                account_id=account_id,
                session_id=session_id,
                agent_id=agent_id,
                name=name,
                include_content=False,
            )
        )
    return variables


# ─── rlm_query ───────────────────────────────────────────────────────────────


class _RlmQueryArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, description="The sub-query task.")
    vars: list[VariableName] = Field(
        default_factory=list,
        description="Context variables granted read-only to the child.",
    )
    tier: str = Field(
        default="sub",
        description="Model tier NAME (configured in AIOS_MODEL_TIERS); never a raw model string.",
    )
    expect: Literal["text", "json"] = Field(
        default="text", description="Answer shape; 'json' instructs a JSON answer."
    )
    output_schema: dict[str, Any] | None = Field(
        default=None,
        description="Optional JSON Schema the answer must satisfy (implies expect='json').",
    )


async def rlm_query_handler(session_id: str, arguments: dict[str, Any]) -> ToolResult:
    args = _parse(_RlmQueryArgs, arguments)
    model = _require_tier(args.tier)
    tool_call_id = current_tool_call_id()
    if tool_call_id is None:
        raise ToolBail("rlm_query requires a tool-call context")

    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    session = await sessions_service.get_session_basic(pool, session_id, account_id=account_id)
    launcher = await agents_service.load_for_session(pool, session, account_id=account_id)

    budgets, turn_key = await _admit(
        pool, session_id=session_id, account_id=account_id, tool_call_id=tool_call_id
    )

    granted = await _resolve_granted(
        pool,
        account_id=account_id,
        session_id=session_id,
        agent_id=session.agent_id,
        names=args.vars,
    )
    expect_json = args.expect == "json" or args.output_schema is not None
    input_text = "\n\n".join(
        part
        for part in (
            args.prompt,
            _variable_manifest(granted),
            "Answer with JSON." if expect_json and args.output_schema is None else None,
        )
        if part is not None
    )
    return await _spawn_and_await(
        pool,
        parent_session_id=session_id,
        account_id=account_id,
        environment_id=session.environment_id,
        child_id=rlm_child_session_id(session_id, tool_call_id),
        tool_call_id=tool_call_id,
        model=model,
        surface=_child_surface(_QUERY_CHILD_TOOLS, launcher),
        input_text=input_text,
        output_schema=args.output_schema,
        depth=budgets.depth_remaining - 1,
        token_budget=budgets.tokens_remaining,
        granted_variable_ids=[v.id for v in granted],
        turn_key=turn_key,
        tier=args.tier,
        vars_read=[v.name for v in granted],
    )


# ─── rlm_map ─────────────────────────────────────────────────────────────────


class _Chunker(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["lines", "bytes", "json_items"]
    size: int = Field(
        default=1,
        ge=1,
        description="Lines/bytes per chunk, or JSON array items per chunk.",
    )


class _RlmMapArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, description="The per-chunk task.")
    var: VariableName = Field(description="The variable whose content is chunked.")
    chunker: _Chunker
    tier: str = Field(default="sub")
    output_schema: dict[str, Any] | None = Field(
        default=None, description="Optional JSON Schema each chunk's answer must satisfy."
    )


def _chunk(content: str, chunker: _Chunker) -> list[str]:
    if chunker.kind == "lines":
        lines = content.splitlines()
        return [
            "\n".join(lines[i : i + chunker.size]) for i in range(0, len(lines), chunker.size)
        ] or [""]
    if chunker.kind == "bytes":
        # Byte windows snapped BACK to utf-8 character boundaries so no byte is
        # ever dropped: the next chunk starts exactly where this one ended.
        encoded = content.encode("utf-8")
        chunks: list[str] = []
        start = 0
        while start < len(encoded):
            end = min(start + chunker.size, len(encoded))
            while end < len(encoded) and end > start and (encoded[end] & 0xC0) == 0x80:
                end -= 1
            if end <= start:  # size smaller than one character: take the whole char
                end = min(start + chunker.size, len(encoded))
                while end < len(encoded) and (encoded[end] & 0xC0) == 0x80:
                    end += 1
            chunks.append(encoded[start:end].decode("utf-8"))
            start = end
        return chunks or [""]
    try:
        items = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ToolBail(f"variable content is not valid JSON: {exc}") from exc
    if not isinstance(items, list):
        raise ToolBail("json_items chunking requires the variable to hold a JSON array")
    return [
        json.dumps(items[i : i + chunker.size]) for i in range(0, len(items), chunker.size)
    ] or ["[]"]


async def rlm_map_handler(session_id: str, arguments: dict[str, Any]) -> ToolResult:
    args = _parse(_RlmMapArgs, arguments)
    model = _require_tier(args.tier)
    tool_call_id = current_tool_call_id()
    if tool_call_id is None:
        raise ToolBail("rlm_map requires a tool-call context")

    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    session = await sessions_service.get_session_basic(pool, session_id, account_id=account_id)
    launcher = await agents_service.load_for_session(pool, session, account_id=account_id)

    source = await ctx_service.resolve_readable(
        pool,
        account_id=account_id,
        session_id=session_id,
        agent_id=session.agent_id,
        name=args.var,
    )
    chunks = _chunk(source.content or "", args.chunker)
    max_children = get_settings().rlm_max_children_per_step
    if len(chunks) > max_children:
        raise ToolBail(
            f"chunker produces {len(chunks)} children, over the per-step cap",
            detail={
                "kind": "rlm_children_exhausted",
                "chunks": len(chunks),
                "max_children_per_step": max_children,
            },
        )
    # The json_items chunker re-serializes (ensure_ascii escapes multi-byte
    # runs), so a chunk can outgrow the variable byte cap even though the
    # source fit. Refuse the whole call up front — a structured, non-evicting
    # refusal the model answers by shrinking the chunker size.
    oversized = [i for i, c in enumerate(chunks) if len(c.encode("utf-8")) > MAX_CONTENT_BYTES]
    if oversized:
        raise ToolBail(
            "chunk exceeds the context-variable byte cap; use a smaller chunker size",
            detail={"chunks": oversized, "max_bytes": MAX_CONTENT_BYTES},
        )

    budgets, turn_key = await _admit(
        pool,
        session_id=session_id,
        account_id=account_id,
        tool_call_id=tool_call_id,
        children=len(chunks),
    )
    per_child_budget = max(budgets.tokens_remaining // len(chunks), 1)
    surface = _child_surface(_QUERY_CHILD_TOOLS, launcher)

    # Stage every chunk as a PARENT-scoped variable BEFORE any spawn, granted
    # to its child inside the spawn transaction — the child can never wake
    # (not even via the periodic sweep) before its input is durable, and a
    # crash between spawns leaves no chunkless child.
    safe_tcid = _NAME_SAFE.sub("_", tool_call_id)
    chunk_vars: list[ContextVariable] = []
    for index, chunk in enumerate(chunks):
        chunk_vars.append(
            await ctx_service.write_variable(
                pool,
                account_id=account_id,
                scope="session",
                session_id=session_id,
                agent_id=None,
                name=f"rlm_chunk_{index}_{safe_tcid}"[:128],
                content=chunk,
                kind="data",
                description=f"chunk {index + 1}/{len(chunks)} of {args.var!r}",
                metadata={"writer_session_id": session_id, "rlm_map_tool_call": tool_call_id},
            )
        )

    async def _one(index: int, chunk_var: ContextVariable) -> ToolResult:
        child_id = rlm_child_session_id(session_id, tool_call_id, ordinal=index)
        try:
            input_text = "\n\n".join(
                (
                    args.prompt,
                    f"Your input is chunk {index + 1}/{len(chunks)} of variable "
                    f"{args.var!r}, granted to you as the context variable "
                    f"{chunk_var.name!r} — read it with ctx_peek/ctx_grep.",
                )
            )
            stim = sessions_service.AskNewSession(
                session_id=child_id,
                agent_id=None,
                environment_id=session.environment_id,
                agent_version=None,
                model=model,
                parent_run_id=None,
                surface=surface,
                vault_ids=[],
                request_id=tool_call_id if index == 0 else f"{tool_call_id}#{index}",
                input=input_text,
                output_schema=args.output_schema,
                depth=budgets.depth_remaining - 1,
                caller={"kind": "session", "id": session_id, "tool_call_id": tool_call_id},
                context_variable_ids=[chunk_var.id],
                rlm_token_budget=per_child_budget,
            )
            spawned = await sessions_service.stimulate(pool, stim, account_id=account_id)
            if spawned:
                await defer_wake(pool, child_id, account_id=account_id, cause="rlm_child_spawn")
            resolved = await _park_and_resolve(
                pool,
                servicer_kind="session",
                servicer_id=child_id,
                request_id=stim.request_id,
                account_id=account_id,
                output_schema=args.output_schema,
            )
            async with pool.acquire() as conn:
                usage = await queries.get_session_usage(conn, child_id, account_id=account_id)
                if usage.total_tokens:
                    await queries.add_rlm_child_tokens(
                        conn, session_id=session_id, turn_key=turn_key, tokens=usage.total_tokens
                    )
        except ToolBail as exc:
            # Contain per-chunk failures: gather must never see an exception —
            # a raise would discard every sibling's result and (worse) take the
            # generic-exception path that evicts the parent's sandbox.
            return ToolResult(
                content={"error": exc.message, **exc.detail, "child_session_id": child_id},
                is_error=True,
            )
        except Exception as exc:
            return ToolResult(
                content={
                    "error": f"{type(exc).__name__}: {exc}",
                    "child_session_id": child_id,
                },
                is_error=True,
            )
        if isinstance(resolved, ToolResult):
            content = resolved.content
            body = content if isinstance(content, dict) else {"error": content}
            return ToolResult(content={**body, "child_session_id": child_id}, is_error=True)
        return ToolResult(
            content={
                "ok": resolved["ok"],
                "child_session_id": child_id,
                "tokens": usage.total_tokens,
            }
        )

    results = await asyncio.gather(*(_one(i, v) for i, v in enumerate(chunk_vars)))
    # The staged chunks are per-call scratch on the parent; archive them so the
    # roster doesn't accumulate one entry per historical map call. Best-effort:
    # a failure here must not discard the harvested results.
    for chunk_var in chunk_vars:
        try:
            await ctx_service.archive_variable(pool, chunk_var.id, account_id=account_id)
        except Exception:
            log.warning("rlm_map.chunk_archive_failed", variable_id=chunk_var.id)
    total_tokens = sum(r.content.get("tokens", 0) for r in results if isinstance(r.content, dict))
    return ToolResult(
        content={
            "results": [r.content for r in results],
            "chunks": len(chunks),
            "errors": sum(1 for r in results if r.is_error),
            "tier": args.tier,
            "tokens": total_tokens,
        },
        metadata={
            "rlm": {
                "children": [
                    r.content.get("child_session_id")
                    for r in results
                    if isinstance(r.content, dict)
                ],
                "tier": args.tier,
                "source_var": args.var,
            }
        },
    )


# ─── rlm_verify ──────────────────────────────────────────────────────────────


class _RlmVerifyArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str = Field(min_length=1, description="The claim to verify.")
    vars: list[VariableName] = Field(
        min_length=1, description="The variables the claim must be supported by."
    )
    tier: str = Field(default="verify")


async def rlm_verify_handler(session_id: str, arguments: dict[str, Any]) -> ToolResult:
    args = _parse(_RlmVerifyArgs, arguments)
    model = _require_tier(args.tier)
    tool_call_id = current_tool_call_id()
    if tool_call_id is None:
        raise ToolBail("rlm_verify requires a tool-call context")

    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    session = await sessions_service.get_session_basic(pool, session_id, account_id=account_id)
    launcher = await agents_service.load_for_session(pool, session, account_id=account_id)

    budgets, turn_key = await _admit(
        pool, session_id=session_id, account_id=account_id, tool_call_id=tool_call_id
    )
    granted = await _resolve_granted(
        pool,
        account_id=account_id,
        session_id=session_id,
        agent_id=session.agent_id,
        names=args.vars,
    )
    input_text = "\n\n".join(
        (
            "Verify the following claim STRICTLY against the granted context "
            "variables. Cite verbatim quotes as evidence — a claim without a "
            "supporting span in a cited variable is unsupported. Do not use "
            "outside knowledge.",
            f"Claim: {args.claim}",
            _variable_manifest(granted),
        )
    )
    return await _spawn_and_await(
        pool,
        parent_session_id=session_id,
        account_id=account_id,
        environment_id=session.environment_id,
        child_id=rlm_child_session_id(session_id, tool_call_id),
        tool_call_id=tool_call_id,
        model=model,
        surface=_child_surface(_VERIFY_CHILD_TOOLS, launcher),
        input_text=input_text,
        output_schema=VERIFY_OUTPUT_SCHEMA,
        depth=0,
        token_budget=min(budgets.tokens_remaining, 50_000),
        granted_variable_ids=[v.id for v in granted],
        turn_key=turn_key,
        tier=args.tier,
        vars_read=[v.name for v in granted],
    )


# ─── descriptions + registration ─────────────────────────────────────────────

RLM_QUERY_DESCRIPTION = (
    "Delegate a sub-query to a cheap child session that sees ONLY your prompt "
    "plus the granted context variables (read-only). The child inspects them "
    "with ctx tools and answers once; you stay responsive while it works, and "
    "several rlm calls in one turn run concurrently. Use for anything that "
    "would otherwise pull large variable content into your own context. "
    "tier names a configured model tier (e.g. 'sub'); pass output_schema to "
    "force a validated JSON answer. Budgets (depth, children per step, child "
    "tokens per turn) are enforced server-side and return structured errors."
)
RLM_MAP_DESCRIPTION = (
    "Fan a per-chunk sub-query out over a context variable: the chunker "
    "(lines/bytes/json_items x size) splits its content, one child session "
    "per chunk runs concurrently on the named tier, and the ordered results "
    "return together. Each child sees only your prompt plus its chunk "
    "(staged as its context variable 'chunk'). Chunk count is capped by the "
    "per-step children budget."
)
RLM_VERIFY_DESCRIPTION = (
    "Verify a claim strictly against cited context variables before acting "
    "on or asserting it. A verifier child (verify tier) limited to "
    "ctx_peek/ctx_grep on exactly those variables returns "
    "{supported, evidence: [{var, quote, offset}], notes} — treat "
    "supported=false or empty evidence as 'do not assert this'."
)


def _register() -> None:
    registry.register(
        name="rlm_query",
        description=RLM_QUERY_DESCRIPTION,
        parameters_schema=_RlmQueryArgs.model_json_schema(),
        handler=rlm_query_handler,
        transport="agent_tool",
        resumable=True,
    )
    # NOT resumable: a map parks on N edges sharing one tool_call_id, but the
    # crash re-park path (find_parked_servicer, LIMIT 1) can only rediscover a
    # single edge — resuming would silently collapse the map to one child's
    # result. Error-repair + model retry is the correct crash semantic; the
    # orphaned children run to completion and self-archive.
    registry.register(
        name="rlm_map",
        description=RLM_MAP_DESCRIPTION,
        parameters_schema=_RlmMapArgs.model_json_schema(),
        handler=rlm_map_handler,
        transport="agent_tool",
    )
    registry.register(
        name="rlm_verify",
        description=RLM_VERIFY_DESCRIPTION,
        parameters_schema=_RlmVerifyArgs.model_json_schema(),
        handler=rlm_verify_handler,
        transport="agent_tool",
        resumable=True,
    )


_register()
