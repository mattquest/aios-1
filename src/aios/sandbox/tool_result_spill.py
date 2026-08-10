"""Spill oversized tool-result content into a session-scoped context variable.

A tool result larger than ``tool_result_max_chars`` would, stored inline,
exceed the model context window on its own (windowing can only drop whole
events, never shrink one). Capping it here at the append boundary keeps
``cumulative_tokens`` honest. The full body is written to a session-scoped
context variable (``kind='spill'``, docs/rlm.md) and the event stores a stub
carrying the variable handle plus a deterministic preview — recoverable with
``ctx_peek`` / ``ctx_grep`` / ``ctx_eval`` instead of re-entering the prompt
wholesale.

This inverts the original file spill (``/mnt/attachments/tool_results/…`` +
the ``metadata.attachments`` GC record): the variable store is the one
out-of-context recovery convention, and the stub's preview means even a
surface without ctx tools sees the head of the content rather than a bare
truncation notice. Pre-inversion events keep working — their attachment
records still protect the old files from the boot GC.

Ordering is load-bearing: both result sinks call this BEFORE
``precompute_event_append``, so the running token sums count the stub. The
variable write also runs before the tool-result dedup lock, so a losing
appender's write may land without its event — the upsert is idempotent on
``(session_id, name)`` and simply overwrites with identical content.

Only ``str`` content is capped (the two append sinks guard on
``isinstance(content, str)``); structured/list tool-result content — rare and
typically bounded — passes through uncapped, as do oversized non-tool events.
Tool results are the dominant overflow source; the residual shapes are left
for a follow-up.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

import asyncpg

from aios.models.context_variables import MAX_CONTENT_BYTES

# Deterministic preview frozen into the event content at append time. The
# stub is replayed verbatim by ``build_messages`` on every wake — it must
# never depend on live variable state.
PREVIEW_CHARS = 2_000

_NAME_SAFE = re.compile(r"[^A-Za-z0-9_.-]")
_TRUNCATION_MARKER = "\n[…content truncated at the context-variable byte cap]"


@dataclass(frozen=True)
class CappedToolResult:
    """Outcome of :func:`cap_tool_result_content`.

    ``content`` is the value to store on the tool-result event — the
    original when within the cap, or the handle+preview stub when spilled.
    ``variable_name`` is ``None`` when no spill occurred.
    """

    content: str
    variable_name: str | None


def spill_variable_name(tool_call_id: str) -> str:
    """The session-scoped variable handle for a spilled result.

    A pure function of ``tool_call_id`` (sanitized to the variable-name
    charset, bounded to the 128-char name cap) so retries and the
    worker-vs-API append race key the same handle. When truncation is
    needed, an 8-hex digest of the FULL id is folded in so two long ids
    sharing a 116-char prefix can never alias one variable.
    """
    base = f"tool_result_{_NAME_SAFE.sub('_', tool_call_id)}"
    if len(base) <= 128:
        return base
    digest = hashlib.sha256(tool_call_id.encode()).hexdigest()[:8]
    return f"{base[:119]}_{digest}"


def _bounded_content(content: str) -> str:
    """``content`` cut to the variable byte cap at a character boundary.

    Pathological results beyond the 8 MiB variable cap are truncated with a
    deterministic marker — honest, bounded, and far past anything a model
    can usefully grep.
    """
    encoded = content.encode("utf-8")
    if len(encoded) <= MAX_CONTENT_BYTES:
        return content
    keep = MAX_CONTENT_BYTES - len(_TRUNCATION_MARKER.encode("utf-8"))
    return encoded[:keep].decode("utf-8", errors="ignore") + _TRUNCATION_MARKER


async def cap_tool_result_content(
    executor: asyncpg.Pool[Any] | asyncpg.Connection[Any],
    session_id: str,
    tool_call_id: str,
    content: str,
    *,
    max_chars: int,
) -> CappedToolResult:
    """Return ``content`` unchanged if within ``max_chars``; otherwise spill
    the full body into the session-scoped context variable
    ``tool_result_<tool_call_id>`` and return the handle+preview stub."""
    from aios.db import queries

    if len(content) <= max_chars:
        return CappedToolResult(content=content, variable_name=None)
    name = spill_variable_name(tool_call_id)
    # The stub must itself respect ``max_chars`` (an operator can configure it
    # well below the default): reserve ~400 chars for the prose and shrink the
    # preview to fit. Deterministic given settings, so replay stays pure.
    preview = content[: min(PREVIEW_CHARS, max(0, max_chars - 400))]
    body = _bounded_content(content)
    encoded = body.encode("utf-8")
    try:
        await queries.spill_tool_result_variable(
            executor,
            session_id=session_id,
            name=name,
            content=body,
            content_sha256=hashlib.sha256(encoded).hexdigest(),
            content_size_bytes=len(encoded),
            tool_call_id=tool_call_id,
        )
    except asyncpg.UndefinedTableError:
        # Deploy window (new code, pre-0159 schema): post-deploy migrate has
        # not landed the table yet. Degrade to the bounded preview alone
        # rather than erroring the tool result; the window closes minutes
        # later and the next oversized result spills normally.
        stub = (
            f"[Tool result truncated: the full output ({len(content):,} characters) "
            f"exceeded the inline result limit and the spill store is not yet "
            f"available. Preview below.]\n{preview}"
        )
        return CappedToolResult(content=stub, variable_name=None)
    stub = (
        f"[Tool result spilled: the full output ({len(content):,} characters) exceeded "
        f"the inline result limit and was saved to the context variable {name!r} "
        f"(session scope). Preview below; use ctx_peek/ctx_grep/ctx_eval on {name!r} "
        f"to read the rest.]\n{preview}"
    )
    return CappedToolResult(content=stub, variable_name=name)
