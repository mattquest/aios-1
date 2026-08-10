"""Context-variable prompt affordance (docs/rlm.md).

Two pieces, both existing patterns:

* :func:`augment_with_context_vars` — a **static, cache-stable** system-prompt
  block describing the ctx facility, present iff the surface holds any
  ``ctx_*``/``rlm_*`` tool. Content never varies per step (the
  prompt-prefix-cache rule); the live roster lives in the tail.
* :func:`build_context_vars_tail_block` — the **ephemeral, rebuilt-each-step**
  roster of readable variables (name, scope, kind, size, age), the
  ``obligations.py`` tail trio: block + :func:`max_context_vars_block_local`
  upper bound reserved in ``prelude_overhead_local`` + an
  ``EPHEMERAL_TAIL_KEY`` tag so the cache-breakpoint recognizer skips it.

The roster is what keeps a long-lived agent from cold-starting blind: every
step it sees WHAT exists (metadata only, most-recently-updated first) and
reaches for ``ctx_peek``/``ctx_grep``/``rlm_query`` when it needs content.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from aios.harness._text import join_blocks
from aios.harness.context import EPHEMERAL_TAIL_KEY

if TYPE_CHECKING:
    from aios.models.agents import ToolSpec
    from aios.models.context_variables import ContextVariable

# Max variables rendered as roster lines; beyond this a ``+K more`` marker
# collapses the tail (the MAX_RENDERED_OBLIGATIONS pattern) so the reserved
# budget stays bounded regardless of variable count.
MAX_RENDERED_VARIABLES = 30

# Description clause truncation, mirroring the obligations summary cap.
_DESCRIPTION_MAX = 60

_HEADER = "━━━ Context variables (ctx_peek/ctx_grep to read; never inlined) ━━━"

_CTX_TOOL_TYPES = frozenset(
    {
        "ctx_list",
        "ctx_peek",
        "ctx_grep",
        "ctx_write",
        "ctx_eval",
        "rlm_query",
        "rlm_map",
        "rlm_verify",
    }
)

_FACILITY_BLOCK = """\
## Context variables

Your durable state lives in named context variables OUTSIDE this prompt —
inspect them programmatically instead of ever holding them in context.
Session-scoped variables are private scratch for this session; agent-scoped
variables persist across every session of your agent (your world model).
A roster of what exists is appended at the end of each turn; content is
never auto-inlined.

- ctx_list / ctx_peek / ctx_grep — list metadata, read bounded slices,
  regex-search content.
- ctx_write — persist state (scope='agent' for anything future sessions
  need). Oversized tool results auto-spill into variables; read them back
  the same way.
- ctx_eval — run Python over staged variables in your sandbox; save
  reusable scripts as kind='helper' variables and re-run them by name.
- rlm_query / rlm_map — delegate reading to cheap concurrent child
  sessions that see only your prompt plus granted variables. Prefer this
  over peeking large content into your own context.
- rlm_verify — check a claim against cited variables before acting on or
  asserting it; treat unsupported verdicts as "do not assert"."""


def surface_has_ctx_tools(tools: list[ToolSpec]) -> bool:
    """True iff the (effective) surface holds any ctx/rlm builtin — the gate
    for both the facility block and the per-step roster fetch."""
    return any(t.enabled and t.type in _CTX_TOOL_TYPES for t in tools)


def augment_with_context_vars(base_system: str, *, enabled: bool) -> str:
    """Append the static ctx-facility block when the surface warrants it."""
    if not enabled:
        return base_system
    return join_blocks(base_system, _FACILITY_BLOCK)


def _format_age(updated_at: datetime, now: datetime) -> str:
    secs = max(0, int((now - updated_at).total_seconds()))
    if secs < 60:
        return f"{secs}s"
    if secs < 3600:
        return f"{secs // 60}m"
    if secs < 86400:
        return f"{secs // 3600}h"
    return f"{secs // 86400}d"


def _variable_line(variable: ContextVariable, now: datetime) -> str:
    desc = variable.description.replace("\n", " ").strip()
    if len(desc) > _DESCRIPTION_MAX:
        desc = desc[:_DESCRIPTION_MAX] + "…"
    desc_clause = f' "{desc}"' if desc else ""
    tag = f" <{variable.schema_tag}>" if variable.schema_tag else ""
    return (
        f"• {variable.name} [{variable.scope}/{variable.kind}]{tag} "
        f"{variable.content_size_bytes:,}B ({_format_age(variable.updated_at, now)} ago)"
        f"{desc_clause}"
    )


def build_context_vars_tail_block(
    variables: list[ContextVariable],
    *,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """Ephemeral per-step roster of readable variables, newest-updated first.

    ``None`` on an empty set (zero tail, zero tokens). Never enters
    ``cumulative_tokens`` — render-only, tagged ``EPHEMERAL_TAIL_KEY`` so the
    cache-breakpoint recognizer skips it.
    """
    if not variables:
        return None
    if now is None:
        now = datetime.now(UTC)
    lines = [_HEADER]
    rendered = variables[:MAX_RENDERED_VARIABLES]
    for variable in rendered:
        lines.append(_variable_line(variable, now))
    remaining = len(variables) - len(rendered)
    if remaining > 0:
        lines.append(f"…(+{remaining} more — ctx_list for the full set)")
    return {"role": "user", "content": "\n".join(lines), EPHEMERAL_TAIL_KEY: True}


def max_context_vars_block_local(variables: list[ContextVariable]) -> int:
    """Worst-case local-token cost of the roster tail — the reserve fed into
    ``prelude_overhead_local`` (the ``max_obligations_block_local`` pattern:
    bounded from the REAL fetched set, so the reserve never overshoots)."""
    if not variables:
        return 0
    from aios.harness.context import _USER_MESSAGE_SEPARATOR_CONTENT
    from aios.harness.tokens import approx_tokens

    # A far-future ``now`` renders every age clause at its fattest (``NNNNNNNd``),
    # keeping this a true upper bound on the send-time render.
    block = build_context_vars_tail_block(variables, now=datetime(9999, 1, 1, tzinfo=UTC))
    if block is None:
        return 0
    return approx_tokens(
        [
            {"role": "assistant", "content": _USER_MESSAGE_SEPARATOR_CONTENT},
            block,
        ]
    )
