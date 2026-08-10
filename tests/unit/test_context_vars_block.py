"""Unit tests for the context-variable prompt affordance (docs/rlm.md).

Covers :mod:`aios.harness.context_vars`: the surface gate, the static
system-prompt augmenter, the ephemeral per-step roster tail (header, line
render, ``EPHEMERAL_TAIL_KEY`` tag, the ``MAX_RENDERED_VARIABLES`` collapse),
and the ``prelude_overhead_local`` reserve's upper-bound property.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from aios.harness.context import _USER_MESSAGE_SEPARATOR_CONTENT, EPHEMERAL_TAIL_KEY
from aios.harness.context_vars import (
    MAX_RENDERED_VARIABLES,
    augment_with_context_vars,
    build_context_vars_tail_block,
    max_context_vars_block_local,
    surface_has_ctx_tools,
)
from aios.harness.tokens import approx_tokens
from aios.models.agents import ToolSpec
from aios.models.context_variables import ContextVariable, VariableKind, VariableScope

_NOW = datetime(2026, 8, 10, 12, 0, 0, tzinfo=UTC)


def _variable(
    name: str = "notes",
    *,
    size: int = 2048,
    scope: VariableScope = "session",
    kind: VariableKind = "data",
    description: str = "",
    schema_tag: str | None = None,
    updated_at: datetime | None = None,
) -> ContextVariable:
    stamp = updated_at if updated_at is not None else _NOW - timedelta(seconds=30)
    return ContextVariable(
        id=f"ctxvar_{name}",
        scope=scope,
        session_id="ses_1" if scope == "session" else None,
        agent_id="agt_1" if scope == "agent" else None,
        name=name,
        kind=kind,
        description=description,
        schema_tag=schema_tag,
        content_sha256="0" * 64,
        content_size_bytes=size,
        metadata={},
        created_at=stamp,
        updated_at=stamp,
    )


# ─── surface_has_ctx_tools ───────────────────────────────────────────────────


def test_surface_gate_true_for_enabled_ctx_tool() -> None:
    assert surface_has_ctx_tools([ToolSpec(type="ctx_list")]) is True


def test_surface_gate_false_for_disabled_ctx_tool() -> None:
    assert surface_has_ctx_tools([ToolSpec(type="ctx_list", enabled=False)]) is False


def test_surface_gate_false_for_bash_only_surface() -> None:
    assert surface_has_ctx_tools([ToolSpec(type="bash")]) is False


# ─── augment_with_context_vars ───────────────────────────────────────────────


def test_augment_appends_facility_block_when_enabled() -> None:
    out = augment_with_context_vars("base prompt", enabled=True)
    assert out.startswith("base prompt")
    assert "## Context variables" in out


def test_augment_is_identity_when_disabled() -> None:
    assert augment_with_context_vars("base prompt", enabled=False) == "base prompt"


# ─── build_context_vars_tail_block ───────────────────────────────────────────


def test_tail_block_none_on_empty_set() -> None:
    assert build_context_vars_tail_block([]) is None


def test_tail_block_header_and_line_render() -> None:
    block = build_context_vars_tail_block(
        [_variable("wm", scope="agent", kind="digest", description="world model", schema_tag="v2")],
        now=_NOW,
    )
    assert block is not None
    assert block["role"] == "user"
    assert block[EPHEMERAL_TAIL_KEY] is True
    lines = block["content"].splitlines()
    assert lines[0].startswith("━━━ Context variables")
    assert lines[1] == '• wm [agent/digest] <v2> 2,048B (30s ago) "world model"'


def test_tail_block_caps_rendered_variables_with_more_marker() -> None:
    variables = [_variable(f"v{i}") for i in range(MAX_RENDERED_VARIABLES + 5)]
    block = build_context_vars_tail_block(variables, now=_NOW)
    assert block is not None
    lines = block["content"].splitlines()
    # header + MAX_RENDERED_VARIABLES roster lines + the collapse marker.
    assert len(lines) == 1 + MAX_RENDERED_VARIABLES + 1
    assert lines[-1] == "…(+5 more — ctx_list for the full set)"
    assert sum(1 for line in lines if line.startswith("• ")) == MAX_RENDERED_VARIABLES


# ─── max_context_vars_block_local ────────────────────────────────────────────


def test_reserve_zero_on_empty_set() -> None:
    assert max_context_vars_block_local([]) == 0


def test_reserve_upper_bounds_actual_render_cost() -> None:
    """The reserve (rendered with a far-future ``now``, fattest age clauses) is
    a true upper bound on the real send-time render at any ``now``."""
    variables = [
        _variable(f"v{i}", size=10_000 * (i + 1), description="a note about this variable")
        for i in range(5)
    ]
    reserve = max_context_vars_block_local(variables)
    assert reserve > 0

    block = build_context_vars_tail_block(variables, now=_NOW)
    assert block is not None
    actual = approx_tokens(
        [{"role": "assistant", "content": _USER_MESSAGE_SEPARATOR_CONTENT}, block]
    )
    assert reserve >= actual
