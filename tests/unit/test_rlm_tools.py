"""Unit tests for the ``rlm_*`` builtins (docs/rlm.md).

DB-free: registry shape (model-only transport, the resumable split), the
deterministic child-id scheme, tier validation, chunking, dispatch-path
budget admission, the manifest render, and one end-to-end ``rlm_query``
spawn with every seam mocked. The live spawn/park/harvest path needs
Postgres and lives in the integration/e2e tiers.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import jsonschema
import pytest

import aios.tools  # noqa: F401 — registers the builtins
from aios.models.context_variables import ContextVariable, VariableKind
from aios.services import sessions as sessions_service
from aios.tools import rlm
from aios.tools.invoke import ToolBail, invoke_builtin
from aios.tools.registry import ToolResult
from tests.unit.conftest import fake_pool_yielding_conn

_SESSION = "ses_parent"
_ACCOUNT = "acc_x"


def _variable(
    name: str,
    *,
    size: int = 0,
    kind: VariableKind = "data",
    description: str = "",
    schema_tag: str | None = None,
) -> ContextVariable:
    now = datetime(2026, 8, 1, 12, 0, 0, tzinfo=UTC)
    return ContextVariable(
        id=f"ctxvar_{name}",
        scope="session",
        session_id=_SESSION,
        name=name,
        kind=kind,
        description=description,
        schema_tag=schema_tag,
        content_sha256="0" * 64,
        content_size_bytes=size,
        metadata={},
        created_at=now,
        updated_at=now,
    )


def _settings(**overrides: Any) -> SimpleNamespace:
    base: dict[str, Any] = {
        "model_tiers": {"sub": "openrouter/cheap"},
        "rlm_max_depth": 2,
        "rlm_max_children_per_step": 8,
        "rlm_max_total_child_tokens": 500_000,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


# ─── registration shape ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("name", "resumable"),
    [("rlm_query", True), ("rlm_map", False), ("rlm_verify", True)],
)
def test_registration_shape(name: str, resumable: bool) -> None:
    """All three are model-only with closed schemas; the single-edge parkers
    are resumable, the N-edge ``rlm_map`` deliberately is not."""
    from aios.tools.registry import registry

    definition = registry.get(name)
    assert definition.transport == "agent_tool"
    assert definition.resumable is resumable
    assert definition.parameters_schema.get("additionalProperties") is False


# ─── rlm_child_session_id ────────────────────────────────────────────────────


def test_child_session_id_is_deterministic() -> None:
    base = rlm.rlm_child_session_id("ses_p", "tc_1")
    assert base == rlm.rlm_child_session_id("ses_p", "tc_1")
    assert base.startswith("sess_")


def test_child_session_id_varies_by_ordinal_and_call() -> None:
    base = rlm.rlm_child_session_id("ses_p", "tc_1")
    assert rlm.rlm_child_session_id("ses_p", "tc_1", ordinal=0) != base
    assert rlm.rlm_child_session_id("ses_p", "tc_1", ordinal=0) != rlm.rlm_child_session_id(
        "ses_p", "tc_1", ordinal=1
    )
    assert rlm.rlm_child_session_id("ses_p", "tc_2") != base


# ─── _require_tier ───────────────────────────────────────────────────────────


def test_require_tier_known(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("aios.tools.rlm.get_settings", lambda: _settings())
    assert rlm._require_tier("sub") == "tier:sub"


def test_require_tier_unknown_lists_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "aios.tools.rlm.get_settings",
        lambda: _settings(model_tiers={"sub": "m", "verify": "n"}),
    )
    with pytest.raises(ToolBail, match="unknown model tier") as exc_info:
        rlm._require_tier("frontier")
    assert exc_info.value.detail == {"configured_tiers": ["sub", "verify"]}


# ─── _chunk ──────────────────────────────────────────────────────────────────


def test_chunk_lines_exact_groups() -> None:
    chunker = rlm._Chunker(kind="lines", size=2)
    assert rlm._chunk("a\nb\nc\nd\ne", chunker) == ["a\nb", "c\nd", "e"]
    assert rlm._chunk("", chunker) == [""]


def test_chunk_bytes_exact_groups() -> None:
    chunker = rlm._Chunker(kind="bytes", size=4)
    assert rlm._chunk("abcdef", chunker) == ["abcd", "ef"]
    assert rlm._chunk("", chunker) == [""]


def test_chunk_json_items_exact_groups() -> None:
    chunker = rlm._Chunker(kind="json_items", size=2)
    assert rlm._chunk("[1, 2, 3, 4, 5]", chunker) == ["[1, 2]", "[3, 4]", "[5]"]
    assert rlm._chunk("[]", chunker) == ["[]"]


def test_chunk_json_items_non_json_bails() -> None:
    with pytest.raises(ToolBail, match="not valid JSON"):
        rlm._chunk("definitely: not json", rlm._Chunker(kind="json_items", size=1))


def test_chunk_json_items_non_array_bails() -> None:
    with pytest.raises(ToolBail, match="JSON array"):
        rlm._chunk('{"a": 1}', rlm._Chunker(kind="json_items", size=1))


# ─── _admit (budget admission) ───────────────────────────────────────────────


class _AdmitSeams(SimpleNamespace):
    spawn_budget: AsyncMock
    parent_seq: AsyncMock
    last_user_seq: AsyncMock
    admit_child: AsyncMock


@pytest.fixture
def admit_seams(monkeypatch: pytest.MonkeyPatch) -> _AdmitSeams:
    monkeypatch.setattr("aios.tools.rlm.get_settings", lambda: _settings())
    seams = _AdmitSeams(
        spawn_budget=AsyncMock(return_value=None),
        parent_seq=AsyncMock(return_value=5),
        last_user_seq=AsyncMock(return_value=7),
        admit_child=AsyncMock(return_value=(1, 0)),
    )
    monkeypatch.setattr("aios.db.queries.get_rlm_spawn_budget", seams.spawn_budget)
    monkeypatch.setattr("aios.db.queries.get_tool_call_parent_seq", seams.parent_seq)
    monkeypatch.setattr("aios.db.queries.get_session_last_user_seq", seams.last_user_seq)
    monkeypatch.setattr("aios.db.queries.admit_rlm_child", seams.admit_child)
    return seams


async def _admit(children: int = 1) -> tuple[Any, int]:
    return await rlm._admit(
        MagicMock(),
        session_id=_SESSION,
        account_id=_ACCOUNT,
        tool_call_id="tc_1",
        children=children,
    )


async def test_admit_root_falls_back_to_settings(admit_seams: _AdmitSeams) -> None:
    """No inbound spawn edge (a root session) → depth and token budget come
    from settings; the resolved budgets and turn key are returned."""
    budgets, turn_key = await _admit()

    assert budgets.depth_remaining == 2
    assert budgets.token_budget == 500_000
    assert budgets.tokens_spent == 0
    assert budgets.tokens_remaining == 500_000
    assert turn_key == 7
    assert admit_seams.admit_child.await_args is not None
    assert admit_seams.admit_child.await_args.kwargs["step_key"] == 5
    assert admit_seams.admit_child.await_args.kwargs["turn_key"] == 7


async def test_admit_depth_exhausted(admit_seams: _AdmitSeams) -> None:
    admit_seams.spawn_budget.return_value = SimpleNamespace(depth=0, token_budget=1_000)

    with pytest.raises(ToolBail, match="depth exhausted") as exc_info:
        await _admit()
    assert exc_info.value.detail["kind"] == "rlm_depth_exhausted"
    admit_seams.admit_child.assert_not_awaited()


async def test_admit_children_over_cap(admit_seams: _AdmitSeams) -> None:
    admit_seams.admit_child.return_value = (9, 0)  # over rlm_max_children_per_step=8

    with pytest.raises(ToolBail, match="children budget") as exc_info:
        await _admit()
    assert exc_info.value.detail["kind"] == "rlm_children_exhausted"
    assert exc_info.value.detail["max_children_per_step"] == 8


async def test_admit_tokens_exhausted(admit_seams: _AdmitSeams) -> None:
    admit_seams.admit_child.return_value = (1, 500_000)  # >= the turn budget

    with pytest.raises(ToolBail, match="token budget") as exc_info:
        await _admit()
    assert exc_info.value.detail["kind"] == "rlm_tokens_exhausted"
    assert exc_info.value.detail["tokens_spent"] == 500_000


async def test_admit_counts_each_child(admit_seams: _AdmitSeams) -> None:
    """``children=n`` admits n times; the returned budgets reflect the last."""
    admit_seams.admit_child.side_effect = [(1, 10), (2, 20)]

    budgets, _ = await _admit(children=2)

    assert admit_seams.admit_child.await_count == 2
    assert budgets.tokens_spent == 20
    assert budgets.tokens_remaining == 500_000 - 20


# ─── _variable_manifest ──────────────────────────────────────────────────────


def test_variable_manifest_renders_names_sizes_descriptions() -> None:
    manifest = rlm._variable_manifest(
        [
            _variable("notes", size=1234, description="my notes", schema_tag="v1"),
            _variable("raw", size=5, kind="spill"),
        ]
    )
    lines = manifest.splitlines()
    assert lines[0] == "Context variables granted to you (read-only):"
    assert lines[1] == "- notes (data, 1,234 bytes) [schema: v1] — my notes"
    assert lines[2] == "- raw (spill, 5 bytes)"
    assert "ctx_peek" in lines[-1] and "ctx_grep" in lines[-1]


def test_variable_manifest_empty_set() -> None:
    assert rlm._variable_manifest([]) == (
        "No context variables were granted; answer from the task alone."
    )


# ─── VERIFY_OUTPUT_SCHEMA ────────────────────────────────────────────────────


def test_verify_output_schema_is_valid_json_schema() -> None:
    jsonschema.Draft202012Validator.check_schema(rlm.VERIFY_OUTPUT_SCHEMA)


# ─── rlm_query end-to-end (all seams mocked) ─────────────────────────────────


async def test_rlm_query_spawns_child_and_returns_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("aios.tools.rlm.get_settings", lambda: _settings())

    pool = fake_pool_yielding_conn(MagicMock())
    monkeypatch.setattr("aios.harness.runtime.require_pool", lambda: pool)
    monkeypatch.setattr(
        "aios.services.sessions.load_session_account_id", AsyncMock(return_value=_ACCOUNT)
    )
    monkeypatch.setattr(
        "aios.services.sessions.get_session_basic",
        AsyncMock(return_value=SimpleNamespace(agent_id="agt_1", environment_id="env_1")),
    )
    monkeypatch.setattr(
        "aios.services.agents.load_for_session",
        AsyncMock(return_value=SimpleNamespace(tools=[], mcp_servers=[], http_servers=[])),
    )
    # Pass the declared child surface through unclamped — the meet itself is
    # covered by the attenuation tests.
    monkeypatch.setattr("aios.services.attenuation.clamp", lambda declared, launcher: declared)
    monkeypatch.setattr("aios.db.queries.get_rlm_spawn_budget", AsyncMock(return_value=None))
    monkeypatch.setattr("aios.db.queries.get_tool_call_parent_seq", AsyncMock(return_value=5))
    monkeypatch.setattr("aios.db.queries.get_session_last_user_seq", AsyncMock(return_value=7))
    monkeypatch.setattr("aios.db.queries.admit_rlm_child", AsyncMock(return_value=(1, 0)))
    monkeypatch.setattr(
        "aios.db.queries.get_session_usage",
        AsyncMock(return_value=SimpleNamespace(total_tokens=123, cost_microusd=4_500)),
    )
    monkeypatch.setattr(
        "aios.db.queries.summarize_session_tool_calls",
        AsyncMock(return_value={"ctx_peek": 2}),
    )
    add_tokens_mock = AsyncMock()
    monkeypatch.setattr("aios.db.queries.add_rlm_child_tokens", add_tokens_mock)

    stimulate_mock = AsyncMock(return_value=True)
    monkeypatch.setattr("aios.services.sessions.stimulate", stimulate_mock)
    # defer_wake is imported into rlm's namespace — patch it there.
    wake_mock = AsyncMock()
    monkeypatch.setattr("aios.tools.rlm.defer_wake", wake_mock)
    park_mock = AsyncMock(return_value={"ok": "answer"})
    monkeypatch.setattr("aios.tools.rlm._park_and_resolve", park_mock)

    out = await invoke_builtin(
        _SESSION, "rlm_query", {"prompt": "summarize the notes"}, tool_call_id="tc_1"
    )

    child_id = rlm.rlm_child_session_id(_SESSION, "tc_1")

    # The spawn is the stimulate spine's AskNewSession arm with a session caller.
    assert stimulate_mock.await_args is not None
    stim = stimulate_mock.await_args.args[1]
    assert isinstance(stim, sessions_service.AskNewSession)
    assert stim.session_id == child_id
    assert stim.caller == {"kind": "session", "id": _SESSION, "tool_call_id": "tc_1"}
    assert stim.model == "tier:sub"  # stamped as the tier: scheme, resolved late
    assert stim.parent_run_id is None
    assert stim.depth == 1  # rlm_max_depth=2, down-counted on the edge
    assert stim.rlm_token_budget == 500_000  # tokens_remaining rides the edge
    assert [t.type for t in stim.surface.tools] == list(rlm._QUERY_CHILD_TOOLS)
    assert stim.surface.mcp_servers == [] and stim.surface.http_servers == []
    assert stimulate_mock.await_args.kwargs["account_id"] == _ACCOUNT

    # First spawn → exactly one wake, on the child.
    wake_mock.assert_awaited_once()
    assert wake_mock.await_args is not None
    assert wake_mock.await_args.args[1] == child_id

    # The park targets the child servicer under this tool call's request id.
    assert park_mock.await_args is not None
    assert park_mock.await_args.kwargs["servicer_kind"] == "session"
    assert park_mock.await_args.kwargs["servicer_id"] == child_id
    assert park_mock.await_args.kwargs["request_id"] == "tc_1"

    # Harvest accrued the child's tokens onto the spawner's turn ledger.
    assert add_tokens_mock.await_args is not None
    assert add_tokens_mock.await_args.kwargs["tokens"] == 123
    assert add_tokens_mock.await_args.kwargs["turn_key"] == 7

    assert isinstance(out, ToolResult)
    assert not out.is_error
    content = out.content
    assert isinstance(content, dict)
    assert content["ok"] == "answer"
    assert content["child_session_id"] == child_id
    assert content["tier"] == "sub"
    assert content["tokens"] == 123
    assert out.metadata is not None
    assert out.metadata["rlm"]["ctx_calls"] == {"ctx_peek": 2}
