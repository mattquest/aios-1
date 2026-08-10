"""Unit tests for the session-caller ``call_*`` builtins (#1127).

These stub the worker pool + services so they need no live Postgres. They cover:

* **identity invariant** — a trusted ``caller``/``account_id`` smuggled into the
  arguments is rejected by the tool schema (``additionalProperties: false``) before
  the handler runs.
* **park + resolve** — the handler writes the edge via ``service.invoke`` with
  ``caller={kind:session, id:<this session>}``, parks via the one awaiter
  ``await_task``, and shapes ``{ok | error}`` off its ``outcome``.
* **output_schema** — a non-conforming answer is reported fail-loud as an error.
* **porcelain wiring** — ``call_agent`` (create+invoke) and ``call_workflow``
  (create_run+await) call the right services with the caller's env / lineage.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

import aios.tools  # noqa: F401 — registers the builtins
from aios.models.tasks import AwaitResponse, TaskHandle
from aios.tools.invoke import ToolBail, invoke_builtin
from aios.tools.registry import ToolResult

_CALLER = "ses_caller"
_ACCOUNT = "acc_x"


@pytest.fixture(autouse=True)
def _stub_runtime(monkeypatch: Any) -> None:
    monkeypatch.setattr("aios.harness.runtime.require_pool", lambda: object())
    monkeypatch.setattr("aios.harness.runtime.require_crypto_box", lambda: object())
    monkeypatch.setattr(
        "aios.services.sessions.load_session_account_id", AsyncMock(return_value=_ACCOUNT)
    )
    monkeypatch.setattr(
        "aios.services.sessions.get_session_basic",
        AsyncMock(return_value=SimpleNamespace(environment_id="env_1", parent_run_id=None)),
    )
    # await_completion-backed parks resolve db_url from settings; make it cheap.
    monkeypatch.setattr(
        "aios.config.get_settings", lambda: SimpleNamespace(db_url="postgresql://x")
    )


def _handle(servicer_id: str = "ses_target", request_id: str = "req_1") -> TaskHandle:
    return TaskHandle(servicer_kind="session", servicer_id=servicer_id, request_id=request_id)


async def test_invoke_arg_schema_forbids_caller(monkeypatch: Any) -> None:
    """A smuggled ``caller`` key is rejected by the schema before the handler runs."""
    with pytest.raises(ToolBail):
        await invoke_builtin(
            _CALLER,
            "call_session",
            {"session_id": "ses_target", "caller": {"kind": "session", "id": "evil"}},
        )


async def test_invoke_parks_and_returns_ok(monkeypatch: Any) -> None:
    inv_mock = AsyncMock(return_value=_handle())
    monkeypatch.setattr("aios.services.sessions.invoke", inv_mock)
    await_mock = AsyncMock(return_value=AwaitResponse(outcome="ok", result={"v": 1}))
    monkeypatch.setattr("aios.services.tasks.await_task", await_mock)
    out = await invoke_builtin(_CALLER, "call_session", {"session_id": "ses_target", "input": "hi"})
    assert out == {"ok": {"v": 1}}
    # caller names THIS session, target_kind=session, target is the model-supplied id.
    assert inv_mock.await_args is not None
    kwargs = inv_mock.await_args.kwargs
    assert kwargs["caller"] == {"kind": "session", "id": _CALLER}
    assert kwargs["target_kind"] == "session"
    assert kwargs["target"] == "ses_target"
    # the park dispatches the awaiter on the session servicer from the handle.
    assert await_mock.await_args is not None
    assert await_mock.await_args.kwargs["servicer_kind"] == "session"
    assert await_mock.await_args.kwargs["request_id"] == "req_1"


async def test_invoke_returns_error_outcome(monkeypatch: Any) -> None:
    monkeypatch.setattr("aios.services.sessions.invoke", AsyncMock(return_value=_handle()))
    monkeypatch.setattr(
        "aios.services.tasks.await_task",
        AsyncMock(return_value=AwaitResponse(outcome="errored", error={"kind": "boom"})),
    )
    out = await invoke_builtin(_CALLER, "call_session", {"session_id": "ses_target"})
    assert isinstance(out, ToolResult)
    assert out.is_error
    assert out.content == {"error": {"kind": "boom"}}


async def test_invoke_output_schema_violation(monkeypatch: Any) -> None:
    """A non-conforming answer is reported fail-loud (output_schema_violation)."""
    monkeypatch.setattr("aios.services.sessions.invoke", AsyncMock(return_value=_handle()))
    monkeypatch.setattr(
        "aios.services.tasks.await_task",
        AsyncMock(return_value=AwaitResponse(outcome="ok", result="not-an-object")),
    )
    out = await invoke_builtin(
        _CALLER,
        "call_session",
        {"session_id": "ses_target", "output_schema": {"type": "object"}},
    )
    assert isinstance(out, ToolResult)
    assert out.is_error
    assert isinstance(out.content, str) and "output_schema_violation" in out.content


async def test_invoke_output_schema_conforms(monkeypatch: Any) -> None:
    monkeypatch.setattr("aios.services.sessions.invoke", AsyncMock(return_value=_handle()))
    monkeypatch.setattr(
        "aios.services.tasks.await_task",
        AsyncMock(return_value=AwaitResponse(outcome="ok", result={"a": 1})),
    )
    out = await invoke_builtin(
        _CALLER,
        "call_session",
        {"session_id": "ses_target", "output_schema": {"type": "object"}},
    )
    assert out == {"ok": {"a": 1}}


async def test_invoke_repolls_until_done(monkeypatch: Any) -> None:
    """A not-yet-done await is re-polled until it resolves."""
    monkeypatch.setattr("aios.services.sessions.invoke", AsyncMock(return_value=_handle()))
    await_mock = AsyncMock(
        side_effect=[
            AwaitResponse(outcome=None),  # still pending → re-poll
            AwaitResponse(outcome="ok", result="ok"),
        ]
    )
    monkeypatch.setattr("aios.services.tasks.await_task", await_mock)
    out = await invoke_builtin(_CALLER, "call_session", {"session_id": "ses_target"})
    assert out == {"ok": "ok"}
    assert await_mock.await_count == 2


async def test_call_agent_forwards_spawn_parameters_and_inherits_by_default(
    monkeypatch: Any,
) -> None:
    inv_mock = AsyncMock(return_value=_handle(servicer_id="ses_child"))
    monkeypatch.setattr("aios.services.sessions.invoke", inv_mock)
    monkeypatch.setattr(
        "aios.services.tasks.await_task",
        AsyncMock(return_value=AwaitResponse(outcome="ok", result="r")),
    )
    await invoke_builtin(
        _CALLER,
        "call_agent",
        {
            "agent_id": "agt_1",
            "input": "go",
            "vault_ids": ["vlt_1"],
            "env": {"MODE": "build"},
            "resources": [],
            "title": "builder",
            "metadata": {"job": 1},
        },
    )
    assert inv_mock.await_args is not None
    kwargs = inv_mock.await_args.kwargs
    assert kwargs["launcher_session_id"] == _CALLER
    assert kwargs["vault_ids"] == ["vlt_1"]
    assert kwargs["env"] == {"MODE": "build"}
    assert kwargs["resources"] == []
    assert kwargs["title"] == "builder"
    assert kwargs["metadata"] == {"job": 1}


@pytest.mark.parametrize("tool_args, expected", [({}, None), ({"vault_ids": []}, [])])
async def test_call_agent_preserves_omitted_vs_empty_vault_selection(
    monkeypatch: Any, tool_args: dict[str, Any], expected: list[str] | None
) -> None:
    """Omitted means inherit all; an explicit empty list means inherit none."""
    inv_mock = AsyncMock(return_value=_handle(servicer_id="ses_child"))
    monkeypatch.setattr("aios.services.sessions.invoke", inv_mock)
    monkeypatch.setattr(
        "aios.services.tasks.await_task",
        AsyncMock(return_value=AwaitResponse(outcome="ok", result="r")),
    )

    await invoke_builtin(_CALLER, "call_agent", {"agent_id": "agt_1", **tool_args})

    assert inv_mock.await_args is not None
    assert inv_mock.await_args.kwargs["vault_ids"] == expected


async def test_invoke_agent_create_then_invoke(monkeypatch: Any) -> None:
    inv_mock = AsyncMock(return_value=_handle(servicer_id="ses_child"))
    monkeypatch.setattr("aios.services.sessions.invoke", inv_mock)
    monkeypatch.setattr(
        "aios.services.tasks.await_task",
        AsyncMock(return_value=AwaitResponse(outcome="ok", result="r")),
    )
    out = await invoke_builtin(_CALLER, "call_agent", {"agent_id": "agt_1", "input": "go"})
    assert out == {"ok": "r"}
    assert inv_mock.await_args is not None
    kwargs = inv_mock.await_args.kwargs
    assert kwargs["target_kind"] == "agent"
    assert kwargs["target"] == "agt_1"
    assert kwargs["environment_id"] == "env_1"  # inherits caller's env
    assert kwargs["caller"] == {"kind": "session", "id": _CALLER}


async def test_invoke_workflow_create_run_then_await(monkeypatch: Any) -> None:
    run_mock = AsyncMock(return_value=SimpleNamespace(id="run_1"))
    monkeypatch.setattr("aios.services.workflows.create_run", run_mock)
    await_mock = AsyncMock(return_value=AwaitResponse(outcome="ok", result={"k": "v"}))
    monkeypatch.setattr("aios.services.tasks.await_task", await_mock)
    out = await invoke_builtin(
        _CALLER,
        "call_workflow",
        {"workflow_id": "wf_1", "input": {"x": 1}, "vault_ids": ["vlt_1"], "budget_usd": 2.5},
    )
    assert out == {"ok": {"k": "v"}}
    assert run_mock.await_args is not None
    kwargs = run_mock.await_args.kwargs
    assert kwargs["workflow_id"] == "wf_1"
    assert kwargs["environment_id"] == "env_1"
    assert kwargs["launcher_session_id"] == _CALLER
    # Stage-5b: the dropped create_run/await_run model tools' run-shaping args
    # (vault attenuation + spend ceiling) now ride on call_workflow itself.
    assert kwargs["vault_ids"] == ["vlt_1"]
    assert kwargs["budget_usd"] == 2.5
    # launch_awaited_run stamps the awaited contract onto the caller before create_run.
    assert kwargs["caller"] == {"kind": "session", "id": _CALLER, "awaited": True}
    assert isinstance(kwargs["request_id"], str)
    # the park dispatches the one awaiter on the run servicer it just created.
    assert await_mock.await_args is not None
    assert await_mock.await_args.kwargs["servicer_kind"] == "run"
    assert await_mock.await_args.kwargs["servicer_id"] == "run_1"


async def test_call_workflow_rejects_injected_environment_id(monkeypatch: Any) -> None:
    """F2: the run inherits the caller's env — ``environment_id`` is not a field, so a
    smuggled one is rejected by the schema before the handler runs."""
    with pytest.raises(ToolBail):
        await invoke_builtin(
            _CALLER, "call_workflow", {"workflow_id": "wf_1", "environment_id": "env_other"}
        )


async def test_invoke_workflow_error_outcome(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "aios.services.workflows.create_run", AsyncMock(return_value=SimpleNamespace(id="run_1"))
    )
    monkeypatch.setattr(
        "aios.services.tasks.await_task",
        AsyncMock(return_value=AwaitResponse(outcome="errored", error={"kind": "x"})),
    )
    out = await invoke_builtin(_CALLER, "call_workflow", {"workflow_id": "wf_1"})
    assert isinstance(out, ToolResult)
    assert out.is_error
    assert out.content == {"error": {"kind": "x"}}


async def test_invoke_workflow_output_schema_violation(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "aios.services.workflows.create_run", AsyncMock(return_value=SimpleNamespace(id="run_1"))
    )
    monkeypatch.setattr(
        "aios.services.tasks.await_task",
        AsyncMock(return_value=AwaitResponse(outcome="ok", result="bad")),
    )
    out = await invoke_builtin(
        _CALLER, "call_workflow", {"workflow_id": "wf_1", "output_schema": {"type": "object"}}
    )
    assert isinstance(out, ToolResult)
    assert out.is_error
    assert isinstance(out.content, str) and "output_schema_violation" in out.content


# ─── #1431: caller tool_call_id on the edge (crash-resume link) ───────────────


async def test_caller_carries_tool_call_id(monkeypatch: Any) -> None:
    """The launching ``tool_call_id`` rides onto the servicer edge via ``caller`` so a
    parked task can be re-derived and re-parked after a worker crash (#1431)."""
    inv_mock = AsyncMock(return_value=_handle())
    monkeypatch.setattr("aios.services.sessions.invoke", inv_mock)
    monkeypatch.setattr(
        "aios.services.tasks.await_task",
        AsyncMock(return_value=AwaitResponse(outcome="ok", result="r")),
    )
    await invoke_builtin(
        _CALLER, "call_session", {"session_id": "ses_target"}, tool_call_id="tc_42"
    )
    assert inv_mock.await_args is not None
    assert inv_mock.await_args.kwargs["caller"] == {
        "kind": "session",
        "id": _CALLER,
        "tool_call_id": "tc_42",
    }


async def test_caller_omits_tool_call_id_when_unset(monkeypatch: Any) -> None:
    """Outside a dispatched tool (no ``tool_call_id``) the caller edge stays clean — the
    key is omitted, never written as ``null``."""
    inv_mock = AsyncMock(return_value=_handle())
    monkeypatch.setattr("aios.services.sessions.invoke", inv_mock)
    monkeypatch.setattr(
        "aios.services.tasks.await_task",
        AsyncMock(return_value=AwaitResponse(outcome="ok", result="r")),
    )
    await invoke_builtin(_CALLER, "call_session", {"session_id": "ses_target"})
    assert inv_mock.await_args is not None
    assert "tool_call_id" not in inv_mock.await_args.kwargs["caller"]


async def test_workflow_caller_carries_tool_call_id(monkeypatch: Any) -> None:
    """The run servicer's ``caller`` (wf_runs.caller) carries the tool_call_id too — the
    crash-resume link is kind-uniform."""
    run_mock = AsyncMock(return_value=SimpleNamespace(id="run_1"))
    monkeypatch.setattr("aios.services.workflows.create_run", run_mock)
    monkeypatch.setattr(
        "aios.services.tasks.await_task",
        AsyncMock(return_value=AwaitResponse(outcome="ok", result="r")),
    )
    await invoke_builtin(_CALLER, "call_workflow", {"workflow_id": "wf_1"}, tool_call_id="tc_7")
    assert run_mock.await_args is not None
    caller = run_mock.await_args.kwargs["caller"]
    assert caller["tool_call_id"] == "tc_7"
    assert caller["kind"] == "session" and caller["id"] == _CALLER
    assert caller["awaited"] is True  # launch_awaited_run stamps the Ask bit


async def test_current_tool_call_id_scoped_to_call(monkeypatch: Any) -> None:
    """The contextvar is scoped to the handler call — ``None`` before and after."""
    from aios.tools.invoke import current_tool_call_id

    monkeypatch.setattr("aios.services.sessions.invoke", AsyncMock(return_value=_handle()))
    monkeypatch.setattr(
        "aios.services.tasks.await_task",
        AsyncMock(return_value=AwaitResponse(outcome="ok", result="r")),
    )
    assert current_tool_call_id() is None
    await invoke_builtin(_CALLER, "call_session", {"session_id": "ses_target"}, tool_call_id="tc_9")
    assert current_tool_call_id() is None


def test_resumable_tools_are_exactly_the_parking_call_builtins() -> None:
    """The ghost-repair sweep re-parks exactly the registered ``resumable`` builtins
    (:meth:`ToolRegistry.resumable_tool_names` — its single source of truth). The one place
    a tool declares itself pure-await is ``resumable=True`` at registration; this assertion
    is a tripwire that fails if the derived set changes, forcing a deliberate confirmation
    the change was intended (#1431). A parking tool left ``resumable=False`` would be
    silently error-repaired = re-orphaned on restart; a side-effectful tool wrongly marked
    ``resumable`` would be re-parked-and-re-read as if pure — a double-execution risk.

    ``defer_obligations`` (#1533) is a deliberate addition: a pure await on the
    session's own event channel with NO servicer edge, so the resumable branch's
    ``find_parked_servicer`` lookup returns ``None`` and it lands in the retryable
    ``launch_lost`` result — the design's crash semantic (a wait is not a side
    effect; the model just re-defers)."""
    from aios.tools.registry import registry  # ``aios.tools`` imported at module top

    assert registry.resumable_tool_names() == frozenset(
        {
            "call_session",
            "call_agent",
            "call_workflow",
            "defer_obligations",
            # rlm_query/rlm_verify (docs/rlm.md) park on a single durable
            # request edge exactly like call_agent — pure-await, re-parkable.
            # rlm_map is deliberately NOT here: it parks on N edges under one
            # tool_call_id, which the LIMIT-1 re-park lookup cannot rediscover.
            "rlm_query",
            "rlm_verify",
        }
    )
    for name in registry.resumable_tool_names():
        assert registry.get(name).transport == "agent_tool"
    # The default is non-resumable: a side-effectful tool must never be re-parked.
    assert registry.get("bash").resumable is False
