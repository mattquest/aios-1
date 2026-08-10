"""Unit tests for the model-path tool dispatch: the ``validate_arguments`` helper and
the ``_classify_tool_error`` eviction policy.

``validate_arguments`` converts JSON Schema failures into a model-readable error string
that lists every problem at once. ``_classify_tool_error`` decides whether a handler
exception is an expected model-visible refusal (clean ``is_error`` result, no eviction)
or a genuine failure (also evicts the sandbox).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock

import pytest

from aios.db import queries
from aios.errors import (
    AiosError,
    ConflictError,
    CryptoDecryptError,
    ForbiddenError,
    NotFoundError,
    RateLimitedError,
    ValidationError,
)
from aios.harness import runtime, tool_dispatch
from aios.harness.tool_dispatch import (
    _classify_tool_error,
    _tool_lifecycle,
    _unoffered_tool_message,
    reject_unoffered_tool_calls,
)
from aios.models.sessions import Err
from aios.services import sessions as sessions_service
from aios.tools.bash import BashArgumentError
from aios.tools.invoke import ToolBail, validate_arguments

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "channel_id": {"type": ["string", "null"]},
    },
    "required": ["channel_id"],
    "additionalProperties": False,
}


class TestValidateArguments:
    def test_valid_arguments_return_none(self) -> None:
        assert validate_arguments({"channel_id": "signal/a/b"}, _SCHEMA) is None
        assert validate_arguments({"channel_id": None}, _SCHEMA) is None

    def test_missing_required_key_is_reported(self) -> None:
        err = validate_arguments({}, _SCHEMA)
        assert err is not None
        assert "'channel_id' is a required property" in err

    def test_unexpected_property_is_reported(self) -> None:
        """The common weak-model failure mode: wrong param name.  The
        extra-property error must surface so the model can correct it.
        """
        err = validate_arguments({"target": "signal/a/b"}, _SCHEMA)
        assert err is not None
        assert "'target' was unexpected" in err

    def test_wrong_type_is_reported(self) -> None:
        err = validate_arguments({"channel_id": 42}, _SCHEMA)
        assert err is not None
        assert "42" in err  # what was sent
        # jsonschema's message varies; just assert we got SOMETHING useful
        assert "channel_id" in err or "type" in err.lower() or "42" in err

    def test_multi_error_accumulation(self) -> None:
        """Missing required + extra property → BOTH surface in a single
        error.  Model sees every issue at once; one retry fixes all.
        """
        err = validate_arguments({"target": "x"}, _SCHEMA)
        assert err is not None
        assert "'channel_id' is a required property" in err
        assert "'target' was unexpected" in err

    def test_passed_arguments_echoed_for_context(self) -> None:
        """The error includes the arguments the model sent so the model
        can see exactly what shape it produced vs. what was expected.
        """
        err = validate_arguments({"target": "oops"}, _SCHEMA)
        assert err is not None
        assert '"target": "oops"' in err

    def test_error_ends_with_hint_to_consult_schema(self) -> None:
        """Closing guidance points the model back at the tool's
        declared ``parameters`` — its authoritative reference."""
        err = validate_arguments({"target": "x"}, _SCHEMA)
        assert err is not None
        assert "parameters" in err.lower()

    def test_permissive_schema_allows_anything(self) -> None:
        """A schema with no constraints (no required, no additionalProps
        false) accepts arbitrary payloads — validation is opt-in per
        tool."""
        loose: dict[str, Any] = {"type": "object"}
        assert validate_arguments({"anything": "goes"}, loose) is None
        assert validate_arguments({}, loose) is None


class _InternalAiosError(AiosError):
    error_type = "internal_thing"
    status_code = 500


class TestClassifyToolError:
    @pytest.mark.parametrize(
        ("err", "evict", "message", "detail"),
        [
            # Expected, model-visible refusals — never evict the sandbox. The headline
            # fix: a 400 *ArgumentError (BashArgumentError) is a clean refusal now,
            # where it used to evict the container along with every other AiosError.
            (ToolBail("bad args"), False, "bad args", {}),
            # #1680: a ToolBail's structured detail rides the third tuple slot so the
            # single writer can merge it into the {"error": msg} content (edit/write's
            # {path, matches} context).
            (
                ToolBail("no match", detail={"path": "/p", "matches": 3}),
                False,
                "no match",
                {"path": "/p", "matches": 3},
            ),
            (
                BashArgumentError("command must be non-empty"),
                False,
                "command must be non-empty",
                {},
            ),
            (ForbiddenError("denied", detail={"x": 1}), False, 'denied ({"x": 1})', {}),
            (NotFoundError("no such workflow"), False, "no such workflow", {}),
            (ConflictError("stale version"), False, "stale version", {}),
            (RateLimitedError("slow down"), False, "slow down", {}),
            (ValidationError("bad field"), False, "bad field", {}),
            # Genuine failures — also evict the sandbox.
            (_InternalAiosError("internal boom"), True, "internal boom", {}),
            # A 5xx WITH detail still gets the json-suffixed message (the detail-format
            # branch is independent of the evict decision). An AiosError folds its own
            # detail into the message, so the third slot stays empty.
            (_InternalAiosError("boom", detail={"k": 2}), True, 'boom ({"k": 2})', {}),
            (CryptoDecryptError("decrypt failed"), True, "decrypt failed", {}),
            (ValueError("oops"), True, "ValueError: oops", {}),
            (RuntimeError("kaboom"), True, "RuntimeError: kaboom", {}),
        ],
    )
    def test_classification(
        self, err: BaseException, evict: bool, message: str, detail: dict[str, Any]
    ) -> None:
        assert _classify_tool_error(err) == (evict, message, detail)


class TestToolLifecycleEviction:
    """End-to-end: the classifier wired into ``_tool_lifecycle`` — a refusal appends a
    clean is_error result without evicting; a genuine failure also evicts."""

    @staticmethod
    async def _drive(monkeypatch: Any, err: BaseException) -> tuple[list[str], Any]:
        """Run a handler that raises ``err`` through the lifecycle; the DB writes are
        stubbed. Returns ``(evict_calls, append_result_mock)``."""
        monkeypatch.setattr(
            sessions_service,
            "append_event",
            AsyncMock(return_value=MagicMock(id="span_1")),
        )
        append_result = AsyncMock()
        monkeypatch.setattr(tool_dispatch, "_append_tool_result", append_result)
        monkeypatch.setattr(tool_dispatch, "_trigger_sweep", AsyncMock())

        evicted: list[str] = []
        call = {"id": "tc_1", "function": {"name": "demo", "arguments": "{}"}}
        async with _tool_lifecycle(
            MagicMock(),
            "ses_1",
            call,
            account_id="acc_1",
            log_prefix="tool",
            on_exception=evicted.append,
        ):
            raise err
        return evicted, append_result  # type: ignore[unreachable]

    async def test_client_error_refusal_does_not_evict(self, monkeypatch: Any) -> None:
        evicted, append_result = await self._drive(
            monkeypatch, ForbiddenError("denied", detail={"x": 1})
        )
        assert evicted == []  # a 4xx refusal must NOT evict the sandbox
        assert append_result.await_count == 1  # but still appends a model-visible result
        assert append_result.await_args.kwargs["error"] == 'denied ({"x": 1})'

    async def test_toolbail_does_not_evict(self, monkeypatch: Any) -> None:
        evicted, append_result = await self._drive(monkeypatch, ToolBail("bad json"))
        assert evicted == []
        assert append_result.await_args.kwargs["error"] == "bad json"

    async def test_toolbail_detail_flows_to_the_single_writer(self, monkeypatch: Any) -> None:
        # #1680: a migrated handler's structured context (edit/write's {path, matches})
        # rides ToolBail.detail through the lifecycle into _append_tool_result, which
        # merges it into the {"error": msg} event content.
        evicted, append_result = await self._drive(
            monkeypatch, ToolBail("no match", detail={"path": "/p", "matches": 3})
        )
        assert evicted == []
        assert append_result.await_args.kwargs["error"] == "no match"
        assert append_result.await_args.kwargs["detail"] == {"path": "/p", "matches": 3}

    async def test_server_error_evicts(self, monkeypatch: Any) -> None:
        evicted, append_result = await self._drive(monkeypatch, CryptoDecryptError("boom"))
        assert evicted == ["ses_1"]  # a 5xx is a genuine failure → evict
        assert append_result.await_count == 1

    async def test_unexpected_exception_evicts(self, monkeypatch: Any) -> None:
        evicted, append_result = await self._drive(monkeypatch, ValueError("oops"))
        assert evicted == ["ses_1"]
        assert append_result.await_args.kwargs["error"] == "ValueError: oops"


class TestUnofferedToolMessage:
    """The #1773 defect-2 rejection wording: short + imperative, names the
    currently-offered tools."""

    def test_names_the_call_and_the_offered_set(self) -> None:
        msg = _unoffered_tool_message("return", ["bash", "read"])
        assert "return" in msg
        assert "not offered" in msg
        assert "bash" in msg and "read" in msg
        assert "do not call it" in msg

    def test_empty_offered_set_is_worded_clearly(self) -> None:
        msg = _unoffered_tool_message("return", [])
        assert "return" in msg
        assert "none offered right now" in msg


class TestRejectUnofferedToolCalls:
    """``reject_unoffered_tool_calls`` — the #1773 defect-2 harness invariant: a call
    naming a tool absent from the step's frozen offered set is rejected through the
    exact same lifecycle bracket a real dispatch uses (span pair, dedup-guarded
    ``is_error=True`` result, tail sweep) — it just never reaches a handler.
    """

    async def _drive(
        self,
        monkeypatch: Any,
        tool_calls: list[dict[str, Any]],
        offered_names: list[str],
        *,
        closed: tuple[Any, Any] | None = None,
        get_closed_request: AsyncMock | None = None,
    ) -> Any:
        monkeypatch.setattr(
            sessions_service,
            "append_event",
            AsyncMock(return_value=MagicMock(id="span_1")),
        )
        append_result = AsyncMock()
        monkeypatch.setattr(tool_dispatch, "_append_tool_result", append_result)
        monkeypatch.setattr(tool_dispatch, "_trigger_sweep", AsyncMock())
        from aios.harness.inflight_tool_registry import InflightToolRegistry

        monkeypatch.setattr(runtime, "require_inflight_tool_registry", InflightToolRegistry)

        # A de-offered `return`/`error` classifies by DURABLE request state (#1789),
        # exactly like the return/error handler: it resolves the request_id via
        # ``get_closed_request`` over a pooled connection. Mock both so the rejection
        # path can distinguish a genuinely-closed request (``closed`` is a tuple) from a
        # session that never owned one (``closed=None`` → the generic rejection).
        conn = MagicMock()
        acquire = MagicMock()
        acquire.__aenter__ = AsyncMock(return_value=conn)
        acquire.__aexit__ = AsyncMock(return_value=False)
        pool = MagicMock()
        pool.acquire = MagicMock(return_value=acquire)
        monkeypatch.setattr(runtime, "require_pool", lambda: pool)
        gcr = (
            get_closed_request if get_closed_request is not None else AsyncMock(return_value=closed)
        )
        monkeypatch.setattr(queries, "get_closed_request", gcr)

        reject_unoffered_tool_calls(
            MagicMock(),
            "ses_1",
            tool_calls,
            offered_names=offered_names,
            account_id="acc_1",
        )
        # The tasks are launched fire-and-forget; give the loop a beat to run them.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        return append_result

    async def test_closed_control_call_gets_terminal_stop_message(self, monkeypatch: Any) -> None:
        # A de-offered `return` answering a GENUINELY-CLOSED request (durable state
        # resolves a real closed request) → defect-1's eval-validated terminal stop.
        call = {
            "id": "tc_1",
            "function": {"name": "return", "arguments": '{"request_id": "req_1", "value": 1}'},
        }
        closed = (Err(error={"kind": "timeout"}), datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC))
        append_result = await self._drive(monkeypatch, [call], ["bash", "read"], closed=closed)
        assert append_result.await_count == 1
        args = append_result.await_args.args
        kwargs = append_result.await_args.kwargs
        assert args[1] == "ses_1"  # session_id
        assert args[3] == "return"  # tool name
        error = kwargs["error"]
        assert "already answered" in error
        assert "end your turn" in error
        assert "deadline timeout" in error  # the eval-validated handler wording, verbatim
        assert "tool not offered" not in error

    async def test_hallucinated_control_name_without_closed_request_gets_generic(
        self, monkeypatch: Any
    ) -> None:
        """#1789 regression: the state-classification the seat gate demanded.

        A session that never owned a request can also hallucinate the literal name
        ``return``/``error`` — those names are unoffered for it too. With no closed
        request resolving (``get_closed_request`` → ``None``), it must get the GENERIC
        unoffered rejection, NOT the factual falsehood ``this request was already
        answered``. The pre-fix name-only branch failed exactly here.
        """
        call = {
            "id": "tc_1",
            "function": {"name": "return", "arguments": '{"request_id": "req_ghost", "value": 1}'},
        }
        append_result = await self._drive(monkeypatch, [call], ["bash", "read"], closed=None)
        error = append_result.await_args.kwargs["error"]
        assert "tool not offered" in error
        assert "return" in error
        assert "bash" in error and "read" in error
        assert "already answered" not in error
        assert "end your turn" not in error

    async def test_de_offered_error_name_with_no_request_id_gets_generic(
        self, monkeypatch: Any
    ) -> None:
        # Pure name hallucination with no request_id at all — resolved to the generic
        # rejection (never the closed-request lie), and without even a state lookup.
        call = {"id": "tc_1", "function": {"name": "error", "arguments": "{}"}}
        gcr = AsyncMock(return_value=None)
        append_result = await self._drive(monkeypatch, [call], ["bash"], get_closed_request=gcr)
        error = append_result.await_args.kwargs["error"]
        assert "tool not offered" in error
        assert "error" in error
        assert "already answered" not in error
        # No request_id → short-circuits before the DB read: get_closed_request is
        # never called (the whole point — no state lookup for a nameless hallucination).
        gcr.assert_not_called()

    async def test_hallucinated_name_keeps_generic_unoffered_message(
        self, monkeypatch: Any
    ) -> None:
        call = {"id": "tc_1", "function": {"name": "hallucinated", "arguments": "{}"}}
        append_result = await self._drive(monkeypatch, [call], ["bash", "read"])
        error = append_result.await_args.kwargs["error"]
        assert "tool not offered" in error
        assert "hallucinated" in error
        assert "bash" in error and "read" in error
        assert "end your turn" not in error

    async def test_never_reaches_a_handler(self, monkeypatch: Any) -> None:
        """The tool's own handler must not run — only the rejection path does."""
        call = {"id": "tc_1", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}
        invoke_builtin = AsyncMock(return_value=MagicMock())
        monkeypatch.setattr(tool_dispatch, "invoke_builtin", invoke_builtin)
        await self._drive(monkeypatch, [call], ["read"])
        invoke_builtin.assert_not_called()

    async def test_per_call_granularity(self, monkeypatch: Any) -> None:
        """Only the unoffered calls in the batch are rejected — each independently."""
        calls = [
            {"id": "tc_1", "function": {"name": "return", "arguments": "{}"}},
            {"id": "tc_2", "function": {"name": "error", "arguments": "{}"}},
        ]
        append_result = await self._drive(monkeypatch, calls, [])
        assert append_result.await_count == 2
        names = {c.args[3] for c in append_result.await_args_list}
        assert names == {"return", "error"}


class TestCancelDuringStartSpan:
    """A (single) cancel landing on the ``tool_execute_start`` append — which happens inside
    ``__aenter__``, before the main try/finally (the pg-notify interrupt listener and a
    graceful drain both deliver one) — must still resolve the call in-task:
    a clean ``cancelled`` result + a tail sweep, with the body never running. Without the
    targeted handler the task would escape with no result and no sweep, stranding the call
    until the ≤30s ghost sweep and then misclassifying it as 'may have completed'. (A *second*
    cancel interrupting the cleanup awaits falls back to that ghost sweep — same exposure as
    the in-body cancel handler — so it is the documented backstop, not pinned here.)"""

    async def test_cancel_during_start_span_appends_cancelled_and_sweeps(
        self, monkeypatch: Any
    ) -> None:
        span_committed = asyncio.Event()
        release = asyncio.Event()

        async def fake_append_event(
            _pool: Any, _session_id: str, _kind: str, data: dict[str, Any], **_kw: Any
        ) -> Any:
            # Model the start-span write: signal it committed server-side, then hold the
            # client await open (suspended, cancellable) so the cancel lands right here.
            if data.get("event") == "tool_execute_start":
                span_committed.set()
                await release.wait()
            return MagicMock(id="span_1")

        monkeypatch.setattr(sessions_service, "append_event", fake_append_event)
        append_result: Any = AsyncMock()
        trigger_sweep = AsyncMock()
        monkeypatch.setattr(tool_dispatch, "_append_tool_result", append_result)
        monkeypatch.setattr(tool_dispatch, "_trigger_sweep", trigger_sweep)

        body_ran = asyncio.Event()
        call = {"id": "tc_1", "function": {"name": "demo", "arguments": "{}"}}

        async def driver() -> None:
            async with _tool_lifecycle(
                MagicMock(), "ses_1", call, account_id="acc_1", log_prefix="tool"
            ):
                body_ran.set()  # must NOT happen — the cancel struck during the start span

        task = asyncio.create_task(driver())
        await span_committed.wait()  # span "committed"; the lifecycle is suspended on it
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert not body_ran.is_set()  # the body provably never ran
        assert append_result.await_count == 1  # resolved in-task, not left to the sweep
        assert append_result.await_args.kwargs["error"] == "cancelled"
        assert trigger_sweep.await_count == 1  # tail sweep fired so the step wakes now

    async def test_cancel_during_start_span_drops_archived_sweep_tail_and_reraises_cancel(
        self, monkeypatch: Any
    ) -> None:
        # The archive-fence twin of the above, on the path the ``TestArchiveFenceRace`` cancel
        # test cannot reach: the cancel struck the ``tool_execute_start`` append (inside
        # ``__aenter__``), so this arm owns its OWN tail sweep — OUTSIDE the body try/finally
        # that fences the normal-path tails. Model "archival winning BEFORE the sweep tail":
        # the fenced cancelled-result append lands (session still live at that instant), THEN
        # Phase B archives, so the tail sweep's ``sweep_start`` append hits the archived fence
        # (``NotFoundError``). The fence MUST be a clean drop — the ``CancelledError`` still
        # propagates and is never replaced by the sweep's ``NotFoundError``.
        #
        # Fail-before / pass-after: with the pre-fix unfenced ``await _trigger_sweep(...)`` tail
        # the sweep's ``NotFoundError`` propagated in place of the cancel and this
        # ``pytest.raises(asyncio.CancelledError)`` failed (it saw ``NotFoundError``).
        span_committed = asyncio.Event()
        release = asyncio.Event()

        async def fake_append_event(
            _pool: Any, _session_id: str, _kind: str, data: dict[str, Any], **_kw: Any
        ) -> Any:
            if data.get("event") == "tool_execute_start":
                span_committed.set()
                await release.wait()  # hold the client await open so the cancel lands here
            return MagicMock(id="span_1")

        monkeypatch.setattr(sessions_service, "append_event", fake_append_event)
        # The fenced cancelled-result append succeeds (archival has not yet won at this point).
        append_result = AsyncMock()
        monkeypatch.setattr(tool_dispatch, "_append_tool_result", append_result)
        # ...then archival wins: the tail sweep races the flip and hits the archived fence.
        trigger_sweep = AsyncMock(side_effect=NotFoundError("archived fence"))
        monkeypatch.setattr(tool_dispatch, "_trigger_sweep", trigger_sweep)
        # The session is archived, so the fence resolves to a clean drop (not a live 404).
        monkeypatch.setattr(
            sessions_service, "load_live_session_account_id", AsyncMock(return_value=None)
        )

        body_ran = asyncio.Event()
        call = {"id": "tc_1", "function": {"name": "demo", "arguments": "{}"}}

        async def driver() -> None:
            async with _tool_lifecycle(
                MagicMock(), "ses_1", call, account_id="acc_1", log_prefix="tool"
            ):
                body_ran.set()  # must NOT happen — the cancel struck during the start span

        task = asyncio.create_task(driver())
        await span_committed.wait()  # span "committed"; the lifecycle is suspended on it
        task.cancel()
        with pytest.raises(asyncio.CancelledError):  # the cancel propagates, NOT the fence
            await task

        assert not body_ran.is_set()  # the body provably never ran
        assert append_result.await_count == 1  # cancelled-result append attempted (fenced)
        assert trigger_sweep.await_count == 1  # tail sweep fired and hit the archived fence


class TestArchiveFenceRace:
    """Edge-owned-lifetime P1 (#1823 C2): once Phase B archives the session, ANY result
    append in the lifecycle's error/cancel/notfound handlers hits the archived fence and
    ``append_event`` raises ``NotFoundError``. That fence is a CLEAN DROP (the work is gone
    because the session is gone), NOT an error to escape.

    Before the consolidated fence, only a ``NotFoundError`` from the yielded tool *body*
    entered the liveness arm; a fence hit by the CANCEL append, the GENERIC-exception
    append, or the notfound-arm's own replacement append (a TOCTOU: live at the check,
    archived before the append) escaped instead. Every handler append now routes through
    ``_append_result_fenced``, so the class is closed — while a real live-session
    ``NotFoundError`` still propagates, and cancellation is never swallowed."""

    @staticmethod
    def _stub(monkeypatch: Any, *, append_raises: BaseException, live_account: Any) -> AsyncMock:
        """Stub the lifecycle's DB touchpoints. ``append_tool_result`` raises
        ``append_raises`` (the archived fence, or a live-session 404).
        ``load_live_session_account_id`` returns ``live_account`` — ``None`` models an
        archived session (clean drop), a truthy id a still-live one (real error), and a
        list is a per-call sequence (the notfound-arm TOCTOU: live at the check, archived
        before the replacement append). Returns the ``_append_tool_result`` mock."""
        monkeypatch.setattr(
            sessions_service, "append_event", AsyncMock(return_value=MagicMock(id="span_1"))
        )
        append_result = AsyncMock(side_effect=append_raises)
        monkeypatch.setattr(tool_dispatch, "_append_tool_result", append_result)
        monkeypatch.setattr(tool_dispatch, "_trigger_sweep", AsyncMock())
        live = (
            AsyncMock(side_effect=live_account)
            if isinstance(live_account, list)
            else AsyncMock(return_value=live_account)
        )
        monkeypatch.setattr(sessions_service, "load_live_session_account_id", live)
        return append_result

    _CALL: ClassVar[dict[str, Any]] = {
        "id": "tc_1",
        "function": {"name": "demo", "arguments": "{}"},
    }

    async def test_generic_exception_after_archive_is_clean_drop(self, monkeypatch: Any) -> None:
        # A handler-failure (generic Exception arm) append racing the flip: the archived
        # fence must produce a clean drop, NOT an escaping NotFoundError.
        append_result = self._stub(
            monkeypatch, append_raises=NotFoundError("archived fence"), live_account=None
        )
        async with _tool_lifecycle(
            MagicMock(), "ses_1", self._CALL, account_id="acc_1", log_prefix="tool"
        ):
            raise ValueError("handler boom")
        # mypy can't see the lifecycle SUPPRESSES the handled exception (the fence drop) —
        # the assert is reached at runtime.
        assert append_result.await_count == 1  # type: ignore[unreachable]  # append attempted, then dropped

    async def test_cancel_after_archive_drops_append_but_reraises_cancel(
        self, monkeypatch: Any
    ) -> None:
        # A cancel landing after archival: the append's archived-fence NotFoundError is
        # dropped, but the CancelledError itself is NEVER swallowed — it re-raises.
        append_result = self._stub(
            monkeypatch, append_raises=NotFoundError("archived fence"), live_account=None
        )
        with pytest.raises(asyncio.CancelledError):
            async with _tool_lifecycle(
                MagicMock(), "ses_1", self._CALL, account_id="acc_1", log_prefix="tool"
            ):
                raise asyncio.CancelledError()
        assert append_result.await_count == 1  # cancelled-result append attempted, then dropped

    async def test_notfound_arm_toctou_flip_is_clean_drop(self, monkeypatch: Any) -> None:
        # The explicit NotFoundError arm sees the session LIVE, then archival wins before
        # its replacement error append — that append's fence must drop cleanly, not escape.
        self._stub(
            monkeypatch,
            append_raises=NotFoundError("archived fence"),
            live_account=["acc_1", None],  # live at the arm's check, archived at the re-check
        )
        async with _tool_lifecycle(
            MagicMock(), "ses_1", self._CALL, account_id="acc_1", log_prefix="tool"
        ):
            raise NotFoundError("no such workflow")
        # reaching here == no escaping NotFoundError

    async def test_live_session_notfound_still_propagates(self, monkeypatch: Any) -> None:
        # The other half of the invariant: a NotFoundError from a STILL-LIVE session is a
        # real error — the fence must NOT swallow it (guards against an over-broad fix).
        self._stub(monkeypatch, append_raises=NotFoundError("real 404"), live_account="acc_1")
        with pytest.raises(NotFoundError):
            async with _tool_lifecycle(
                MagicMock(), "ses_1", self._CALL, account_id="acc_1", log_prefix="tool"
            ):
                raise ValueError("handler boom")


class TestParentChannelThreading:
    """The hot builtin dispatch path threads the parent assistant's focal
    stamp (``parent_focal_at_arrival``) into the tool-result append, so
    ``append_event`` never re-derives the channel under the row lock (#862).
    The error/cancel path omits it (default ``...`` sentinel → pre-tx lookup).
    """

    @staticmethod
    def _stub_lifecycle_deps(monkeypatch: Any) -> None:
        monkeypatch.setattr(
            sessions_service,
            "append_event",
            AsyncMock(return_value=MagicMock(id="span_1")),
        )
        monkeypatch.setattr(tool_dispatch, "_trigger_sweep", AsyncMock())
        # The dispatch preamble (#1903) parses/looks-up/validates before the
        # handler; these tests use synthetic tool names, so stub it to a
        # no-op admit. Individual tests re-override to exercise refusal.
        monkeypatch.setattr(tool_dispatch, "prepare_builtin", lambda *_a: {})

    async def test_quota_refusal_never_invokes_tool_and_is_model_visible(
        self, monkeypatch: Any
    ) -> None:
        from aios.services.outbound_tool_quota import QuotaAdmission

        self._stub_lifecycle_deps(monkeypatch)
        invoke = AsyncMock()
        append_error = AsyncMock()
        monkeypatch.setattr(tool_dispatch, "invoke_builtin", invoke)
        monkeypatch.setattr(tool_dispatch, "_append_tool_result", append_error)
        monkeypatch.setattr(tool_dispatch, "prepare_builtin", lambda *_a: {})
        monkeypatch.setattr(
            "aios.services.outbound_tool_quota.reserve_outbound_tool_quota",
            AsyncMock(
                return_value=QuotaAdmission(
                    refusal="quota_exceeded: telegram_send 100/100 per hour"
                )
            ),
        )

        call = {"id": "tc_1", "function": {"name": "telegram_send", "arguments": "{}"}}
        await tool_dispatch._execute_tool_async(MagicMock(), "ses_1", call, account_id="acc_1")

        invoke.assert_not_awaited()
        assert append_error.await_args is not None
        assert append_error.await_args.kwargs["error"] == (
            "quota_exceeded: telegram_send 100/100 per hour"
        )

    async def test_prepare_refusal_precedes_quota_admission(self, monkeypatch: Any) -> None:
        """A call that would refuse anyway (bad args / unknown tool / schema
        mismatch) is bailed BEFORE quota admission — it must consume no
        outbound dispatch capacity (#1903)."""
        from aios.tools.invoke import ToolBail

        self._stub_lifecycle_deps(monkeypatch)
        invoke = AsyncMock()
        append_error = AsyncMock()
        reserve = AsyncMock()
        monkeypatch.setattr(tool_dispatch, "invoke_builtin", invoke)
        monkeypatch.setattr(tool_dispatch, "_append_tool_result", append_error)

        def _bail(*_a: Any) -> None:
            raise ToolBail("arguments were not valid JSON")

        monkeypatch.setattr(tool_dispatch, "prepare_builtin", _bail)
        monkeypatch.setattr(
            "aios.services.outbound_tool_quota.reserve_outbound_tool_quota", reserve
        )

        call = {"id": "tc_1", "function": {"name": "telegram_send", "arguments": "not json"}}
        await tool_dispatch._execute_tool_async(MagicMock(), "ses_1", call, account_id="acc_1")

        reserve.assert_not_awaited()
        invoke.assert_not_awaited()
        append_call = append_error.await_args
        assert append_call is not None
        assert append_call.kwargs["error"] == "arguments were not valid JSON"

    async def test_admission_store_failure_fails_closed_as_tool_error(
        self, monkeypatch: Any
    ) -> None:
        """A DB failure inside quota admission lands in the lifecycle's
        generic-exception arm: the call resolves with a typed model-visible
        error result, never an unresolved call, and the tool never runs."""
        self._stub_lifecycle_deps(monkeypatch)
        invoke = AsyncMock()
        append_error = AsyncMock()
        monkeypatch.setattr(tool_dispatch, "invoke_builtin", invoke)
        monkeypatch.setattr(tool_dispatch, "_append_tool_result", append_error)
        monkeypatch.setattr(tool_dispatch, "prepare_builtin", lambda *_a: {})
        monkeypatch.setattr(tool_dispatch, "_evict_session_container", lambda _s: None)
        monkeypatch.setattr(
            "aios.services.outbound_tool_quota.reserve_outbound_tool_quota",
            AsyncMock(side_effect=RuntimeError("admission store down")),
        )

        call = {"id": "tc_1", "function": {"name": "telegram_send", "arguments": "{}"}}
        await tool_dispatch._execute_tool_async(MagicMock(), "ses_1", call, account_id="acc_1")

        invoke.assert_not_awaited()
        append_call = append_error.await_args
        assert append_call is not None
        assert "admission store down" in append_call.kwargs["error"]

    async def test_admitted_dispatch_marks_reservation_completed(self, monkeypatch: Any) -> None:
        from aios.services.outbound_tool_quota import QuotaAdmission
        from aios.tools.registry import ToolResult

        self._stub_lifecycle_deps(monkeypatch)
        monkeypatch.setattr(
            tool_dispatch, "invoke_builtin", AsyncMock(return_value=ToolResult(content="ok"))
        )
        monkeypatch.setattr(tool_dispatch, "prepare_builtin", lambda *_a: {})
        monkeypatch.setattr(tool_dispatch, "_append_tool_result_event", AsyncMock())
        mark = AsyncMock()
        monkeypatch.setattr(
            "aios.services.outbound_tool_quota.reserve_outbound_tool_quota",
            AsyncMock(return_value=QuotaAdmission(reservation_id="res_1")),
        )
        monkeypatch.setattr(
            "aios.services.outbound_tool_quota.mark_outbound_dispatch_completed", mark
        )

        call = {"id": "tc_1", "function": {"name": "telegram_send", "arguments": "{}"}}
        await tool_dispatch._execute_tool_async(MagicMock(), "ses_1", call, account_id="acc_1")

        mark.assert_awaited_once()
        mark_call = mark.await_args
        assert mark_call is not None
        assert mark_call.args[1] == "res_1"

    async def test_execute_tool_async_threads_parent_channel(self, monkeypatch: Any) -> None:
        from aios.tools.registry import ToolResult

        self._stub_lifecycle_deps(monkeypatch)
        monkeypatch.setattr(
            tool_dispatch,
            "invoke_builtin",
            AsyncMock(return_value=ToolResult(content="ok")),
        )
        append_event_ev: Any = AsyncMock()
        monkeypatch.setattr(tool_dispatch, "_append_tool_result_event", append_event_ev)

        call = {"id": "tc_1", "function": {"name": "demo", "arguments": "{}"}}
        await tool_dispatch._execute_tool_async(
            MagicMock(),
            "ses_1",
            call,
            account_id="acc_1",
            parent_focal_at_arrival="tg:42",
        )
        assert append_event_ev.await_count == 1
        assert append_event_ev.await_args.kwargs["tool_parent_channel"] == "tg:42"

    async def test_execute_tool_async_threads_none_stamp(self, monkeypatch: Any) -> None:
        from aios.tools.registry import ToolResult

        self._stub_lifecycle_deps(monkeypatch)
        monkeypatch.setattr(
            tool_dispatch,
            "invoke_builtin",
            AsyncMock(return_value=ToolResult(content="ok")),
        )
        append_event_ev: Any = AsyncMock()
        monkeypatch.setattr(tool_dispatch, "_append_tool_result_event", append_event_ev)

        call = {"id": "tc_1", "function": {"name": "demo", "arguments": "{}"}}
        # A non-connector parent stamps focal None — the hot path must still
        # PASS it (None), not fall through to the lookup default.
        await tool_dispatch._execute_tool_async(
            MagicMock(),
            "ses_1",
            call,
            account_id="acc_1",
            parent_focal_at_arrival=None,
        )
        assert append_event_ev.await_args.kwargs["tool_parent_channel"] is None

    async def test_error_path_omits_parent_channel(self, monkeypatch: Any) -> None:
        """A handler raising routes through ``_append_tool_result`` (the
        error/cancel append), which does NOT pass ``tool_parent_channel`` —
        so ``_append_tool_result_event`` keeps the default ``...`` and
        ``append_event`` falls back to the pre-tx parent lookup."""
        self._stub_lifecycle_deps(monkeypatch)
        monkeypatch.setattr(
            tool_dispatch,
            "invoke_builtin",
            AsyncMock(side_effect=ValueError("boom")),
        )
        append_event_ev: Any = AsyncMock()
        monkeypatch.setattr(tool_dispatch, "_append_tool_result_event", append_event_ev)
        monkeypatch.setattr(tool_dispatch, "_evict_session_container", lambda _s: None)

        call = {"id": "tc_1", "function": {"name": "demo", "arguments": "{}"}}
        await tool_dispatch._execute_tool_async(
            MagicMock(),
            "ses_1",
            call,
            account_id="acc_1",
            parent_focal_at_arrival="tg:42",
        )
        assert append_event_ev.await_count == 1
        assert "tool_parent_channel" not in append_event_ev.await_args.kwargs

    async def test_append_tool_result_event_forwards_precomputed_channel(
        self, monkeypatch: Any
    ) -> None:
        """Hot path (#991): ``_append_tool_result_event`` resolves the precompute
        OUTSIDE the lock and forwards it as ``precomputed=`` to
        ``queries.append_event``.  A supplied ``tool_parent_channel`` flows
        through as ``resolved_tool_channel`` with NO DB lookup."""
        from aios.db import queries

        captured: dict[str, Any] = {}

        async def _fake_append_event(conn: Any, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return MagicMock()

        monkeypatch.setattr(queries, "append_event", _fake_append_event)
        monkeypatch.setattr(
            queries,
            "find_tool_result_event",
            AsyncMock(return_value=None),
        )

        conn = MagicMock()
        conn.execute = AsyncMock()
        tx = MagicMock()
        tx.__aenter__ = AsyncMock(return_value=None)
        tx.__aexit__ = AsyncMock(return_value=None)
        conn.transaction = MagicMock(return_value=tx)
        acquire = MagicMock()
        acquire.__aenter__ = AsyncMock(return_value=conn)
        acquire.__aexit__ = AsyncMock(return_value=None)
        pool = MagicMock()
        pool.acquire = MagicMock(return_value=acquire)

        await tool_dispatch._append_tool_result_event(
            pool,
            "ses_1",
            "tc_1",
            {"role": "tool", "tool_call_id": "tc_1", "content": "ok"},
            account_id="acc_1",
            tool_parent_channel="tg:42",
        )
        # The hot path passes ``precomputed=`` (not ``tool_parent_channel=``);
        # its resolved channel echoes the supplied stamp with no DB round-trip.
        assert "tool_parent_channel" not in captured
        assert captured["precomputed"].resolved_tool_channel == "tg:42"


def _mock_pool_conn() -> tuple[Any, Any]:
    """Build a mock asyncpg pool whose ``acquire()`` yields a conn with a
    nesting-safe ``transaction()`` (the inner ``append_event`` SAVEPOINT and
    the outer ``_append_tool_result_event`` transaction both no-op here)."""
    conn = MagicMock()
    conn.execute = AsyncMock()
    tx = MagicMock()
    tx.__aenter__ = AsyncMock(return_value=None)
    tx.__aexit__ = AsyncMock(return_value=None)
    conn.transaction = MagicMock(return_value=tx)
    acquire = MagicMock()
    acquire.__aenter__ = AsyncMock(return_value=conn)
    acquire.__aexit__ = AsyncMock(return_value=None)
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=acquire)
    return pool, conn


class TestToolResultUniqueFloor:
    """The partial UNIQUE index ``events_tool_result_idx`` (#1082) is the
    structural floor under the manual dedup: a racing committed appender that
    slips past the read-check makes ``append_event`` raise
    ``UniqueViolationError``; ``_append_tool_result_event`` must re-read, treat
    it as a dedup-skip, and apply the ``open_tool_call_count`` compensation —
    NOT propagate the raw DB error."""

    async def test_unique_violation_treated_as_dedup_skip(self, monkeypatch: Any) -> None:
        import asyncpg

        from aios.db import queries

        winner = MagicMock()
        winner.data = {"is_error": False}
        # The read-check sees nothing (the racing row committed AFTER it),
        # then the re-read after the violation finds the winning row.
        find = AsyncMock(side_effect=[None, winner])
        monkeypatch.setattr(queries, "find_tool_result_event", find)
        monkeypatch.setattr(
            queries,
            "append_event",
            AsyncMock(side_effect=asyncpg.UniqueViolationError("dup")),
        )
        # The cold path (default ``...``) precomputes off a brief pool.acquire()
        # BEFORE the lock — stub it so the mock conn needs no real query support.
        monkeypatch.setattr(
            queries,
            "precompute_event_append",
            AsyncMock(return_value=queries._PrecomputedAppend(0, None)),
        )
        decrement = AsyncMock()
        monkeypatch.setattr(queries, "decrement_open_tool_call_count", decrement)

        pool, _conn = _mock_pool_conn()

        # Must NOT raise — the late append no-ops.
        await tool_dispatch._append_tool_result_event(
            pool,
            "ses_1",
            "tc_1",
            {"role": "tool", "tool_call_id": "tc_1", "content": "ok"},
            account_id="acc_1",
        )
        assert find.await_count == 2  # read-check + post-violation re-read
        assert decrement.await_count == 1  # id-blind +1 compensation applied once

    async def test_read_check_hit_skips_append(self, monkeypatch: Any) -> None:
        """Fast path: when the read-check already finds the winning row, the
        append is never attempted and the compensation runs once."""
        from aios.db import queries

        winner = MagicMock()
        winner.data = {"is_error": False}
        monkeypatch.setattr(queries, "find_tool_result_event", AsyncMock(return_value=winner))
        append = AsyncMock()
        monkeypatch.setattr(queries, "append_event", append)
        monkeypatch.setattr(
            queries,
            "precompute_event_append",
            AsyncMock(return_value=queries._PrecomputedAppend(0, None)),
        )
        decrement = AsyncMock()
        monkeypatch.setattr(queries, "decrement_open_tool_call_count", decrement)

        pool, _conn = _mock_pool_conn()
        await tool_dispatch._append_tool_result_event(
            pool,
            "ses_1",
            "tc_1",
            {"role": "tool", "tool_call_id": "tc_1", "content": "ok"},
            account_id="acc_1",
        )
        assert append.await_count == 0
        assert decrement.await_count == 1


class TestAppendToolResultUniqueFloor:
    """``services.sessions.append_tool_result`` (#1082): a ``UniqueViolation``
    from the racing-committed-row case re-reads + re-classifies — collapsing
    into idempotent-retry (return existing) or deny-after-success
    (``ConflictError``) exactly as the read-check path does."""

    @staticmethod
    def _patch_common(monkeypatch: Any) -> None:
        from aios.db import queries

        monkeypatch.setattr(queries, "lock_active_session_for_update", AsyncMock(return_value=None))
        monkeypatch.setattr(
            queries, "lookup_tool_name_by_call_id", AsyncMock(return_value=("demo", None))
        )
        # No spill capping in unit tests: return the content unchanged with
        # no attachment record (the within-cap shape).
        from aios.sandbox.tool_result_spill import CappedToolResult

        monkeypatch.setattr(
            "aios.sandbox.tool_result_spill.cap_tool_result_content",
            AsyncMock(
                side_effect=lambda *a, **k: CappedToolResult(content=a[3], variable_name=None)
            ),
        )

    @staticmethod
    def _mock_conn() -> Any:
        conn = MagicMock()
        tx = MagicMock()
        tx.__aenter__ = AsyncMock(return_value=None)
        tx.__aexit__ = AsyncMock(return_value=None)
        conn.transaction = MagicMock(return_value=tx)
        return conn

    async def test_unique_violation_matching_intent_returns_existing(
        self, monkeypatch: Any
    ) -> None:
        import asyncpg

        from aios.db import queries
        from aios.models.events import Event

        self._patch_common(monkeypatch)
        existing = MagicMock(spec=Event)
        existing.data = {"is_error": False}
        monkeypatch.setattr(
            queries, "find_tool_result_event", AsyncMock(side_effect=[None, existing])
        )
        monkeypatch.setattr(
            queries, "append_event", AsyncMock(side_effect=asyncpg.UniqueViolationError("dup"))
        )
        decrement = AsyncMock()
        monkeypatch.setattr(queries, "decrement_open_tool_call_count", decrement)

        result = await sessions_service.append_tool_result(
            self._mock_conn(),
            account_id="acc_1",
            session_id="ses_1",
            tool_call_id="tc_1",
            content="ok",
            is_error=False,
        )
        assert result is existing
        assert decrement.await_count == 1

    async def test_unique_violation_intent_mismatch_raises_conflict(self, monkeypatch: Any) -> None:
        import asyncpg

        from aios.db import queries
        from aios.models.events import Event

        self._patch_common(monkeypatch)
        existing = MagicMock(spec=Event)
        existing.data = {"is_error": False}  # prior SUCCESS already committed
        monkeypatch.setattr(
            queries, "find_tool_result_event", AsyncMock(side_effect=[None, existing])
        )
        monkeypatch.setattr(
            queries, "append_event", AsyncMock(side_effect=asyncpg.UniqueViolationError("dup"))
        )
        monkeypatch.setattr(queries, "decrement_open_tool_call_count", AsyncMock())

        # Late DENY (is_error=True) against a committed SUCCESS — too-late.
        with pytest.raises(ConflictError):
            await sessions_service.append_tool_result(
                self._mock_conn(),
                account_id="acc_1",
                session_id="ses_1",
                tool_call_id="tc_1",
                content="denied",
                is_error=True,
            )
