"""Unit test: the model-path MCP dispatcher threads ``McpServerSpec.headers``
into ``call_mcp_tool`` as ``spec_headers``.

Drives ``_execute_mcp_tool_async`` directly with the I/O boundaries
(``_tool_lifecycle`` span/sweep machinery, auth resolution, the result-event
append, and the crypto box) mocked, so the assertion is purely about the
``spec_headers=`` kwarg flowing through from the server map.
"""

from __future__ import annotations

import contextlib
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from aios.harness.tool_dispatch import _execute_mcp_tool_async, _ToolCall
from aios.models.agents import McpServerSpec


@contextlib.asynccontextmanager
async def _fake_lifecycle(*_args: Any, **_kwargs: Any) -> Any:
    """Stand-in for ``_tool_lifecycle`` that yields a ready ``_ToolCall`` and
    swallows the span/sweep/event machinery the real one runs."""
    yield _ToolCall(
        call_id="call_1",
        name="mcp__gh__create_issue",
        raw_args="{}",
        bound_log=MagicMock(),
    )


class TestMcpDispatchSpecHeaders:
    async def test_spec_headers_passed_to_call_mcp_tool(self) -> None:
        spec = McpServerSpec(
            name="gh",
            url="https://mcp.github/",
            headers={"X-MCP-Toolsets": "issues"},
        )
        mcp_server_map = {"gh": spec}

        call_mock = AsyncMock(return_value={"content": "ok"})
        with (
            patch("aios.harness.tool_dispatch._tool_lifecycle", _fake_lifecycle),
            patch("aios.harness.tool_dispatch._append_tool_result_event", new_callable=AsyncMock),
            patch("aios.harness.tool_dispatch.runtime.require_crypto_box", return_value=object()),
            # Outbound suppression off (#710): the dispatch path consults this
            # gate before deciding to make the real call; this test exercises
            # the real-dispatch path, so it stays open.
            patch(
                "aios.harness.tool_dispatch._mcp_call_suppressed",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch(
                "aios.mcp.client.resolve_auth_for_target_url",
                new_callable=AsyncMock,
                return_value=(None, {}),
            ),
            patch("aios.mcp.client.call_mcp_tool", call_mock),
        ):
            await _execute_mcp_tool_async(
                MagicMock(),
                "sess_x",
                {"id": "call_1", "function": {"name": "mcp__gh__create_issue", "arguments": "{}"}},
                mcp_server_map,
                account_id="acc_test_stub",
            )

        call_mock.assert_awaited_once()
        assert call_mock.await_args is not None
        assert call_mock.await_args.kwargs.get("spec_headers") == {"X-MCP-Toolsets": "issues"}
        # The resolved URL comes from the spec, not a bare string map value.
        assert call_mock.await_args.args[0] == "https://mcp.github/"

    async def test_second_tool_error_in_user_turn_emits_loop_warning(self) -> None:
        spec = McpServerSpec(name="kine", url="https://mcp.kine.test/")
        bound_log = MagicMock()

        @contextlib.asynccontextmanager
        async def lifecycle(*_args: Any, **_kwargs: Any) -> Any:
            yield _ToolCall(
                call_id="call_2",
                name="mcp__kine__propose_workout",
                raw_args="{}",
                bound_log=bound_log,
            )

        conn = MagicMock()
        acquire = MagicMock()
        acquire.__aenter__ = AsyncMock(return_value=conn)
        acquire.__aexit__ = AsyncMock(return_value=None)
        pool = MagicMock()
        pool.acquire.return_value = acquire

        with (
            patch("aios.harness.tool_dispatch._tool_lifecycle", lifecycle),
            patch(
                "aios.harness.tool_dispatch._append_tool_result_event", new_callable=AsyncMock
            ) as append_result,
            patch("aios.harness.tool_dispatch.runtime.require_crypto_box", return_value=object()),
            patch(
                "aios.harness.tool_dispatch._mcp_call_suppressed",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch(
                "aios.mcp.client.resolve_auth_for_target_url",
                new_callable=AsyncMock,
                return_value=(None, {}),
            ),
            patch(
                "aios.mcp.client.call_mcp_tool",
                new_callable=AsyncMock,
                return_value={
                    "error": '{"error":"invalid","code":"missing_required"}',
                    "code": "tool_error",
                },
            ),
            patch(
                "aios.db.queries.tool_error_counts_since_last_user",
                new_callable=AsyncMock,
                return_value=(2, 2, 3),
            ) as count_errors,
        ):
            await _execute_mcp_tool_async(
                pool,
                "sess_x",
                {
                    "id": "call_2",
                    "function": {"name": "mcp__kine__propose_workout", "arguments": "{}"},
                },
                {"kine": spec},
                account_id="acc_test_stub",
            )

        count_errors.assert_awaited_once_with(
            conn,
            "sess_x",
            "mcp__kine__propose_workout",
            "missing_required",
            account_id="acc_test_stub",
        )
        assert append_result.await_args is not None
        appended = append_result.await_args.args[3]
        assert appended["metadata"] == {"mcp_error_code": "missing_required"}
        rejection_log = bound_log.bind.return_value
        bound_log.bind.assert_called_once_with(
            pair_rejection_count=2,
            tool_rejection_count=2,
            turn_rejection_count=3,
            error_code="missing_required",
        )
        rejection_log.info.assert_called_once_with("mcp_tool.rejected")
        rejection_log.warning.assert_called_once_with("mcp_tool.rejection_loop")

    async def test_unknown_server_bails(self) -> None:
        """A tool naming a server absent from the map raises ``ToolBail`` —
        the spec lookup replaced the old ``url is None`` guard."""
        from aios.tools.invoke import ToolBail

        captured: dict[str, Any] = {}

        @contextlib.asynccontextmanager
        async def _capture_lifecycle(*_args: Any, **_kwargs: Any) -> Any:
            tc = _ToolCall(
                call_id="call_1",
                name="mcp__missing__do",
                raw_args="{}",
                bound_log=MagicMock(),
            )
            try:
                yield tc
            except ToolBail as err:
                captured["bail"] = str(err)

        with (
            patch("aios.harness.tool_dispatch._tool_lifecycle", _capture_lifecycle),
            patch("aios.harness.tool_dispatch.runtime.require_crypto_box", return_value=object()),
        ):
            await _execute_mcp_tool_async(
                MagicMock(),
                "sess_x",
                {"id": "call_1", "function": {"name": "mcp__missing__do", "arguments": "{}"}},
                {},
                account_id="acc_test_stub",
            )

        assert "missing" in captured.get("bail", "")


class TestMcpDispatchSuppression:
    """When the suppression gate (#710) fires, the model-path dispatcher
    synthesizes a success, records the audit event, and never calls
    ``call_mcp_tool``."""

    async def test_suppressed_call_synthesizes_and_records(self) -> None:
        spec = McpServerSpec(name="gh", url="https://mcp.github/")
        mcp_server_map = {"gh": spec}

        call_mock = AsyncMock(return_value={"content": "ok"})
        record_mock = AsyncMock()
        with (
            patch("aios.harness.tool_dispatch._tool_lifecycle", _fake_lifecycle),
            patch("aios.harness.tool_dispatch._append_tool_result_event", new_callable=AsyncMock),
            patch(
                "aios.harness.tool_dispatch._mcp_call_suppressed",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "aios.services.outbound_suppression.record_mcp_suppression",
                record_mock,
            ),
            patch("aios.mcp.client.call_mcp_tool", call_mock),
        ):
            await _execute_mcp_tool_async(
                MagicMock(),
                "sess_x",
                {
                    "id": "call_1",
                    "function": {
                        "name": "mcp__gh__create_issue",
                        "arguments": "{}",
                    },
                },
                mcp_server_map,
                account_id="acc_test_stub",
            )

        call_mock.assert_not_awaited()  # no real MCP round-trip
        record_mock.assert_awaited_once()
        assert record_mock.await_args is not None
        kwargs = record_mock.await_args.kwargs
        assert kwargs["server_name"] == "gh"
        assert kwargs["tool_name"] == "create_issue"
        # arguments come from the lifecycle-yielded _ToolCall.raw_args ("{}").
        assert kwargs["arguments"] == {}
