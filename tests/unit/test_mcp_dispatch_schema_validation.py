"""Unit tests: MCP dispatch validates arguments against the cached tool schema
before ``call_mcp_tool``, matching builtins' ``validate_arguments`` path.

Drives ``_execute_mcp_tool_async`` with the same I/O-boundary mocks as
``test_mcp_dispatch_spec_headers`` so the assertions are about admission
(schema miss / hit / ToolBail) and log fields, not the span/sweep machinery.
"""

from __future__ import annotations

import contextlib
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from aios.harness import runtime
from aios.harness.tool_dispatch import (
    _execute_mcp_tool_async,
    _mcp_error_log_fields,
    _ToolCall,
)
from aios.mcp.client import _headers_key
from aios.mcp.pool import McpSessionPool
from aios.mcp.schema import make_function_tool
from aios.models.agents import McpServerSpec
from aios.tools.invoke import ToolBail

_URL = "https://mcp.kine.test/"
_QUALIFIED = "mcp__kine__propose_workout"
_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "athlete_id": {"type": "string"},
        "draft": {"type": "object"},
    },
    "required": ["athlete_id", "draft"],
    "additionalProperties": False,
}

# Coach / TrainIQ shape: nested draft.blocks[].type is what
# ``invalid_draft_structure`` is about. Validation must name that path.
_DRAFT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "athlete_id": {"type": "string"},
        "draft": {
            "type": "object",
            "properties": {
                "blocks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"type": {"type": "string"}},
                        "required": ["type"],
                    },
                },
            },
            "required": ["blocks"],
        },
    },
    "required": ["athlete_id", "draft"],
}

_REF_DRAFT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "$defs": {
        "block": {
            "type": "object",
            "properties": {"type": {"type": "string"}},
            "required": ["type"],
        }
    },
    "properties": {
        "athlete_id": {"type": "string"},
        "draft": {
            "type": "object",
            "properties": {
                "blocks": {"type": "array", "items": {"$ref": "#/$defs/block"}},
            },
            "required": ["blocks"],
        },
    },
    "required": ["athlete_id", "draft"],
}


def _openai_tool(qualified: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": qualified, "description": "", "parameters": parameters},
    }


@contextlib.asynccontextmanager
async def _lifecycle_for(raw_args: str, bound_log: Any | None = None) -> Any:
    yield _ToolCall(
        call_id="call_1",
        name=_QUALIFIED,
        raw_args=raw_args,
        bound_log=bound_log or MagicMock(),
    )


@contextlib.asynccontextmanager
async def _capturing_lifecycle(raw_args: str, captured: dict[str, Any]) -> Any:
    tc = _ToolCall(
        call_id="call_1",
        name=_QUALIFIED,
        raw_args=raw_args,
        bound_log=MagicMock(),
    )
    try:
        yield tc
    except ToolBail as err:
        captured["bail"] = str(err)


def _spec() -> McpServerSpec:
    return McpServerSpec(name="kine", url=_URL)


def _install_cached_schema(parameters: dict[str, Any]) -> McpSessionPool:
    pool = McpSessionPool()
    pool.set_cached_tools(
        _URL,
        "v",
        _headers_key(None),
        "agt_1:3",
        [_openai_tool(_QUALIFIED, parameters)],
        None,
    )
    runtime.mcp_session_pool = pool
    return pool


def _as_lifecycle(inner: Any) -> Any:
    """Adapt a no-arg async context manager to the ``_tool_lifecycle`` signature."""

    @contextlib.asynccontextmanager
    async def lifecycle(*_args: Any, **_kwargs: Any) -> Any:
        async with inner as tc:
            yield tc

    return lifecycle


async def _dispatch(
    *,
    raw_args: str,
    lifecycle: Any,
    call_mock: AsyncMock | None = None,
    quota_mock: AsyncMock | None = None,
) -> AsyncMock:
    call = call_mock or AsyncMock(return_value={"content": "ok"})
    quota = quota_mock or AsyncMock(return_value=MagicMock(refusal=None, reservation_id=None))
    conn = MagicMock()
    acquire = MagicMock()
    acquire.__aenter__ = AsyncMock(return_value=conn)
    acquire.__aexit__ = AsyncMock(return_value=None)
    pg_pool = MagicMock()
    pg_pool.acquire.return_value = acquire
    with (
        patch("aios.harness.tool_dispatch._tool_lifecycle", lifecycle),
        patch("aios.harness.tool_dispatch._append_tool_result_event", new_callable=AsyncMock),
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
        patch("aios.mcp.client.call_mcp_tool", call),
        patch("aios.services.outbound_tool_quota.reserve_outbound_tool_quota", quota),
        patch(
            "aios.db.queries.tool_error_counts_since_last_user",
            new_callable=AsyncMock,
            return_value=(0, 0, 0),
        ),
    ):
        await _execute_mcp_tool_async(
            pg_pool,
            "sess_x",
            {
                "id": "call_1",
                "function": {"name": _QUALIFIED, "arguments": raw_args},
            },
            {"kine": _spec()},
            account_id="acc_test_stub",
        )
    return call


class TestMcpDispatchSchemaValidation:
    def setup_method(self) -> None:
        self._prior_pool = runtime.mcp_session_pool
        runtime.mcp_session_pool = None

    def teardown_method(self) -> None:
        runtime.mcp_session_pool = self._prior_pool

    async def test_invalid_args_never_call_mcp_and_include_paths(self) -> None:
        _install_cached_schema(_SCHEMA)
        captured: dict[str, Any] = {}
        quota = AsyncMock()
        call = await _dispatch(
            raw_args='{"athlete_id": 123}',
            lifecycle=_as_lifecycle(_capturing_lifecycle('{"athlete_id": 123}', captured)),
            quota_mock=quota,
        )

        call.assert_not_awaited()
        quota.assert_not_awaited()
        bail = captured.get("bail", "")
        assert "athlete_id" in bail
        assert "draft" in bail
        assert "required" in bail.lower() or "type" in bail.lower()
        assert "at athlete_id" in bail or "at <root>" in bail or "at draft" in bail

    async def test_valid_args_call_mcp(self) -> None:
        _install_cached_schema(_SCHEMA)
        raw = json.dumps({"athlete_id": "ath_1", "draft": {}})
        call = await _dispatch(raw_args=raw, lifecycle=_as_lifecycle(_lifecycle_for(raw)))
        call.assert_awaited_once()
        assert call.await_args is not None
        assert call.await_args.args[3] == "propose_workout"
        assert call.await_args.args[4] == {"athlete_id": "ath_1", "draft": {}}

    async def test_missing_schema_fails_open(self) -> None:
        """No cache entry → warn and still call MCP (don't block Coach)."""
        runtime.mcp_session_pool = McpSessionPool()
        bound_log = MagicMock()
        raw = '{"athlete_id": 123}'
        call = await _dispatch(
            raw_args=raw,
            lifecycle=_as_lifecycle(_lifecycle_for(raw, bound_log)),
        )
        call.assert_awaited_once()
        bound_log.warning.assert_called_once_with("mcp_tool.schema_cache_miss", tool=_QUALIFIED)

    async def test_no_pool_fails_open(self) -> None:
        runtime.mcp_session_pool = None
        raw = '{"athlete_id": 123}'
        call = await _dispatch(raw_args=raw, lifecycle=_as_lifecycle(_lifecycle_for(raw)))
        call.assert_awaited_once()

    async def test_validates_sanitized_make_function_tool_schema(self) -> None:
        """The cache stores ``make_function_tool`` envelopes; lookup must use
        that sanitized ``parameters`` dict, not a divergent copy."""
        tool = MagicMock()
        tool.name = "propose_workout"
        tool.description = ""
        tool.inputSchema = _SCHEMA
        tool.outputSchema = None
        envelope = make_function_tool(_QUALIFIED, tool)
        pool = McpSessionPool()
        pool.set_cached_tools(_URL, "v", _headers_key(None), "agt_1:3", [envelope], None)
        runtime.mcp_session_pool = pool

        captured: dict[str, Any] = {}
        call = await _dispatch(
            raw_args="{}",
            lifecycle=_as_lifecycle(_capturing_lifecycle("{}", captured)),
        )
        call.assert_not_awaited()
        assert "athlete_id" in captured.get("bail", "")
        assert "draft" in captured.get("bail", "")

    async def test_nested_draft_path_errors_never_call_mcp(self) -> None:
        """Missing ``draft.blocks[0].type`` is a path-level ToolBail before MCP."""
        _install_cached_schema(_DRAFT_SCHEMA)
        raw = json.dumps({"athlete_id": "ath_1", "draft": {"blocks": [{}]}})
        captured: dict[str, Any] = {}
        quota = AsyncMock()
        call = await _dispatch(
            raw_args=raw,
            lifecycle=_as_lifecycle(_capturing_lifecycle(raw, captured)),
            quota_mock=quota,
        )

        call.assert_not_awaited()
        quota.assert_not_awaited()
        bail = captured.get("bail", "")
        assert "at draft.blocks.0.type" in bail
        assert "required" in bail.lower()

    async def test_ref_schema_reports_nested_path(self) -> None:
        """TrainIQ-style ``$ref`` / ``$defs`` still produce path-level errors."""
        _install_cached_schema(_REF_DRAFT_SCHEMA)
        raw = json.dumps({"athlete_id": "ath_1", "draft": {"blocks": [{}]}})
        captured: dict[str, Any] = {}
        call = await _dispatch(
            raw_args=raw,
            lifecycle=_as_lifecycle(_capturing_lifecycle(raw, captured)),
        )
        call.assert_not_awaited()
        assert "at draft.blocks.0.type" in captured.get("bail", "")

    async def test_validator_exception_fails_open(self) -> None:
        """Uncompilable cached schema must not block Coach or escape admission."""
        _install_cached_schema(_SCHEMA)
        bound_log = MagicMock()
        raw = json.dumps({"athlete_id": "ath_1", "draft": {}})
        with patch(
            "aios.harness.tool_dispatch.validate_arguments",
            side_effect=RuntimeError("schema engine exploded"),
        ):
            call = await _dispatch(
                raw_args=raw,
                lifecycle=_as_lifecycle(_lifecycle_for(raw, bound_log)),
            )
        call.assert_awaited_once()
        bound_log.warning.assert_called_once_with(
            "mcp_tool.schema_validate_failed",
            tool=_QUALIFIED,
            error_type="RuntimeError",
        )


class TestMcpErrorLogFields:
    def test_extracts_nested_code_and_reason(self) -> None:
        result = {
            "error": json.dumps(
                {
                    "error": "invalid",
                    "code": "invalid_draft_structure",
                    "reason": "draft.blocks[0].type is required",
                    "draft": {"huge": "athlete payload that must not be logged"},
                }
            ),
            "code": "tool_error",
        }
        fields = _mcp_error_log_fields(result)
        assert fields["error_code"] == "invalid_draft_structure"
        assert fields["error_reason"] == "draft.blocks[0].type is required"
        assert "draft" not in fields
        assert "huge" not in json.dumps(fields)

    def test_truncates_reason(self) -> None:
        result = {
            "error": json.dumps({"code": "invalid_draft_structure", "reason": "x" * 500}),
            "code": "tool_error",
        }
        fields = _mcp_error_log_fields(result)
        assert fields["error_reason"] == "x" * 200

    def test_falls_back_to_envelope_code(self) -> None:
        fields = _mcp_error_log_fields({"error": "boom", "code": "tool_error"})
        assert fields == {"error_code": "tool_error"}

    async def test_completed_log_includes_extracted_fields(self) -> None:
        prior = runtime.mcp_session_pool
        runtime.mcp_session_pool = None
        bound_log = MagicMock()
        payload = {
            "error": "invalid",
            "code": "invalid_draft_structure",
            "reason": "missing required field",
            "draft": {"sets": [1, 2, 3]},
        }
        try:
            await _dispatch(
                raw_args="{}",
                lifecycle=_as_lifecycle(_lifecycle_for("{}", bound_log)),
                call_mock=AsyncMock(
                    return_value={"error": json.dumps(payload), "code": "tool_error"}
                ),
            )
        finally:
            runtime.mcp_session_pool = prior

        bound_log.info.assert_any_call(
            "mcp_tool.completed",
            is_error=True,
            error_code="invalid_draft_structure",
            error_reason="missing required field",
        )
        completed = [
            c for c in bound_log.info.call_args_list if c.args and c.args[0] == "mcp_tool.completed"
        ]
        assert completed
        completed_kwargs = completed[0].kwargs
        assert "draft" not in completed_kwargs
        assert "sets" not in json.dumps(completed_kwargs)
