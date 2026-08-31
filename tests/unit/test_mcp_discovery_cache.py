"""Unit tests for MCP discovery caching + per-server timeout / circuit breaker (#1391).

A single unresponsive MCP server used to stall the whole turn prelude: discovery
re-polled ``list_tools()`` live every step (no result cache) with no per-server
timeout, so one hung server starved the turn until the 960s step budget fired.

These tests cover the fix:

* the pool's ``(tools, instructions)`` result cache, keyed on transport identity
  + binding identity, so a static tool set is paid once per binding (no per-step
  RPC) and re-discovered only on a binding-identity bump or ``list_changed``;
* the per-server discovery timeout (``_DISCOVERY_TIMEOUT_S``) that bounds
  ``list_tools()``;
* the per-key circuit breaker that skips a recently-failed server fast.

All MCP SDK + httpx interactions are mocked. No network calls.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.harness import runtime
from aios.mcp.client import (
    _DISCOVERY_TIMEOUT_S,
    _headers_key,
    discover_mcp_tools,
)
from aios.mcp.pool import McpSessionPool
from aios.models.agents import AgentBinding, GenericChildBinding, StepSurface, ToolSpec
from aios.services.agents import tool_cache_binding_id

URL = "https://m.example/"
EMPTY_KEY = _headers_key(None)


def _make_mock_tool(name: str) -> MagicMock:
    tool = MagicMock()
    tool.name = name
    tool.description = f"desc {name}"
    tool.inputSchema = {"type": "object"}
    return tool


def _make_mock_session(tool_names: list[str], instructions: str | None = None) -> AsyncMock:
    session = AsyncMock()
    init = MagicMock()
    init.instructions = instructions
    session.initialize = AsyncMock(return_value=init)
    result = MagicMock()
    result.tools = [_make_mock_tool(n) for n in tool_names]
    session.list_tools = AsyncMock(return_value=result)
    return session


def _transport_mock() -> MagicMock:
    t = MagicMock()
    t.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock(), MagicMock()))
    t.__aexit__ = AsyncMock(return_value=False)
    return t


def _session_ctx(session: AsyncMock) -> MagicMock:
    cls = MagicMock()
    cls.return_value.__aenter__ = AsyncMock(return_value=session)
    cls.return_value.__aexit__ = AsyncMock(return_value=False)
    return cls


@pytest.fixture
def pool_runtime() -> Any:
    """Install a fresh ``McpSessionPool`` on ``runtime`` for the test, restoring after."""
    prior = runtime.mcp_session_pool
    pool = McpSessionPool()
    runtime.mcp_session_pool = pool
    try:
        yield pool
    finally:
        runtime.mcp_session_pool = prior


# ── result cache ────────────────────────────────────────────────────────────


class TestDiscoveryResultCache:
    async def test_second_discovery_same_binding_skips_list_tools_rpc(
        self, pool_runtime: McpSessionPool
    ) -> None:
        """Acceptance: with caching on, the SECOND discovery for the same
        binding identity serves from cache and issues NO ``list_tools()`` RPC —
        removing it from the per-step hot path."""
        session = _make_mock_session(["create_issue"])
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx(session)),
        ):
            tools1, instr1 = await discover_mcp_tools(URL, "v", {}, "github", binding_id="agt_1:3")
            assert session.list_tools.await_count == 1
            tools2, instr2 = await discover_mcp_tools(URL, "v", {}, "github", binding_id="agt_1:3")

        # Cache hit: no further RPC, identical result.
        assert session.list_tools.await_count == 1
        assert tools2 == tools1
        assert instr2 == instr1
        assert tools1[0]["function"]["name"] == "mcp__github__create_issue"

    async def test_binding_version_bump_rediscovers(self, pool_runtime: McpSessionPool) -> None:
        """Acceptance: a binding-identity bump (agent-version change) re-pays
        discovery — the new tool set propagates with no staleness."""
        session = _make_mock_session(["tool_v3"])
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx(session)),
        ):
            await discover_mcp_tools(URL, "v", {}, "s", binding_id="agt_1:3")
            assert session.list_tools.await_count == 1
            # New version → fresh cache key → re-discovers.
            await discover_mcp_tools(URL, "v", {}, "s", binding_id="agt_1:4")

        assert session.list_tools.await_count == 2

    async def test_no_binding_id_never_caches(self, pool_runtime: McpSessionPool) -> None:
        """The API/test path (``binding_id=None``) is never cached — every call
        re-polls (preserving legacy uncached behaviour for the broker-less path)."""
        session = _make_mock_session(["t"])
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx(session)),
        ):
            await discover_mcp_tools(URL, "v", {}, "s")
            await discover_mcp_tools(URL, "v", {}, "s")

        assert session.list_tools.await_count == 2

    async def test_list_changed_notification_invalidates_cache(
        self, pool_runtime: McpSessionPool
    ) -> None:
        """Acceptance: the MCP ``tools/list_changed`` notification invalidates
        the cache so the next discovery re-polls (no staleness regression)."""
        session = _make_mock_session(["t"])
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx(session)),
        ):
            await discover_mcp_tools(URL, "v", {}, "s", binding_id="agt_1:3")
            assert session.list_tools.await_count == 1

            # Simulate the server pushing a tools/list_changed notification by
            # invoking the same invalidation the registered handler performs.
            pool_runtime._invalidate_tools_for_pool_key((URL, "v", EMPTY_KEY))

            await discover_mcp_tools(URL, "v", {}, "s", binding_id="agt_1:3")

        assert session.list_tools.await_count == 2


# ── binding-identity accessor (#1554) ───────────────────────────────────────


def _agent_surface(*, agent_id: str, version: int, tools: list[ToolSpec]) -> StepSurface:
    """A ``StepSurface`` with an ``agent`` binding — the latest/pinned/agented-child
    identity that keys the #1391 cache on ``(agent_id, version)``."""
    return StepSurface(
        model="test/dummy",
        system="sys",
        tools=tools,
        skills=[],
        mcp_servers=[],
        http_servers=[],
        litellm_extra={},
        window_min=1000,
        window_max=100000,
        preempt_policy="wait",
        binding=AgentBinding(agent_id=agent_id, version=version),
    )


def _generic_child_surface(*, session_id: str, tools: list[ToolSpec]) -> StepSurface:
    """A ``StepSurface`` with a ``generic_child`` binding — keys the #1391 cache
    on its own ``session_id`` (no agent identity, no sentinel)."""
    return StepSurface(
        model="test/dummy",
        system="sys",
        tools=tools,
        skills=[],
        mcp_servers=[],
        http_servers=[],
        litellm_extra={},
        window_min=1000,
        window_max=100000,
        preempt_policy="wait",
        binding=GenericChildBinding(session_id=session_id),
    )


class TestToolCacheBindingId:
    """#1554/#1688: the binding identity that keys the #1391 tool-list cache
    must distinguish distinct agents/sessions, never collapse them onto
    ``"?:..."``. Post-#1688 the identity is a total match on ``binding.kind``.
    """

    def test_distinct_version_pinned_agents_same_version_get_distinct_ids(self) -> None:
        """Two ``agent``-bound surfaces of *different* agents pinned to the same
        version number must NOT collide — they bind to ``"agt_A:3"`` /
        ``"agt_B:3"``. On master both duck-typed to ``"?:3"``."""
        a = _agent_surface(agent_id="agt_A", version=3, tools=[])
        b = _agent_surface(agent_id="agt_B", version=3, tools=[])
        id_a = tool_cache_binding_id(a)
        id_b = tool_cache_binding_id(b)
        assert id_a == "agt_A:3"
        assert id_b == "agt_B:3"
        assert id_a != id_b

    def test_distinct_generic_children_get_distinct_session_anchored_ids(self) -> None:
        """Two generic workflow children (``generic_child`` binding) carry
        distinct attenuated per-run surfaces; they key on their own
        ``session_id`` and so differ. On master both collapsed to ``"?:0"``."""
        a = _generic_child_surface(session_id="ses_child_a", tools=[])
        b = _generic_child_surface(session_id="ses_child_b", tools=[])
        id_a = tool_cache_binding_id(a)
        id_b = tool_cache_binding_id(b)
        assert id_a == "child:ses_child_a"
        assert id_b == "child:ses_child_b"
        assert id_a != id_b

    def test_latest_agent_path_unchanged(self) -> None:
        """The latest-agent path keeps its ``"<id>:<version>"`` form —
        the one path that was already correct."""
        agent = _agent_surface(agent_id="agt_1", version=3, tools=[])
        assert tool_cache_binding_id(agent) == "agt_1:3"

    def test_agented_child_shares_with_siblings(self) -> None:
        """Trap 1 (#1688): an *agented* workflow child keeps an ``agent`` binding
        on ``(agent_id, version)`` so sibling runs share the raw-discovery cache
        — it must NOT be forced onto a per-session key like a generic child."""
        child_run_1 = _agent_surface(agent_id="agt_X", version=5, tools=[])
        child_run_2 = _agent_surface(agent_id="agt_X", version=5, tools=[])
        assert tool_cache_binding_id(child_run_1) == tool_cache_binding_id(child_run_2) == "agt_X:5"

    async def test_cross_agent_pool_key_not_poisoned(self, pool_runtime: McpSessionPool) -> None:
        """End-to-end: agent A's discovered tool list is NOT served to a
        version-pinned agent B sharing one ``_PoolKey`` (same url/vault/headers).
        On master both bind to ``"?:3"`` → B is served A's tools."""
        a = _agent_surface(agent_id="agt_A", version=3, tools=[])
        b = _agent_surface(agent_id="agt_B", version=3, tools=[])
        bind_a = tool_cache_binding_id(a)
        bind_b = tool_cache_binding_id(b)
        # Distinct agents pinned to the same version must NOT share a cache slot.
        assert bind_a != bind_b

        # One transport session (same ``_PoolKey``) whose tool set changes between
        # the two discoveries — A then B. If B shared A's cache key it would be
        # served A's stale tool list (await_count stays 1); with distinct keys B
        # re-discovers and sees its own.
        session = _make_mock_session(["tool_for_A"])
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx(session)),
        ):
            tools_a, _ = await discover_mcp_tools(URL, "v", {}, "s", binding_id=bind_a)
            assert session.list_tools.await_count == 1

            result_b = MagicMock()
            result_b.tools = [_make_mock_tool("tool_for_B")]
            session.list_tools = AsyncMock(return_value=result_b)
            tools_b, _ = await discover_mcp_tools(URL, "v", {}, "s", binding_id=bind_b)

        # B re-discovers (distinct key) and gets its own tool, not A's cached one.
        assert session.list_tools.await_count == 1
        assert tools_a[0]["function"]["name"] == "mcp__s__tool_for_A"
        assert tools_b[0]["function"]["name"] == "mcp__s__tool_for_B"


# ── per-server discovery timeout ────────────────────────────────────────────


class TestDiscoveryTimeout:
    async def test_slow_list_tools_times_out_and_marks_unhealthy(
        self, pool_runtime: McpSessionPool
    ) -> None:
        """Acceptance: a hung ``list_tools()`` is bounded by the per-server
        timeout (does not hang forever) and the key is marked unhealthy."""

        async def _never_returns() -> Any:
            await asyncio.sleep(3600)

        session = _make_mock_session(["t"])
        session.list_tools = AsyncMock(side_effect=_never_returns)

        # Speed: patch the timeout to a tiny value so the test is fast.
        with (
            patch("aios.mcp.client._DISCOVERY_TIMEOUT_S", 0.05),
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx(session)),
            pytest.raises(TimeoutError),
        ):
            await asyncio.wait_for(
                discover_mcp_tools(URL, "v", {}, "s", binding_id="agt_1:3"),
                timeout=5.0,
            )

        # The key is now in backoff so the next prelude can skip it fast.
        assert pool_runtime.is_unhealthy(URL, "v", EMPTY_KEY)

    async def test_default_discovery_timeout_is_finite_and_tight(self) -> None:
        """The discovery timeout exists and is well under the 960s step budget —
        the precondition that prevents a prelude stall (#1391)."""
        assert 0 < _DISCOVERY_TIMEOUT_S < 960


# ── circuit breaker bookkeeping ─────────────────────────────────────────────


class TestCircuitBreaker:
    def test_mark_unhealthy_then_healthy(self) -> None:
        pool = McpSessionPool()
        assert not pool.is_unhealthy(URL, "v", EMPTY_KEY, now=100.0)
        pool.mark_unhealthy(URL, "v", EMPTY_KEY, backoff_s=60.0, now=100.0)
        # Inside the window → unhealthy.
        assert pool.is_unhealthy(URL, "v", EMPTY_KEY, now=130.0)
        # After the window → healthy again (and the entry is reaped).
        assert not pool.is_unhealthy(URL, "v", EMPTY_KEY, now=200.0)
        # A successful discovery clears the breaker explicitly.
        pool.mark_unhealthy(URL, "v", EMPTY_KEY, backoff_s=60.0, now=300.0)
        pool.mark_healthy(URL, "v", EMPTY_KEY)
        assert not pool.is_unhealthy(URL, "v", EMPTY_KEY, now=310.0)

    def test_cache_get_set_and_invalidate(self) -> None:
        pool = McpSessionPool()
        assert pool.get_cached_tools(URL, "v", EMPTY_KEY, "agt_1:3") is None
        pool.set_cached_tools(URL, "v", EMPTY_KEY, "agt_1:3", [{"x": 1}], "instr")
        assert pool.get_cached_tools(URL, "v", EMPTY_KEY, "agt_1:3") == ([{"x": 1}], "instr")
        # Different binding → independent entry.
        assert pool.get_cached_tools(URL, "v", EMPTY_KEY, "agt_1:4") is None
        # list_changed drops every binding for the key.
        pool.set_cached_tools(URL, "v", EMPTY_KEY, "agt_1:4", [{"y": 2}], None)
        pool._invalidate_tools_for_pool_key((URL, "v", EMPTY_KEY))
        assert pool.get_cached_tools(URL, "v", EMPTY_KEY, "agt_1:3") is None
        assert pool.get_cached_tools(URL, "v", EMPTY_KEY, "agt_1:4") is None

    async def test_list_changed_handler_invalidates_only_on_tool_change(self) -> None:
        """The ``ClientSession`` message handler the pool registers (#1391)
        drops the cache on ``tools/list_changed`` but ignores other server
        notifications — so an unrelated notification can't thrash the cache."""
        import mcp.types as t

        pool = McpSessionPool()
        key = (URL, "v", EMPTY_KEY)
        handler = pool._make_list_changed_handler(key)

        pool.set_cached_tools(URL, "v", EMPTY_KEY, "agt_1:3", [{"x": 1}], None)
        await handler(t.ServerNotification(t.ToolListChangedNotification()))
        assert pool.get_cached_tools(URL, "v", EMPTY_KEY, "agt_1:3") is None

        pool.set_cached_tools(URL, "v", EMPTY_KEY, "agt_1:3", [{"x": 1}], None)
        await handler(
            t.ServerNotification(
                t.LoggingMessageNotification(
                    params=t.LoggingMessageNotificationParams(level="info", data="hi")
                )
            )
        )
        assert pool.get_cached_tools(URL, "v", EMPTY_KEY, "agt_1:3") is not None

    def test_invalidate_by_vault(self) -> None:
        pool = McpSessionPool()
        pool.set_cached_tools(URL, "vlt_1", EMPTY_KEY, "agt_1:3", [{"a": 1}], None)
        pool.set_cached_tools(URL, "vlt_2", EMPTY_KEY, "agt_1:3", [{"b": 2}], None)
        pool.invalidate_tools_by_vault("vlt_1")
        assert pool.get_cached_tools(URL, "vlt_1", EMPTY_KEY, "agt_1:3") is None
        # Other vaults' entries survive.
        assert pool.get_cached_tools(URL, "vlt_2", EMPTY_KEY, "agt_1:3") is not None

    def test_lookup_cached_tool_parameters_from_openai_envelope(self) -> None:
        """Dispatch reads sanitized ``function.parameters`` off the cached
        OpenAI-shaped dicts ``discover_mcp_tools`` stores — not a parallel
        registry."""
        schema = {
            "type": "object",
            "properties": {"athlete_id": {"type": "string"}},
            "required": ["athlete_id"],
        }
        envelope = {
            "type": "function",
            "function": {
                "name": "mcp__kine__propose_workout",
                "description": "",
                "parameters": schema,
            },
        }
        pool = McpSessionPool()
        pool.set_cached_tools(URL, "v", EMPTY_KEY, "agt_1:3", [envelope], None)
        assert pool.lookup_cached_tool_parameters("mcp__kine__propose_workout") == schema
        assert pool.lookup_cached_tool_parameters("mcp__kine__propose_workout", url=URL) == schema
        assert (
            pool.lookup_cached_tool_parameters(
                "mcp__kine__propose_workout", url="https://other.example/"
            )
            is None
        )
        assert pool.lookup_cached_tool_parameters("mcp__kine__missing") is None

    def test_lookup_cached_tool_parameters_same_schema_across_bindings(self) -> None:
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        pool = McpSessionPool()
        pool.set_cached_tools(
            URL,
            "v",
            EMPTY_KEY,
            "agt_1:3",
            [
                {
                    "type": "function",
                    "function": {
                        "name": "mcp__kine__propose_workout",
                        "parameters": schema,
                    },
                }
            ],
            None,
        )
        pool.set_cached_tools(
            URL,
            "v",
            EMPTY_KEY,
            "agt_1:4",
            [
                {
                    "type": "function",
                    "function": {
                        "name": "mcp__kine__propose_workout",
                        "parameters": schema,
                    },
                }
            ],
            None,
        )
        assert pool.lookup_cached_tool_parameters("mcp__kine__propose_workout", url=URL) == schema

    def test_lookup_cached_tool_parameters_ambiguous_same_url_fails_open(self) -> None:
        old_schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        new_schema = {"type": "object", "properties": {"b": {"type": "number"}}}
        pool = McpSessionPool()
        pool.set_cached_tools(
            URL,
            "vault_a",
            EMPTY_KEY,
            "agt_1:3",
            [
                {
                    "type": "function",
                    "function": {
                        "name": "mcp__kine__propose_workout",
                        "parameters": old_schema,
                    },
                }
            ],
            None,
        )
        pool.set_cached_tools(
            URL,
            "vault_b",
            EMPTY_KEY,
            "agt_1:4",
            [
                {
                    "type": "function",
                    "function": {
                        "name": "mcp__kine__propose_workout",
                        "parameters": new_schema,
                    },
                }
            ],
            None,
        )
        assert pool.lookup_cached_tool_parameters("mcp__kine__propose_workout", url=URL) is None
