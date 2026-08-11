"""Test matrix for the single-source tool-call classifier (#1076).

``aios.harness.tool_disposition.classify_tool_call`` is the ONE walk of the
permission ladder shared by all three consumers (``loop._classify_tool_call``,
``sessions._classify_awaiting``, ``sweep._was_dispatched``).  These tests
exercise all five dispositions x the ``confirmation_resolved`` axis x the
``mcp_server_map`` axis, replacing the scattered per-consumer fragments that
previously had to be kept byte-consistent by hand.

The load-bearing case for the live defect this issue fixes is
``test_route_gated_always_ask_unconfirmed_needs_confirm``: a route-gated
``http_request`` refined to ``always_ask`` and NOT yet confirmed must classify
as ``NEEDS_CONFIRM`` — the route refinement that ``_was_dispatched`` historically
lacked.
"""

from __future__ import annotations

from typing import Any

from aios.harness.tool_disposition import ToolDisposition, classify_tool_call
from aios.models.agents import (
    AgentBinding,
    HttpPermissionPolicy,
    HttpRouteSpec,
    HttpServerSpec,
    McpServerSpec,
    StepSurface,
    ToolSpec,
)


def test_dispatch_kind_literals_match_disposition_values() -> None:
    """``loop.ToolDispatchKind``'s literals ARE ``ToolDisposition``'s values.

    ``loop._classify_tool_call`` returns ``disposition.value`` directly, so the
    two enumerations must stay byte-identical or the dispatch projection would
    emit an out-of-Literal string. This guard converts that coupling from prose
    into a checked invariant.
    """
    from typing import get_args

    from aios.harness.loop import ToolDispatchKind

    assert set(get_args(ToolDispatchKind.__value__)) == {d.value for d in ToolDisposition}


def _agent(
    *,
    tools: list[ToolSpec] | None = None,
    http_servers: list[HttpServerSpec] | None = None,
    mcp_servers: list[McpServerSpec] | None = None,
) -> StepSurface:
    return StepSurface(
        model="gpt-test",
        system="be helpful",
        tools=tools or [],
        skills=[],
        mcp_servers=mcp_servers or [],
        http_servers=http_servers or [],
        litellm_extra={},
        window_min=1,
        window_max=10,
        preempt_policy="wait",
        binding=AgentBinding(agent_id="agt_test", version=1),
    )


def _http_agent(policy: str) -> StepSurface:
    route = HttpRouteSpec(
        path_pattern="/lights/*",
        enabled=True,
        permission_policy=HttpPermissionPolicy(type=policy),
        methods=None,
    )
    server = HttpServerSpec(name="hue", base_url="https://api.example.com/v1", routes=[route])
    return _agent(tools=[ToolSpec(type="http_request")], http_servers=[server])


def _http_args() -> dict[str, Any]:
    return {"server_ref": "hue", "path": "/lights/1", "method": "GET"}


# ── built-in / custom branch ─────────────────────────────────────────────────


class TestBuiltinBranch:
    def test_always_allow_builtin_immediate(self) -> None:
        agent = _agent(tools=[ToolSpec(type="bash", permission="always_allow")])
        assert (
            classify_tool_call("bash", "{}", agent, confirmation_resolved=False)
            == ToolDisposition.IMMEDIATE
        )

    def test_no_permission_set_immediate(self) -> None:
        # resolve_permission → None → not always_ask → immediate.
        agent = _agent(tools=[ToolSpec(type="bash")])
        assert (
            classify_tool_call("bash", "{}", agent, confirmation_resolved=False)
            == ToolDisposition.IMMEDIATE
        )

    def test_always_ask_builtin_unconfirmed_needs_confirm(self) -> None:
        agent = _agent(tools=[ToolSpec(type="bash", permission="always_ask")])
        assert (
            classify_tool_call("bash", "{}", agent, confirmation_resolved=False)
            == ToolDisposition.NEEDS_CONFIRM
        )

    def test_always_ask_builtin_confirmed_immediate(self) -> None:
        # The confirmation gate has been satisfied → no longer awaiting → immediate.
        agent = _agent(tools=[ToolSpec(type="bash", permission="always_ask")])
        assert (
            classify_tool_call("bash", "{}", agent, confirmation_resolved=True)
            == ToolDisposition.IMMEDIATE
        )

    def test_unknown_bare_name_custom(self) -> None:
        agent = _agent(tools=[])
        assert (
            classify_tool_call("some_client_tool", "{}", agent, confirmation_resolved=False)
            == ToolDisposition.CUSTOM
        )
        # confirmation_resolved is irrelevant for custom tools.
        assert (
            classify_tool_call("some_client_tool", "{}", agent, confirmation_resolved=True)
            == ToolDisposition.CUSTOM
        )


# ── route refinement (the #1076 defect) ──────────────────────────────────────


class TestRouteRefinement:
    def test_route_gated_always_allow_immediate(self) -> None:
        agent = _http_agent("always_allow")
        assert (
            classify_tool_call("http_request", _http_args(), agent, confirmation_resolved=False)
            == ToolDisposition.IMMEDIATE
        )

    def test_route_gated_always_ask_unconfirmed_needs_confirm(self) -> None:
        # The live defect: ``_was_dispatched`` historically applied only
        # ``resolve_permission`` (the tool's BASE permission, always_allow) and
        # missed the route refinement → wrongly reported dispatched=True. The
        # single classifier applies ``classify_permission`` → NEEDS_CONFIRM.
        agent = _http_agent("always_ask")
        assert (
            classify_tool_call("http_request", _http_args(), agent, confirmation_resolved=False)
            == ToolDisposition.NEEDS_CONFIRM
        )

    def test_route_gated_always_ask_confirmed_immediate(self) -> None:
        agent = _http_agent("always_ask")
        assert (
            classify_tool_call("http_request", _http_args(), agent, confirmation_resolved=True)
            == ToolDisposition.IMMEDIATE
        )

    def test_string_form_arguments_refined(self) -> None:
        # Providers differ on str- vs dict-form arguments; parse_arguments
        # normalizes both, so the route refinement fires either way.
        agent = _http_agent("always_ask")
        raw = '{"server_ref": "hue", "path": "/lights/1", "method": "GET"}'
        assert (
            classify_tool_call("http_request", raw, agent, confirmation_resolved=False)
            == ToolDisposition.NEEDS_CONFIRM
        )

    def test_unparseable_arguments_fall_through_to_base(self) -> None:
        # Malformed args → no route refinement → tool's base permission (None
        # → not always_ask) → immediate, so the schema validator can emit a
        # typed error the model self-corrects from.
        agent = _http_agent("always_ask")
        assert (
            classify_tool_call("http_request", "not json", agent, confirmation_resolved=False)
            == ToolDisposition.IMMEDIATE
        )


# ── MCP branch ───────────────────────────────────────────────────────────────


class TestMcpBranch:
    def test_mcp_always_allow_immediate(self) -> None:
        agent = _agent(
            tools=[ToolSpec(type="mcp_toolset", mcp_server_name="srv", permission="always_allow")]
        )
        assert (
            classify_tool_call("mcp__srv__tool", "{}", agent, confirmation_resolved=False)
            == ToolDisposition.MCP_IMMEDIATE
        )

    def test_disabled_mcp_tool_is_not_model_callable(self) -> None:
        agent = _agent(
            tools=[
                ToolSpec.model_validate(
                    {
                        "type": "mcp_toolset",
                        "mcp_server_name": "srv",
                        "permission": "always_allow",
                        "configs": [{"name": "tool", "enabled": False}],
                    }
                )
            ]
        )
        assert (
            classify_tool_call("mcp__srv__tool", "{}", agent, confirmation_resolved=False)
            == ToolDisposition.MCP_BLOCKED
        )

    def test_cli_only_mcp_tool_is_not_model_callable(self) -> None:
        agent = _agent(
            tools=[
                ToolSpec.model_validate(
                    {
                        "type": "mcp_toolset",
                        "mcp_server_name": "srv",
                        "permission": "always_allow",
                        "configs": [{"name": "tool", "transport": "cli"}],
                    }
                )
            ]
        )
        assert (
            classify_tool_call("mcp__srv__tool", "{}", agent, confirmation_resolved=False)
            == ToolDisposition.MCP_BLOCKED
        )

    def test_undeclared_mcp_toolset_is_not_model_callable(self) -> None:
        agent = _agent(tools=[])
        assert (
            classify_tool_call("mcp__srv__tool", "{}", agent, confirmation_resolved=False)
            == ToolDisposition.MCP_BLOCKED
        )

    def test_undeclared_mcp_toolset_stays_blocked_after_confirmation(self) -> None:
        agent = _agent(tools=[])
        assert (
            classify_tool_call("mcp__srv__tool", "{}", agent, confirmation_resolved=True)
            == ToolDisposition.MCP_BLOCKED
        )

    def test_unknown_mcp_server_with_map_unknown_mcp(self) -> None:
        # Dispatch path supplies a server map → unregistered server is unknown_mcp.
        agent = _agent(tools=[])
        assert (
            classify_tool_call(
                "mcp__ghost__tool",
                "{}",
                agent,
                confirmation_resolved=False,
                mcp_server_map={},
            )
            == ToolDisposition.UNKNOWN_MCP
        )

    def test_registered_enabled_mcp_server_with_map_not_unknown(self) -> None:
        agent = _agent(
            tools=[
                ToolSpec(
                    type="mcp_toolset",
                    mcp_server_name="srv",
                    permission="always_ask",
                )
            ]
        )
        server = McpServerSpec(name="srv", url="https://x.example")
        assert (
            classify_tool_call(
                "mcp__srv__tool",
                "{}",
                agent,
                confirmation_resolved=False,
                mcp_server_map={"srv": server},
            )
            == ToolDisposition.NEEDS_CONFIRM
        )

    def test_unknown_mcp_server_without_map_still_enforces_enabled(self) -> None:
        agent = _agent(tools=[])
        assert (
            classify_tool_call("mcp__ghost__tool", "{}", agent, confirmation_resolved=False)
            == ToolDisposition.MCP_BLOCKED
        )

    def test_malformed_mcp_name_with_map_unknown_mcp(self) -> None:
        agent = _agent(tools=[])
        assert (
            classify_tool_call("mcp__", "{}", agent, confirmation_resolved=False, mcp_server_map={})
            == ToolDisposition.UNKNOWN_MCP
        )
