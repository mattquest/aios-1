"""Agent resource: model + system prompt + tools + MCP servers.

Agents are versioned: every update creates a new immutable version. The
``agents`` table holds the latest config; the ``agent_versions`` table stores
the full history. The ``model`` field is a free-form LiteLLM model string
(e.g. ``anthropic/claude-opus-4-6``, ``ollama_chat/llama3.3``).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import datetime
from typing import Annotated, Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aios.actors import Actor
from aios.models.skills import AgentSkillRef
from aios.models.target_urls import validate_outbound_target_url
from aios.retirements.registry import tolerated_rename_map
from aios.retirements.telemetry import record_tolerance_hit

# Built-in tool types. Custom tools use type="custom" with extra fields.
BuiltinToolType = Literal[
    "bash",
    "read",
    "write",
    "edit",
    "glob",
    "grep",
    "web_fetch",
    "web_search",
    "search_events",
    "memory_search",
    "schedule_wake",
    "wake_session",
    "wake_self",
    "list_related_sessions",
    "http_request",
    "trigger_create",
    "trigger_remove",
    "trigger_update",
    "trigger_list",
    "list_account_triggers",
    "create_workflow",
    "update_workflow",
    "archive_workflow",
    "unarchive_workflow",
    "resume_gate",
    "get_workflow",
    "list_workflows",
    "get_run",
    "list_runs",
    "archive_run",
    "list_run_events",
    "call_session",
    "call_agent",
    "call_workflow",
    "create_agent",
    "update_agent",
    "archive_agent",
    "get_agent",
    "list_agents",
    "create_goal",
    "list_obligations",
    "defer_obligations",
    "stop_task",
    "list_tasks",
    "skill_upsert",
    "skill_archive",
    "ctx_list",
    "ctx_peek",
    "ctx_grep",
    "ctx_write",
    "ctx_eval",
    "rlm_query",
    "rlm_map",
    "rlm_verify",
]

# Permission policy for built-in tools. Custom tools are always client-controlled
# and ignore this field.
PermissionPolicy = Literal["always_allow", "always_ask"]

# Transport classification — which callers may invoke a tool.
#   "agent_tool": model only (the LLM's tool-call surface).
#   "cli":        sandbox-side ``tool`` CLI only (bash inside the session).
#   "both":       reachable from either.
# The substrate's security frontier: outbound-side-effect tools live as
# ``agent_tool`` so the model is the bottleneck for irreversible effects.
# Enforcement is structural (the broker refuses non-CLI tools). Built-ins
# get a registry default; an agent's ``ToolSpec`` /
# ``McpToolsetConfig`` / ``McpToolConfig`` can override per-tool.
ToolTransport = Literal["cli", "agent_tool", "both"]

# HTTP methods an ``http_request`` route may be scoped to. Mirrors the broker's
# ``_ALLOWED_METHODS`` tuple in ``tools/http_request.py``.
HttpMethod = Literal["GET", "POST", "PUT", "DELETE", "PATCH"]

# What happens to an in-flight model call when a new wake-eligible event (e.g.
# a user message) arrives mid-step. "wait" (default): the step finishes and the
# queued wake handles the event next step. "preempt": the model phase is
# cancelled and the step restarts against context that includes the event.
# Deliberately not named "interrupt" — that vocabulary is the broad operator
# cancel-everything mechanism (``POST /sessions/:id/interrupt``); this is the
# narrow, automatic, model-phase-only policy.
PreemptPolicy = Literal["preempt", "wait"]

_BUILTIN_NAMES: frozenset[str] = frozenset(get_args(BuiltinToolType))

# Read-tolerance for the builtin tool renames (#1419 invoke*→call_*, #1428 cancel_run→stop_task).
# Agent/workflow/run/session rows persisted before a rename carry the pre-rename builtin tool
# names in their `tools` JSONB; without a map they'd fail `ToolSpec` validation on read (a
# deploy-breaker — every agent that exposed the renamed tool to its model would 500). The
# `mode="before"` validator below maps them so old rows still load; the two-step
# create_run/await_run launch tools fold into the unified `call_workflow`.
#
# #1574: the rename map is no longer a hand-maintained constant here — it is sourced from the
# retirement REGISTRY (#1573) via `tolerated_rename_map`, which tolerates a token ONLY while its
# descriptor's `contract_rev IS NULL` (the data migration that rewrites persisted rows has not
# yet been declared as the contract-point). Once a descriptor stamps `contract_rev`, its tokens
# drop out of the map and the validator stops remapping them, so a stale legacy `type` then
# correctly fails validation. Each tolerance fire emits `retirement_tolerance_hits{token}` +
# stamps `last_seen` (corroboration only — NEVER the gate). Keeping this in a `mode="before"`
# validator is deliberate: it travels to every `ToolSpec.model_validate` site automatically (the
# eight read sites), so no loader choke-point is needed and the per-site property is preserved.

# Read-tolerance for RETIRED builtins that have NO canonical successor (#1562). Unlike a
# rename (mapped above), a retired builtin's persisted ``tools`` entry is DROPPED on read —
# there is no model-listed tool to remap it to. #1525 (unify-obligations #2) removed
# ``complete_goal``/``fail_goal`` from ``BuiltinToolType`` + the registry but shipped neither a
# read shim nor a data migration, so any long-lived agent whose ``agents.tools`` JSONB still
# listed them failed ``ToolSpec.model_validate`` on every wake — a pre-context-build throw that
# wedged the agent into an infinite reschedule (only the live kedalion-ultron agent hit it).
# ``return``/``error`` are general step verbs, not model-listed builtins, so there is no
# successor — REMOVE, do not remap. The data migration (0122) rewrites persisted rows; this set
# + :func:`load_tool_specs` cover the post-deploy/pre-migrate window and any future respawn from
# an unmigrated row. (Teardown can drop both once 0122 has run everywhere.)
_RETIRED_BUILTINS: frozenset[str] = frozenset({"complete_goal", "fail_goal"})


def load_tool_specs(raw: Iterable[Any]) -> list[ToolSpec]:
    """Validate a persisted ``tools`` JSONB array into ``ToolSpec``\\ s, with read-tolerance.

    The DB read path persists historical ``tools`` arrays; an entry whose ``type`` is a
    :data:`_RETIRED_BUILTINS` member (a builtin removed from ``BuiltinToolType`` + the registry
    with no canonical successor — #1562's ``complete_goal``/``fail_goal``) is **dropped**, not
    remapped: there is no model-listed tool to validate it against, and ``ToolSpec.model_validate``
    would otherwise raise on the now-illegal Literal, poisoning the whole row's hydration.

    This is the list-level counterpart to the per-entry ``mode="before"`` *rename* shim
    (:meth:`ToolSpec._map_legacy_builtin_names`, now registry-driven via
    :func:`aios.retirements.registry.tolerated_rename_map`): a rename can be done in-place inside a
    single ``ToolSpec``, but dropping an element must happen where the array is iterated. Order is
    preserved; a row
    that listed only retired builtins hydrates to ``[]``. Use this anywhere a persisted ``tools``
    array is loaded so the tolerance is uniform across agents / agent_versions / workflows /
    workflow_versions / wf_runs / sessions.
    """
    out: list[ToolSpec] = []
    for entry in raw:
        if isinstance(entry, dict) and entry.get("type") in _RETIRED_BUILTINS:
            continue
        out.append(ToolSpec.model_validate(entry))
    return out


# Header names the MCP streamable-http transport authors on every request
# (see ``mcp.client.streamable_http._prepare_headers``). A spec header named
# after one of these never reaches the wire — the transport overwrites it
# per-request — yet it would still fragment the connection-pool key
# (``_headers_key``). Compared case-insensitively; HTTP header names are too.
_RESERVED_MCP_HEADERS: frozenset[str] = frozenset(
    {"accept", "content-type", "mcp-session-id", "mcp-protocol-version"}
)

# RFC 7230 ``token``: the legal character set for an HTTP header field name.
# Excludes whitespace, control chars, ``:``, and non-ASCII by construction.
_HEADER_NAME_RE = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")


# ── MCP server declaration ────────────────────────────────────────────────────


class McpServerSpec(BaseModel):
    """One entry in an agent's ``mcp_servers`` list.

    Declares a remote MCP server reachable via streamable HTTP transport.
    The ``name`` is used to cross-reference from ``mcp_toolset`` tool entries
    and to namespace discovered tools as ``mcp__<name>__<tool_name>``.

    ``include_instructions`` controls whether the server's
    ``InitializeResult.instructions`` (per MCP spec) is rendered into the
    system prompt.  Defaults true so connector-mounted servers — and any
    third-party server that ships useful affordance prose — light up
    automatically.  Set false to opt out per agent (unfamiliar prose,
    noisy servers).

    ``headers`` are extra NON-SECRET HTTP headers sent on every request to
    this server — toolset selectors (e.g. GitHub's
    ``X-MCP-Toolsets: discussions,issues``), format hints, API-version
    pins.  Do NOT put secrets here: this dict is stored in plaintext agent
    JSON.  Real credentials belong in the vault path; a vault-derived auth
    header overrides a same-named entry here (auth headers win on
    collision).  Names must be valid HTTP tokens and values printable ASCII
    (validated below) so they can't fail only at connection time; headers
    the MCP transport authors itself (Accept, Content-Type, Mcp-Session-Id,
    Mcp-Protocol-Version) are rejected — setting them here is a silent no-op.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["url"] = "url"
    name: str = Field(min_length=1, max_length=64)
    url: str = Field(min_length=1)
    include_instructions: bool = True

    _validate_url = field_validator("url")(validate_outbound_target_url)
    headers: dict[str, str] | None = Field(default=None)

    @field_validator("name")
    @classmethod
    def _name_not_builtin(cls, v: str) -> str:
        # The ``tool`` CLI uses a flat top-level namespace (``tool <name>``
        # for built-ins, ``tool <server> <method>`` for MCP). Reserving
        # built-in names on ``McpServerSpec`` keeps the broker's
        # name-resolution unambiguous.
        if v in _BUILTIN_NAMES:
            raise ValueError(
                f"MCP server name {v!r} collides with a built-in tool name; "
                f"the `tool` CLI uses a flat top-level namespace where "
                f"built-in tool names are reserved. Pick a different name."
            )
        return v

    @field_validator("headers")
    @classmethod
    def _validate_headers(cls, v: dict[str, str] | None) -> dict[str, str] | None:
        # Reject HTTP-illegal headers at config time. httpx would otherwise
        # raise only when the connection is opened, where the error is caught
        # and the whole server silently dropped (logged at WARN). Fail loud at
        # the write boundary instead.
        if v is None:
            return None
        for name, value in v.items():
            if not _HEADER_NAME_RE.fullmatch(name):
                raise ValueError(
                    f"invalid HTTP header name {name!r}: must be a non-empty RFC 7230 "
                    "token (ASCII letters, digits, and any of !#$%&'*+-.^_`|~)"
                )
            if name.lower() in _RESERVED_MCP_HEADERS:
                raise ValueError(
                    f"header {name!r} is authored by the MCP transport on every request; "
                    "setting it here has no effect on the wire. Remove it."
                )
            # Values: printable ASCII (0x20-0x7E) plus HTAB. CR/LF would enable
            # header injection; non-ASCII can't be encoded on the wire.
            bad = next((c for c in value if c != "\t" and not 0x20 <= ord(c) <= 0x7E), None)
            if bad is not None:
                raise ValueError(
                    f"invalid value for header {name!r}: character {bad!r} is not allowed "
                    "(only printable ASCII and tab — no control chars, CR, LF, or non-ASCII)"
                )
        return v


# ── MCP toolset config (permission policies for discovered tools) ──────────


class McpPermissionPolicy(BaseModel):
    """Wrapper matching Anthropic's ``{type: "always_allow"}`` shape."""

    model_config = ConfigDict(extra="forbid")

    type: PermissionPolicy


class McpToolsetConfig(BaseModel):
    """Default config for all tools discovered from an MCP server."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    permission_policy: McpPermissionPolicy | None = None
    transport: ToolTransport | None = None


class McpToolConfig(BaseModel):
    """Per-tool override within an ``mcp_toolset`` entry.

    ``read_allow`` opts a single discovered tool into the outbound-suppression
    read allowlist (#710): MCP has no HTTP-method convention, so when a session
    runs with ``outbound_suppression == "on"`` every MCP call is *default-deny*
    (suppressed with a synthesized success) UNLESS the operator marked the
    specific tool ``read_allow=True`` at config time. A read-allowed tool runs
    for real even under suppression. Default ``False`` — the safe choice for a
    protocol that can't self-describe side effects.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    enabled: bool = True
    permission_policy: McpPermissionPolicy | None = None
    transport: ToolTransport | None = None
    read_allow: bool = False


# ── HTTP server declaration ──────────────────────────────────────────────────


class HttpPermissionPolicy(BaseModel):
    """Wrapper matching the ``{type: "always_allow"}`` shape used for MCP."""

    model_config = ConfigDict(extra="forbid")

    type: PermissionPolicy


class HttpRouteSpec(BaseModel):
    """One entry in an ``HttpServerSpec.routes`` allowlist.

    ``path_pattern`` is a glob against the request path (``*`` matches one
    segment, ``**`` matches any number of segments).  ``description`` is
    operator-authored prose rendered into the system prompt so the agent
    knows what the route does and how to call it.  ``permission_policy``
    gates *execution*: ``always_ask`` leaves the call unresolved in the
    event log until the client confirms via
    ``POST /sessions/:id/tool-confirmations``.

    ``methods`` scopes the route to a set of HTTP verbs so a surface can
    express read/write attenuation structurally — e.g. ``GET`` everywhere
    but ``POST`` only on a sandbox path (#828).  ``None`` (the default)
    means *all* methods are allowed (the method-dimension lattice top;
    backward-compatible with routes authored before method scoping).  A
    non-empty list restricts the route to exactly those verbs.  An empty
    list (``[]``) is *deny-all* — the method-dimension lattice bottom, and
    the natural result of intersecting two disjoint method sets during
    attenuation; it matches nothing.  The capability meet
    (:mod:`aios.models.attenuation`) intersects ``methods`` per route, so a
    child surface can narrow a parent route's verbs but never widen them.

    GraphQL caveat — **REST-only discipline for attenuated surfaces.**
    Method scoping confines REST read/write because the verb encodes the
    semantics.  A GraphQL endpoint serves both queries (reads) and
    mutations (writes) over a single ``POST`` path, so method scoping
    *cannot* separate read from write there, and the broker does not
    inspect request bodies.  An operator who needs to confine writes on a
    GraphQL surface must place reads and writes behind distinct
    ``base_url`` servers (each with its own credential/route allowlist) or
    accept that granting ``POST`` grants both.

    ``allow_query`` opts the route into permitting a query string on the
    request ``path``.  The default is ``False``: a ``?...`` is rejected at
    the route gate (#485) because ``httpx`` parses the query off the URL
    and an unanticipating allowlist — e.g. a read-only ``/lights/*`` — is
    bypassed when the upstream interprets ``?action=delete`` as a write.
    An operator sets ``allow_query=True`` only on routes where the query
    string cannot escalate beyond what the route already grants and where
    it is functionally required — e.g. a GitHub ``/repos/**`` route that
    already permits every verb and must follow cursor/``page`` pagination
    to read a full comment thread (#1156).  The path portion is still
    glob-matched against ``path_pattern`` (the query is stripped before
    the match), and ``.``/``..`` dot-segment rejection still applies, so a
    query allowance never widens the path-dimension grant.  It is
    launcher-verbatim under attenuation: a child surface cannot turn it on
    where the parent left it off.
    """

    model_config = ConfigDict(extra="forbid")

    path_pattern: str = Field(min_length=1)
    description: str | None = None
    enabled: bool = True
    permission_policy: HttpPermissionPolicy | None = None
    methods: list[HttpMethod] | None = None
    allow_query: bool = False
    # Outbound-suppression override (#710). When a session runs with
    # ``outbound_suppression == "on"`` the broker classifies each matched call
    # as read (passes through) or write (suppressed with a synthesized success
    # + audit event). The default classifier is the HTTP method: ``GET`` is a
    # read, ``POST``/``PUT``/``PATCH``/``DELETE`` are writes. ``suppress`` is the
    # per-route escape hatch for the cases the method doesn't predict: set it
    # ``True`` to suppress a side-effecting ``GET``, ``False`` to let a
    # read-only ``POST`` (a GraphQL/JSON-RPC query, a search endpoint) pass.
    # ``None`` (default) defers to the method classifier. Off when suppression
    # is off — this never gates a normal request.
    suppress: bool | None = None


class HttpServerSpec(BaseModel):
    """One entry in an agent's ``http_servers`` list.

    Declares an authenticated HTTP endpoint the agent can reach via the
    ``http_request`` built-in tool.  ``base_url`` is the common URL
    prefix the agent's ``path`` argument is appended to; ``routes`` is
    the allowlist of path patterns the broker permits.  Credentials are
    resolved at request time from the session's bound vaults, keyed on
    ``base_url``.  Secret never enters the sandbox — the worker authors
    the ``Authorization`` header from the vault credential.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64)
    base_url: str = Field(min_length=1)
    description: str | None = None
    routes: list[HttpRouteSpec] = Field(default_factory=list)
    # Status code returned on a suppressed call's synthesized success response
    # (#710). The synthesized body is empty (``""``) so a JSON parse yields
    # nothing surprising; the status defaults to ``200``. Configurable per
    # http_server for surfaces whose success contract is e.g. ``201 Created`` —
    # the agent must observe a plausible success so its behavior validates.
    suppressed_response_status: int = Field(default=200, ge=100, le=599)


def validate_http_servers(servers: list[HttpServerSpec]) -> None:
    """Cross-item invariants for ingress ``http_servers`` lists.

    ``base_url`` is the credential-resolution key and the attenuation key, so
    each authored surface must contain it at most once.  Equality is exact
    string equality, matching run-time resolution and attenuation semantics.
    """
    seen: set[str] = set()
    for server in servers:
        if server.base_url in seen:
            raise ValueError(f"duplicate base_url {server.base_url!r}")
        seen.add(server.base_url)


def validate_mcp_servers(servers: list[McpServerSpec]) -> None:
    """Cross-item invariant for ingress ``mcp_servers`` lists: unique ``name``.

    ``name`` is the sole join key used everywhere a server is looked up (the
    ``mcp_toolset.mcp_server_name`` cross-reference, the ``mcp__<name>__<tool>``
    dispatch namespace, ``discover_session_mcp_tools``'s ``{name: spec}`` map)
    — a second entry with the same ``name`` is unreachable dead config at best
    (only the runtime's last-writer-wins lookup would ever be used) and, at the
    attenuation meet (:mod:`aios.models.attenuation`, keyed on the joint
    ``(name, url)``), a same-name pair with differing ``url``/``headers``
    breaks the normal-form contract ``attenuate(x, x) == canonicalize(x)``
    (the meet's launcher-verbatim survival picks whichever same-key entry
    happens to land last in the dict comprehension, which the un-deduped
    input was never guaranteed to reproduce). Reject at the ingress boundary
    rather than let it silently persist.
    """
    seen: set[str] = set()
    for server in servers:
        if server.name in seen:
            raise ValueError(f"duplicate mcp server name {server.name!r}")
        seen.add(server.name)


def validate_tools(tools: list[ToolSpec]) -> None:
    """Cross-item invariant for ingress ``tools`` lists: unique attenuation identity.

    Mirrors :func:`validate_http_servers`/:func:`validate_mcp_servers` for the
    tools dimension, keyed on the same identity the attenuation meet joins on
    (``models.attenuation._tool_key``, reproduced inline here to avoid a
    downward import from this module into ``attenuation``): builtins by
    ``type``, custom tools by ``name``, MCP toolsets by ``mcp_server_name``. A
    second entry sharing a key is unreachable dead config (only one survives
    any runtime lookup keyed the same way — the CLI broker's
    ``_find_builtin_spec``/``_find_mcp_toolset``, the resolver ladder in this
    module) and, worse, at the attenuation meet it makes
    ``attenuate(x, x) != canonicalize(x)``: the meet's launcher lookup is a
    ``{key: spec}`` dict (one entry wins), while ``canonicalize`` keeps every
    entry in the output list — so a duplicate key fails the module's own
    documented normal-form contract before it ever reaches the
    security-relevant clamp. Reject at the ingress boundary.
    """
    seen: set[tuple[str, str | None]] = set()
    for t in tools:
        if t.type == "mcp_toolset":
            key: tuple[str, str | None] = ("mcp_toolset", t.mcp_server_name)
        elif t.type == "custom":
            key = ("custom", t.name)
        else:
            key = ("builtin", t.type)
        if key in seen:
            label = t.name or t.mcp_server_name or t.type
            raise ValueError(f"duplicate tool entry {label!r} (identity key {key!r})")
        seen.add(key)


# ── names-only http_server declaration (Ask 3 from #939) ──────────────────────
#
# A workflow author may reference a grant the acting agent already holds by *name
# alone* — ``http_servers: ["davenant"]`` — instead of reconstructing the full
# ``HttpServerSpec(name=..., base_url=...)`` whose identity must match the agent's.
# The bare name is resolved against the acting agent's servers at the authoring
# edge (``aios.services.workflows``); the agent's ``base_url`` + frozen routes are
# then inherited launcher-frozen into storage exactly as the #949 identity-match
# path already does. This is **pure surface ergonomics**: a names-only entry can
# only resolve to a server the agent already has, so it grants no new authority
# and the run-time parent-wins-frozen resolution (keyed on the verbatim agent name)
# is untouched.
#
# Resolution lives in the service (it needs the acting agent); aliasing — declaring
# the agent's server under a *different* name with ``server_ref`` resolving against
# the alias — is an open fork (#953) deliberately NOT shipped here: it would require
# a run-time resolution change beyond the committed surface-only scope.

HttpServerRef = str | HttpServerSpec
"""An authoring-edge http_server entry: a bare name (names-only sugar, resolved
against the acting agent) or a full ``HttpServerSpec`` (identity-match, #949)."""


def resolve_http_server_refs(
    refs: list[HttpServerRef], agent_servers: list[HttpServerSpec]
) -> list[HttpServerSpec]:
    """Resolve names-only entries against the acting agent's ``http_servers``.

    Each ``str`` entry is replaced by ``HttpServerSpec(name=<name>, base_url=<the
    agent's base_url for that name>, routes=[])`` — an empty-routes identity spec
    the existing authoring gate then admits by identity and inherits frozen routes
    into. A name with no matching agent server raises ``ValueError`` (the author
    referenced a grant the agent does not hold). Full ``HttpServerSpec`` entries
    pass through verbatim (the #949 identity-match path).

    Resolution is by name; if the agent declares the same name at multiple
    ``base_url``s the first is taken (agents validate ``base_url`` uniqueness, not
    name uniqueness, so duplicate names are possible but the authoring gate then
    flags any genuine mismatch downstream).
    """
    by_name: dict[str, HttpServerSpec] = {}
    for s in agent_servers:
        by_name.setdefault(s.name, s)
    out: list[HttpServerSpec] = []
    for ref in refs:
        if isinstance(ref, str):
            agent_server = by_name.get(ref)
            if agent_server is None:
                raise ValueError(
                    f"http_servers references {ref!r}, which the acting agent does not grant"
                )
            out.append(HttpServerSpec(name=ref, base_url=agent_server.base_url, routes=[]))
        else:
            out.append(ref)
    return out


# ── Tool declaration ──────────────────────────────────────────────────────────


class ToolSpec(BaseModel):
    """One entry in an agent's ``tools`` list.

    For built-in tools, ``type`` is the tool name (``"bash"``, ``"read"``,
    etc.). For custom (client-executed) tools, ``type`` is ``"custom"`` and
    ``name``, ``description``, and ``input_schema`` are required. For MCP
    toolsets, ``type`` is ``"mcp_toolset"`` and ``mcp_server_name`` is
    required.

    ``enabled`` controls whether the tool is included in the schema sent to
    the model. Disabled tools are invisible to the model.

    ``permission`` controls execution policy for built-in tools:
    ``None`` or ``"always_allow"`` executes immediately (current default);
    ``"always_ask"`` leaves the call unresolved in the event log until
    the client confirms or denies via
    ``POST /sessions/:id/tool-confirmations``. Pending calls surface on
    ``Session.awaiting`` so clients can list what they need to act on.
    """

    model_config = ConfigDict(extra="forbid")

    type: BuiltinToolType | Literal["custom", "mcp_toolset"]
    name: str | None = None
    description: str | None = None
    input_schema: dict[str, Any] | None = None
    enabled: bool = True
    permission: PermissionPolicy | None = None
    # Override the registry default transport for a built-in (or the
    # system ``"both"`` default for a custom tool). ``None`` = inherit.
    # Ignored for ``type == "mcp_toolset"`` — per-server / per-tool MCP
    # transport overrides live on ``default_config`` and ``configs[]``,
    # paralleling ``permission_policy``.
    transport: ToolTransport | None = None

    # mcp_toolset fields
    mcp_server_name: str | None = None
    default_config: McpToolsetConfig | None = None
    configs: list[McpToolConfig] | None = None

    @model_validator(mode="before")
    @classmethod
    def _map_legacy_builtin_names(cls, data: Any) -> Any:
        """Map pre-rename builtin tool names to canonical ones (read-tolerance).

        Runs before field validation so a row persisted with a legacy ``type`` — the #1419
        ``invoke``/``invoke_agent``/``invoke_workflow``/``create_run``/``await_run`` set or the
        #1428 ``cancel_run`` — still validates against the post-rename ``BuiltinToolType``
        Literal. The ``create_run``/``await_run`` collapse to ``call_workflow`` is deduped at the
        list level (``to_openai_tools`` + migrations 0116/0117), not here.

        #1574: the tolerance map is read from the retirement REGISTRY (#1573) via
        :func:`aios.retirements.registry.tolerated_rename_map`, which yields a token's canonical
        successor ONLY while its descriptor's ``contract_rev IS NULL``. A token whose descriptor
        has stamped ``contract_rev`` is absent from the map, so it is no longer remapped and would
        correctly fail validation against the post-rename Literal. Staying ``mode="before"`` keeps
        this per-site: it travels to every ``ToolSpec.model_validate`` site automatically.

        Each fire records ``retirement_tolerance_hits{token}`` + stamps ``last_seen`` via
        :func:`aios.retirements.telemetry.record_tolerance_hit` — **corroboration / telemetry
        only**. The gate is the registry's ``contract_rev IS NULL`` predicate above; the telemetry
        is never consulted to decide whether to tolerate, and a metrics failure can never wedge a
        read.
        """
        if isinstance(data, dict):
            t = data.get("type")
            if isinstance(t, str):
                # Registry-sourced, contract_rev-gated map (the GATE).
                successor = tolerated_rename_map().get(t)
                if successor is not None:
                    # Corroboration only — NEVER the gate. Recorded after the gate decided.
                    record_tolerance_hit(t)
                    data = {**data, "type": successor}
        return data

    @model_validator(mode="after")
    def _check_type_fields(self) -> ToolSpec:
        if self.type == "custom":
            missing = [
                f for f in ("name", "description", "input_schema") if getattr(self, f) is None
            ]
            if missing:
                raise ValueError(f"custom tools require: {', '.join(missing)}")
            # ``mcp__`` is the reserved MCP namespace (see is_mcp_tool_name): a
            # custom tool with that prefix is classified as an MCP call by
            # _classify_tool_call BEFORE the custom fallback, so it routes to
            # the MCP dispatcher (erroring as an unknown server) and is never
            # held for the client. Reject it here, at the operator-controlled
            # definition layer, rather than letting it silently never work.
            if self.name is not None and is_mcp_tool_name(self.name):
                raise ValueError(
                    f"custom tool name {self.name!r} must not start with the reserved "
                    "'mcp__' prefix (it namespaces MCP-dispatched tools)"
                )
        elif self.type == "mcp_toolset":
            if self.mcp_server_name is None:
                raise ValueError("mcp_toolset requires mcp_server_name")
            if self.configs:
                # A ``configs[]`` entry is keyed on ``name`` by every reader (the
                # resolver ladder in this module, the attenuation meet's
                # ``_canon_toolset_name_triple``). Two entries sharing a name are
                # unreachable dead config for the runtime resolvers (first match
                # wins in the ``for cfg in spec.configs`` scan) and, worse, break
                # the attenuation module's own normal-form contract
                # ``attenuate(x, x) == canonicalize(x)``: ``canonicalize`` sorts
                # by name and keeps every entry, while the meet re-derives one
                # resolved triple per distinct name — so a duplicate silently
                # drops an entry under the meet but not under canonicalize.
                seen_names: set[str] = set()
                for cfg in self.configs:
                    if cfg.name in seen_names:
                        raise ValueError(
                            f"duplicate configs[] entry {cfg.name!r} for mcp_toolset "
                            f"{self.mcp_server_name!r}"
                        )
                    seen_names.add(cfg.name)
        return self


class AgentCreate(BaseModel):
    """Request body for `POST /v1/agents`."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=128)
    model: str = Field(
        min_length=1,
        description="LiteLLM model string, e.g. 'anthropic/claude-opus-4-6'.",
    )
    system: str = Field(default="", description="System prompt; empty by default.")
    tools: list[ToolSpec] = Field(default_factory=list)
    skills: list[AgentSkillRef] = Field(default_factory=list)
    mcp_servers: list[McpServerSpec] = Field(default_factory=list)
    http_servers: list[HttpServerSpec] = Field(default_factory=list)
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    litellm_extra: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Provider-specific LiteLLM kwargs merged into every model "
            "request for this agent.  Reasoning depth: "
            "``reasoning_effort`` ('low'/'medium'/'high' everywhere; "
            "'xhigh'/'max' on Claude only) is the portable knob — "
            "LiteLLM translates it per provider, e.g. on current Claude "
            "models into ``thinking: {type: 'adaptive'}`` plus "
            "``output_config: {effort: ...}``.  Provider-native params "
            "(Anthropic ``thinking``, ``output_config``) compose with "
            "it and take precedence over the translated values when "
            "both are set.  Other common shapes: OpenRouter "
            "``extra_body.provider.order`` for provider pinning, raw "
            "sampling knobs (``temperature``, ``max_tokens``), "
            "``api_base`` for self-hosted inference.  Validated by "
            "LiteLLM / the provider; bad kwargs surface as tool-path "
            "errors the model sees.  Security: ``api_base`` redirects "
            "the model call — treat operator-set agents as trusted "
            "and don't accept this field from untrusted principals."
        ),
    )
    window_min: int = Field(default=50_000, ge=1)
    window_max: int = Field(default=150_000, ge=1)
    preempt_policy: PreemptPolicy = Field(
        default="wait",
        description=(
            "Whether a new wake-eligible event (e.g. a user message) arriving "
            "mid-step cancels the in-flight model call so the step restarts "
            "against fresh context ('preempt'), or waits for the step to "
            "finish ('wait', default)."
        ),
    )

    @model_validator(mode="after")
    def _validate_http_servers(self) -> AgentCreate:
        validate_http_servers(self.http_servers)
        validate_mcp_servers(self.mcp_servers)
        validate_tools(self.tools)
        return self


class AgentUpdate(BaseModel):
    """Request body for ``PUT /v1/agents/{id}``.

    All config fields are optional; omitted fields are preserved. The
    ``version`` field is required for optimistic concurrency — it must match
    the current version. If the update produces a change, a new version is
    created; otherwise the existing version is returned unchanged.
    """

    model_config = ConfigDict(extra="forbid")

    version: int = Field(description="Current version for optimistic concurrency.")
    name: str | None = Field(default=None, min_length=1, max_length=128)
    model: str | None = Field(default=None, min_length=1)
    system: str | None = None
    tools: list[ToolSpec] | None = None
    skills: list[AgentSkillRef] | None = None
    mcp_servers: list[McpServerSpec] | None = None
    http_servers: list[HttpServerSpec] | None = None
    description: str | None = None
    metadata: dict[str, Any] | None = None
    litellm_extra: dict[str, Any] | None = None
    window_min: int | None = Field(default=None, ge=1)
    window_max: int | None = Field(default=None, ge=1)
    preempt_policy: PreemptPolicy | None = None

    @model_validator(mode="after")
    def _validate_http_servers(self) -> AgentUpdate:
        if self.http_servers is not None:
            validate_http_servers(self.http_servers)
        if self.mcp_servers is not None:
            validate_mcp_servers(self.mcp_servers)
        if self.tools is not None:
            validate_tools(self.tools)
        return self


class Agent(BaseModel):
    """Read view of an agent (always the latest version)."""

    id: str
    version: int
    name: str
    model: str
    system: str
    tools: list[ToolSpec]
    skills: list[AgentSkillRef] = Field(default_factory=list)
    mcp_servers: list[McpServerSpec]
    http_servers: list[HttpServerSpec] = Field(default_factory=list)
    description: str | None
    metadata: dict[str, Any]
    litellm_extra: dict[str, Any] = Field(default_factory=dict)
    window_min: int
    window_max: int
    preempt_policy: PreemptPolicy = "wait"
    created_by: Actor | None = None
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None = None


class AgentVersion(BaseModel):
    """Read view of a specific agent version from the version history."""

    agent_id: str
    version: int
    model: str
    system: str
    tools: list[ToolSpec]
    skills: list[AgentSkillRef] = Field(default_factory=list)
    mcp_servers: list[McpServerSpec]
    http_servers: list[HttpServerSpec] = Field(default_factory=list)
    litellm_extra: dict[str, Any] = Field(default_factory=dict)
    window_min: int
    window_max: int
    preempt_policy: PreemptPolicy = "wait"
    created_at: datetime


# ── Step surface (harness read-model) ────────────────────────────────────────


class AgentBinding(BaseModel):
    """The step surface's identity when it is backed by an agent.

    Covers *all three* agented cases the harness resolves to an agent
    identity: a "latest" ``Agent``, a version-pinned ``AgentVersion``, and an
    **agented** workflow child (``parent_run_id`` set, ``agent_id`` present).
    Sibling runs of the same ``(agent_id, version)`` share this identity so the
    #1391 MCP raw-discovery cache keeps sharing across them.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["agent"] = "agent"
    agent_id: str
    version: int


class GenericChildBinding(BaseModel):
    """The step surface's identity for a generic workflow child (no agent).

    A generic child (``parent_run_id`` set, ``agent_id`` is ``None``) carries
    only its own per-run-attenuated surface. It keys the #1391 cache on its own
    ``session_id`` — never on an agent identity — so distinct children never
    share a discovery slot. Replaces the ``agent_id=""``/``version=0`` sentinel.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["generic_child"] = "generic_child"
    session_id: str


StepBinding = Annotated[
    AgentBinding | GenericChildBinding,
    Field(discriminator="kind"),
]


class StepSurface(BaseModel):
    """What a session runs as at step time — the harness's single read-model.

    Nominal replacement for the ``Agent | AgentVersion`` structural union (the
    two wire read-models) plus the ``agent_id=""``/``version=0`` sentinel that
    encoded "no agent at all". Carries **exactly** the ten config fields the
    harness consumes off the loaded surface (verified by grep over every
    caller — nothing reads ``name``/``metadata``/``description``/``created_at``
    off it) plus a discriminated :data:`StepBinding` identity.

    ``binding`` is a total, two-arm discriminated kind — no ``.id``/``.agent_id``
    spelling divergence to duck-type across and no sentinel to forget, so a
    d3695683-class identity misclassification (the #1554 cache poisoning) is a
    mypy error at every consumer instead of a latent runtime bug.
    """

    model_config = ConfigDict(frozen=True)

    tools: list[ToolSpec]
    mcp_servers: list[McpServerSpec]
    http_servers: list[HttpServerSpec] = Field(default_factory=list)
    model: str
    system: str
    skills: list[AgentSkillRef] = Field(default_factory=list)
    litellm_extra: dict[str, Any] = Field(default_factory=dict)
    window_min: int
    window_max: int
    preempt_policy: PreemptPolicy
    binding: StepBinding


# ── Tool-name + permission helpers ───────────────────────────────────────────


def is_mcp_tool_name(name: str) -> bool:
    """True if ``name`` is the namespaced form ``mcp__<server>__<tool>``."""
    return name.startswith("mcp__")


# ── Outbound-suppression classification (#710) ───────────────────────────────

# HTTP methods that pass through unchanged under outbound suppression by
# default (reads). Everything else (POST/PUT/PATCH/DELETE) is a write and is
# suppressed. A per-route ``suppress`` override flips either direction.
_SUPPRESSION_READ_METHODS: frozenset[str] = frozenset({"GET", "HEAD", "OPTIONS"})


def http_route_suppressed(route: HttpRouteSpec, method: str) -> bool:
    """Decide whether a matched HTTP route is suppressed for ``method``.

    The default classifier is the HTTP method — ``GET`` (and the other safe
    verbs) read and pass through; ``POST``/``PUT``/``PATCH``/``DELETE`` write
    and are suppressed. The route's optional ``suppress`` override wins when
    set: ``True`` suppresses a side-effecting read, ``False`` lets a read-only
    write through. This is *only* consulted when the session's
    ``outbound_suppression`` is ``"on"`` — callers gate on that first.
    """
    if route.suppress is not None:
        return route.suppress
    return method.upper() not in _SUPPRESSION_READ_METHODS


def mcp_tool_suppressed(name: str, agent_tools: list[ToolSpec]) -> bool:
    """Decide whether an MCP tool call is suppressed under outbound suppression.

    MCP has no method convention, so the policy is *default-deny*: every MCP
    call is suppressed UNLESS its per-tool ``McpToolConfig.read_allow`` is set
    (an operator opt-in for a known-safe read). Resolution mirrors
    :func:`resolve_mcp_enabled`: a matching ``configs[]`` entry decides; absent
    one, the tool is suppressed. An undeclared/unmatched tool is suppressed
    (the safe default). Callers gate on ``outbound_suppression == "on"`` first.
    """
    parts = name.split("__", 2)
    if len(parts) < 3:
        return True
    server_name = parts[1]
    tool_name = parts[2]
    for spec in agent_tools:
        if spec.type == "mcp_toolset" and spec.mcp_server_name == server_name:
            if spec.configs:
                for cfg in spec.configs:
                    if cfg.name == tool_name:
                        return not cfg.read_allow
            return True
    return True


def resolve_permission(name: str, agent_tools: list[ToolSpec]) -> PermissionPolicy | None:
    """Look up the permission policy for a built-in or custom tool by name."""
    for spec in agent_tools:
        tool_name = spec.name if spec.type == "custom" else spec.type
        if tool_name == name:
            return spec.permission
    return None


def resolve_mcp_permission(name: str, agent_tools: list[ToolSpec]) -> PermissionPolicy | None:
    """Look up the permission policy for an MCP tool by namespaced name.

    Precedence (mirrors the broker's resolution so the model path and CLI
    path agree on overrides): per-tool ``configs[]`` entry → ``default_config``
    → bare ``ToolSpec.permission``. Returns ``None`` when nothing is set;
    callers then fall back to ``AIOS_DEFAULT_MCP_PERMISSION_POLICY``.
    """
    parts = name.split("__", 2)
    if len(parts) < 3:
        return None
    server_name = parts[1]
    tool_name = parts[2]
    for spec in agent_tools:
        if spec.type == "mcp_toolset" and spec.mcp_server_name == server_name:
            if spec.configs:
                for cfg in spec.configs:
                    if cfg.name == tool_name:
                        return cfg.permission_policy.type if cfg.permission_policy else None
            if spec.default_config and spec.default_config.permission_policy:
                return spec.default_config.permission_policy.type
            return spec.permission
    return None


def resolve_mcp_transport(name: str, agent_tools: list[ToolSpec]) -> ToolTransport | None:
    """Look up the transport classification for an MCP tool by namespaced name.

    Precedence parallels :func:`resolve_mcp_permission`: per-tool
    ``configs[]`` entry → ``default_config.transport``. Returns ``None``
    when no override is set — callers fall back to the system default
    ``"both"``.
    """
    parts = name.split("__", 2)
    if len(parts) < 3:
        return None
    server_name = parts[1]
    tool_name = parts[2]
    for spec in agent_tools:
        if spec.type == "mcp_toolset" and spec.mcp_server_name == server_name:
            if spec.configs:
                for cfg in spec.configs:
                    if cfg.name == tool_name:
                        return cfg.transport
            if spec.default_config:
                return spec.default_config.transport
            return None
    return None


def resolve_mcp_enabled(name: str, agent_tools: list[ToolSpec]) -> bool:
    """Resolve whether an MCP tool is enabled given the agent's config.

    Precedence parallels :func:`resolve_mcp_permission`: per-tool
    ``configs[]`` entry → ``default_config.enabled`` → ``True`` (the
    field default). Returns ``False`` if no matching ``mcp_toolset``
    entry exists at all — a tool the agent hasn't declared a toolset
    for is implicitly off.
    """
    parts = name.split("__", 2)
    if len(parts) < 3:
        return False
    server_name = parts[1]
    tool_name = parts[2]
    for spec in agent_tools:
        if spec.type == "mcp_toolset" and spec.mcp_server_name == server_name:
            if spec.configs:
                for cfg in spec.configs:
                    if cfg.name == tool_name:
                        return cfg.enabled
            if spec.default_config is not None:
                return spec.default_config.enabled
            return True
    return False


def resolve_builtin_transport(name: str, agent_tools: list[ToolSpec]) -> ToolTransport | None:
    """Look up the transport override for a built-in or custom tool by name.

    Returns the matching ``ToolSpec.transport`` (which may be ``None`` if
    the operator left it unset, meaning "inherit the registry default").
    Returns ``None`` if no matching entry exists at all.
    """
    for spec in agent_tools:
        tool_name = spec.name if spec.type == "custom" else spec.type
        if tool_name == name:
            return spec.transport
    return None
