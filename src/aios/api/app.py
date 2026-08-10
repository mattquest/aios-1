"""FastAPI app factory.

Builds the app, wires in the routers, the exception handlers, and the
lifespan that opens/closes the asyncpg pool and constructs the CryptoBox.
Also mounts the auto-reflected MCP server at ``/mcp`` so agents can
operate the management plane via the same Bearer auth as the HTTP API.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI
from fastapi.routing import APIRoute

from aios.api.deps import require_bearer_auth
from aios.api.middleware import RequestLoggingMiddleware
from aios.api.routers import (
    accounts,
    agents,
    connections,
    connectors,
    context_variables,
    environments,
    health,
    memory_stores,
    model_providers,
    runtime_tokens,
    session_templates,
    sessions,
    skills,
    tasks,
    triggers_ingest,
    vaults,
    workflows,
)
from aios.api.strict_query_params import reject_unknown_query_params
from aios.config import get_settings
from aios.crypto.vault import CryptoBox
from aios.db import queries
from aios.db.pool import create_pool
from aios.errors import install_exception_handlers
from aios.harness import runtime
from aios.jobs.app import app as procrastinate_app
from aios.logging import configure_logging, get_logger
from aios.retirements.boot_gate import (
    RetirementsNotAdmissible,
    assert_retirements_admissible,
)
from aios.sandbox.volumes import attachments_root, memory_stores_root, uploads_root

# How long the api startup loops between boot-admission proof attempts while the
# DB is behind / unreachable. Short enough that readiness flips green promptly
# once the post-deploy migrate lands; long enough not to hammer the DB.
_BOOT_GATE_RETRY_SECONDS = 2.0


async def _await_retirements_admissible(pool: Any, log: Any) -> None:
    """Loop the fail-closed boot-admission gate until the DB is provably safe.

    Refuses to return — keeping the api unready (``app.state.retirements_ok``
    stays ``False``, so ``/ready`` answers 503) — until
    :func:`assert_retirements_admissible` passes. Under Coolify's rolling
    deploy this is exactly the window where the old healthy container keeps
    serving while the new one waits out the post-deploy migrate (#1575).
    """
    logged_wait = False
    while True:
        try:
            await assert_retirements_admissible(pool)
            return
        except RetirementsNotAdmissible as exc:
            if not logged_wait:
                log.warning("api.boot_gate.refusing_readiness", reason=str(exc))
                logged_wait = True
            await asyncio.sleep(_BOOT_GATE_RETRY_SECONDS)


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    log = get_logger("aios.api")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        log.info("api.startup", db_url=_redact_dsn(settings.db_url))
        pool = await create_pool(settings.db_url, max_size=settings.db_pool_max_size)
        crypto_box = CryptoBox.from_base64(settings.vault_key.get_secret_value())
        async with pool.acquire() as conn:
            await queries.audit_credentialless_root(conn)
        await procrastinate_app.open_async()
        app.state.pool = pool
        app.state.crypto_box = crypto_box
        app.state.procrastinate = procrastinate_app
        app.state.db_url = settings.db_url
        # Readiness stays RED until the fail-closed boot-admission gate proves
        # the live DB is safe for this code. ``/ready`` reads this flag, so a
        # behind / residue-bearing / unreachable DB never lets the new
        # container serve while the old one is still healthy (#1575).
        app.state.retirements_ok = False

        # #959: the api runs as uid 1000 and can't repair ownership (no
        # CAP_CHOWN). If the worker (root) created a shared root first and
        # left it root-owned, ``POST /sessions/{id}/files`` 500s on mkdir.
        # The api can't fix it — but it can refuse to fail silently. Log
        # loudly and keep serving non-FS routes; the worker's startup
        # repair pass is what actually heals the tree.
        euid = os.geteuid()
        for p in (
            settings.workspace_root,
            uploads_root(),
            attachments_root(),
            memory_stores_root(),
        ):
            if p.exists() and not os.access(p, os.W_OK):
                try:
                    st_uid = p.stat().st_uid
                except OSError:
                    st_uid = -1
                log.warning("api.workspace_unwritable", path=str(p), euid=euid, st_uid=st_uid)

        # The ``/context`` endpoint (issue #60) reuses the worker's
        # ``compose_step_context`` → ``compute_step_prelude`` path, which
        # reaches for ``runtime.require_crypto_box()`` to decrypt per-vault
        # MCP OAuth tokens and ``runtime.require_tool_provider()`` to merge
        # connection-declared tools (#328 PR 4). Mirror the worker's
        # globals on the API side so the shared composer works from either
        # process.
        from aios_connectors.providers import SubsystemToolProvider

        prev_crypto = runtime.crypto_box
        prev_tool_provider = runtime.tool_provider
        runtime.crypto_box = crypto_box
        runtime.tool_provider = SubsystemToolProvider()

        # Fail-closed boot-admission gate (#1575): block here — readiness RED —
        # until the live DB is proven safe for this code (alembic at/past every
        # contract_rev AND zero live residue on every registered surface). This
        # runs on EVERY boot, including raw restores, and is never cached.
        await _await_retirements_admissible(pool, log)
        app.state.retirements_ok = True
        log.info("api.boot_gate.admitted")

        try:
            yield
        finally:
            log.info("api.shutdown")
            runtime.crypto_box = prev_crypto
            runtime.tool_provider = prev_tool_provider
            await procrastinate_app.close_async()
            await pool.close()

    app = FastAPI(
        title="aios",
        version="0.1.0",
        description="Open-source agent runtime: Postgres-backed sessions, "
        "Docker sandbox, any LiteLLM model.",
        lifespan=lifespan,
        dependencies=[Depends(reject_unknown_query_params)],
        # One schema per Pydantic model (not a Foo-Input / Foo-Output pair).
        # We use distinct ``Foo`` / ``FooCreate`` / ``FooUpdate`` types instead
        # of the read-only-field split, and the hyphenated split-schema names
        # break openapi-python-client when it generates the typed SDK.
        separate_input_output_schemas=False,
    )
    install_exception_handlers(app)
    # Outermost middleware: a pure-ASGI request logger that runs in the same
    # task context as the route (so an ``account_id`` bound by the auth dep is
    # visible on its line) and brackets the whole stack, including the exception
    # handlers, in its duration measurement.
    app.add_middleware(RequestLoggingMiddleware)
    app.include_router(health.router)
    app.include_router(accounts.router)
    app.include_router(environments.router)
    app.include_router(agents.router)
    app.include_router(sessions.router)
    app.include_router(skills.router)
    app.include_router(context_variables.router)
    app.include_router(vaults.router)
    app.include_router(memory_stores.router)
    app.include_router(model_providers.router)
    app.include_router(connections.router)
    app.include_router(runtime_tokens.router)
    app.include_router(triggers_ingest.router)
    app.include_router(connectors.router)
    app.include_router(session_templates.router)
    app.include_router(workflows.router)
    app.include_router(workflows.runs_router)
    app.include_router(tasks.router)
    _mount_mcp(app)
    return app


MCP_INSTRUCTIONS = """\
aios is a Postgres-backed agent runtime. The management plane is organized
around five resource types: agents (versioned config), sessions (live
execution contexts), environments (sandbox templates), memory stores
(shared file-like memory), and vaults (encrypted credentials). Resources
attach to sessions to grant agent read/write access.

Conventions:
- ``archive`` is reversible soft-removal preserving audit history;
  ``delete`` is hard-removal, no audit trail. Prefer archive unless you
  specifically need rows gone.
- Listed resources exclude archived by default unless ``include_archived``
  is supported on the operation.
- Mutations marked ``destructiveHint`` should be confirmed before invoking.
"""


def _mount_mcp(app: FastAPI) -> None:
    """Mount fastapi-mcp at ``/mcp`` after all routers are registered.

    The MCP tool surface is whatever FastAPI's OpenAPI spec says it is,
    minus operations whose ``x-codegen.targets`` excludes ``"mcp"``.
    Today that excludes the SSE / long-poll session-event routes (no
    JSON response shape) and the connectors RPC routes (raw
    ``dict[str, Any]`` envelopes that need Pydantic models authored
    before they MCP cleanly).
    """
    # fastapi-mcp doesn't ship a py.typed marker yet, so mypy can't see
    # the package's actual signatures.
    from fastapi_mcp import AuthConfig, FastApiMCP

    mcp = FastApiMCP(
        app,
        name="aios",
        description=(
            "aios management plane: agents, sessions, environments, vaults, "
            "memory stores, skills, connections, session templates."
        ),
        exclude_operations=_compute_mcp_excluded_operations(app),
        auth_config=AuthConfig(dependencies=[Depends(require_bearer_auth)]),
    )
    _apply_mcp_polish(app, mcp, instructions=MCP_INSTRUCTIONS)
    mcp.mount_http(mount_path="/mcp")


def _iter_operations(app: FastAPI) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """Yield ``(operation_id, http_verb, x_codegen_dict)`` for every operation.

    Walks ``app.routes`` directly instead of the OpenAPI document: building
    the document costs a full Pydantic schema generation pass (~200 ms),
    and fastapi-mcp builds its own copy anyway, so going through
    ``app.openapi()`` here made every ``create_app()`` pay for the schema
    twice.  The operationId is derived exactly as FastAPI's
    ``get_openapi_operation_metadata`` does — ``route.operation_id or
    route.unique_id`` — so the ids match the document (and fastapi-mcp's
    tool names) verbatim.  The ``x_codegen_dict`` is the raw ``x-codegen``
    extension content (or ``{}`` if absent); each caller projects out the
    sub-key it needs.
    """
    for route in app.routes:
        if not isinstance(route, APIRoute) or not route.include_in_schema:
            continue
        op_id = route.operation_id or route.unique_id
        x_codegen = (route.openapi_extra or {}).get("x-codegen", {})
        # sorted() so multi-method routes yield in a deterministic order
        # (route.methods is a set; every aios route is single-method, but
        # fastapi-mcp's own /mcp transport route is GET+POST+DELETE).
        for method in sorted(route.methods):
            yield op_id, method.lower(), x_codegen


def _verb_default_annotations(verb: str) -> dict[str, bool]:
    """Map HTTP verb to default MCP tool annotation hints.

    POST is the asymmetric case: REST convention says POST is additive, but
    operations like ``.../archive`` and ``.../detach`` are POST-but-destructive
    and override via ``x-codegen.mcp`` per route — hence the empty default
    rather than a destructive one.
    """
    if verb == "get":
        return {"readOnlyHint": True}
    if verb == "delete":
        return {"destructiveHint": True}
    if verb == "put":
        return {"destructiveHint": True, "idempotentHint": True}
    return {}


_RESPONSE_BLOCK_SEPARATOR = "\n\n### Responses:\n"

# JSON Schema keywords whose presence on a branch means the branch carries
# enough structure that lifting its keys onto a parent during nullable-anyOf
# flattening would change semantics.  ``_flatten_nullable_anyof`` only
# flattens when the non-null branch's keys are a subset of the
# scalar-metadata allow-list, so this set is the inverse safety net.
_FLATTEN_SAFE_KEYS = frozenset(
    {
        "type",
        "enum",
        "const",
        "format",
        "pattern",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "description",
        "default",
        "examples",
        "title",
    }
)


def _strip_response_block(description: str) -> str | None:
    """Drop fastapi-mcp's auto-appended ``### Responses:`` example block."""
    # ``rpartition`` guards against a route docstring legitimately containing
    # the literal separator — fastapi-mcp's block is always last-appended.
    head, _sep, _rest = description.rpartition(_RESPONSE_BLOCK_SEPARATOR)
    return head.rstrip() or None


def _drop_redundant_titles(schema: dict[str, Any]) -> None:
    """Recursively drop ``title`` fields whose value matches the property key."""
    for key, prop in schema.get("properties", {}).items():
        if prop.get("title") == key:
            prop.pop("title")
        _drop_redundant_titles(prop)
    if (items := schema.get("items")) is not None:
        _drop_redundant_titles(items)
    for combinator in ("anyOf", "oneOf", "allOf"):
        for branch in schema.get(combinator, []):
            _drop_redundant_titles(branch)


def _flatten_nullable_anyof(schema: dict[str, Any]) -> None:
    """Rewrite ``anyOf: [T, {"type": "null"}]`` to ``T`` with ``type: [..., "null"]``.

    Only flattens when the non-null branch's keys are a subset of
    :data:`_FLATTEN_SAFE_KEYS` — anything else (nested ``properties``,
    further combinators) keeps its original ``anyOf`` form to avoid
    silently changing the semantics of complex unions.
    """
    for prop in schema.get("properties", {}).values():
        _flatten_nullable_anyof(prop)
    if (items := schema.get("items")) is not None:
        _flatten_nullable_anyof(items)
    for combinator in ("anyOf", "oneOf", "allOf"):
        for branch in schema.get(combinator, []):
            _flatten_nullable_anyof(branch)

    branches = schema.get("anyOf")
    if not isinstance(branches, list) or len(branches) != 2:
        return
    b0, b1 = branches
    if b0 == {"type": "null"}:
        other = b1
    elif b1 == {"type": "null"}:
        other = b0
    else:
        return
    other_type = other.get("type")
    if not isinstance(other_type, str):
        return
    if not other.keys() <= _FLATTEN_SAFE_KEYS:
        return
    del schema["anyOf"]
    schema.update(other)
    schema["type"] = [other_type, "null"]


def _apply_mcp_polish(app: FastAPI, mcp: Any, *, instructions: str) -> None:
    """Post-construction patch: per-tool annotations, schema cleanup, server instructions.

    fastapi-mcp 0.4 doesn't infer annotations from HTTP verbs and exposes
    no construction-time hook for them, so we walk the spec, build verb-
    based defaults, layer the route's ``x-codegen.mcp`` overrides on top,
    and assign the result to each ``mcp.tools[i].annotations``.  Also
    strips three sources of schema noise — auto-appended response-example
    blocks, redundant ``title`` fields, and verbose ``anyOf`` nullable
    encodings — so the tool list the model sees is roughly half its raw
    fastapi-mcp size.  Finally sets ``mcp.server.instructions`` (the MCP
    handshake field, inlined into the calling LLM's system prompt by
    clients that respect it).
    """
    from mcp.types import ToolAnnotations

    op_meta: dict[str, tuple[str, dict[str, Any]]] = {
        op_id: (verb, codegen.get("mcp", {})) for op_id, verb, codegen in _iter_operations(app)
    }

    for tool in mcp.tools:
        # Direct lookup, not .get + None-skip: fastapi-mcp names tools by
        # the operationId of the same routes op_meta walks (both derive it
        # via ``route.operation_id or route.unique_id``), so every
        # tool.name must be an operationId we've seen. A KeyError here is
        # a real invariant break worth surfacing, not a silent annotation
        # dropout.
        verb, overrides = op_meta[tool.name]
        annotations = _verb_default_annotations(verb) | overrides
        if annotations:
            tool.annotations = ToolAnnotations(**annotations)
        if tool.description is not None:
            tool.description = _strip_response_block(tool.description)
        _drop_redundant_titles(tool.inputSchema)
        _flatten_nullable_anyof(tool.inputSchema)

    mcp.server.instructions = instructions


def _compute_mcp_excluded_operations(app: FastAPI) -> list[str]:
    """Return operationIds whose ``x-codegen.targets`` excludes ``"mcp"``.

    Routes without an ``x-codegen.targets`` list default to included in
    every generator target (per the contract documented in #267); only
    explicit ``targets`` lists that omit ``"mcp"`` exclude.
    """
    return [
        op_id
        for op_id, _verb, codegen in _iter_operations(app)
        if (targets := codegen.get("targets")) is not None and "mcp" not in targets
    ]


def _redact_dsn(dsn: str) -> str:
    """Strip the password from a Postgres DSN for safe logging."""
    if "@" not in dsn or "://" not in dsn:
        return dsn
    scheme, _, rest = dsn.partition("://")
    auth, _, host_part = rest.partition("@")
    if ":" in auth:
        user, _, _ = auth.partition(":")
        return f"{scheme}://{user}:***@{host_part}"
    return dsn


# An app instance importable as ``aios.api.app:app`` for ``uvicorn``.
app: Any = create_app()
