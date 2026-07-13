"""Session queries.

A subsystem module of the ``aios.db.queries`` package — see ``__init__`` for the
shared scoping helpers and the package-level re-export contract. Raw SQL against
asyncpg, same conventions as the rest of the package.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from types import EllipsisType
from typing import Any

import asyncpg

from aios.actors import actor_columns, actor_from_row
from aios.db import queries
from aios.db.queries import (
    _archive_scoped,
    _build_set_assignments,
    _get_scoped,
    _list_scoped,
    open_request_anti_join,
)
from aios.db.queries.clone_policy import (
    EVENTS_POLICY,
    SESSION_GITHUB_REPOSITORIES_POLICY,
    SESSION_MEMORY_STORES_POLICY,
    SESSIONS_POLICY,
    TRIGGERS_POLICY,
    build_projection,
)
from aios.errors import (
    ConflictError,
    NotFoundError,
    RateLimitedError,
)
from aios.ids import (
    EVENT,
    GITHUB_REPOSITORY,
    SESSION,
    TRIGGER,
    make_id,
)
from aios.models.agents import HttpServerSpec, McpServerSpec, ToolSpec, load_tool_specs
from aios.models.attenuation import Surface
from aios.models.sessions import (
    Err,
    Obligation,
    Outcome,
    Session,
    SessionStatus,
    SessionUsage,
)
from aios.retirements.epoch import TOOLS_VOCAB_EPOCH

# ─── sessions ─────────────────────────────────────────────────────────────────

# O(1) session status derivation from five maintained scalar columns
# (``last_reacted_seq``, ``open_tool_call_count``, ``last_error_seq``,
# ``last_user_seq``, ``last_stimulus_seq``). These scalars are updated
# transactionally inside ``append_event`` and backfilled by migration 0066.
# The status predicate is pure column arithmetic — no correlated subqueries
# over ``events``.
#
# ``errored`` = last_error_seq > 0 AND last_error_seq > last_user_seq
# ``active``  = (last_stimulus_seq > last_reacted_seq OR open_tool_call_count > 0)
#               AND NOT errored
# ``idle``    = otherwise
#
# ``last_stimulus_seq`` — NOT ``last_event_seq`` — is the unreacted-stimulus
# watermark: it is the max seq of non-assistant messages (role <> 'assistant';
# user + tool). ``last_event_seq`` includes the session's OWN assistant replies,
# so after a normal idle turn (user → assistant reply) ``last_event_seq >
# last_reacted_seq`` and the session would read wrongly as ``active`` — driving
# one extra model step (#749). This expr is exactly the pre-#732 derivation
# ``EXISTS(non-assistant message with seq > last_reacted_seq)``, as a scalar.
#
# The active/errored booleans have ONE source each — the alias-parameterized
# generators below. The read path binds them at the ``sessions`` alias
# (``_SESSION_ACTIVE_EXPR`` / ``_SESSION_ERRORED_EXPR``); the sweep's
# ``CANDIDATE_ROWS_SQL`` / ``ERRORED_SESSIONS_SQL`` compose the SAME generator
# at its ``s`` alias. The sweep candidate filter and the read-path status
# predicate therefore agree by construction — divergence (the worker waking
# with no progress, or skipping a session that needs inference) is impossible.


def session_errored_predicate(alias: str) -> str:
    """SQL boolean fragment: is the session in the terminal ``errored`` state?

    One source for the errored predicate, alias-parameterized so both the
    read path (``sessions`` alias) and the sweep's ``ERRORED_SESSIONS_SQL``
    (``s`` alias) compose the IDENTICAL boolean. A session is errored when its
    latest error post-dates the latest user message; a later user message bumps
    ``last_user_seq`` and flips the inequality (the recovery semantics).

    Also reused by ``lock_active_session_for_update`` and the clone gate.
    """
    return f"({alias}.last_error_seq > 0 AND {alias}.last_error_seq > {alias}.last_user_seq)"


def session_active_predicate(alias: str, *, stimulus_floor_param: str | None = None) -> str:
    """SQL boolean fragment: does the session have work the model must react to?

    One source for the active predicate, alias-parameterized so the read-path
    status derivation (``sessions`` alias) and the sweep's wake candidate filter
    (``CANDIDATE_ROWS_SQL``, ``s`` alias) compose the IDENTICAL boolean — they
    MUST agree or the worker wakes with no progress (#155) / skips inference.

    ``stimulus_floor_param`` (a positional placeholder like ``"$2"``) raises the
    reacted watermark to ``GREATEST(last_reacted_seq, <param>)`` — the #253
    preemption trigger's re-parameterization against the in-flight step's
    context watermark (``sweep.CANDIDATE_ROWS_FLOORED_SQL``): during a step the
    last-appended assistant's watermark is the *previous* step's, so the
    committed column would re-admit the very stimuli the in-flight step is
    already reacting to. The default emits the byte-identical committed
    predicate.
    """
    reacted = f"{alias}.last_reacted_seq"
    if stimulus_floor_param is not None:
        reacted = f"GREATEST({alias}.last_reacted_seq, {stimulus_floor_param})"
    return (
        f"(({alias}.last_stimulus_seq > {reacted}"
        f" OR {alias}.open_tool_call_count > 0)"
        f" AND NOT {session_errored_predicate(alias)})"
    )


_SESSION_ERRORED_EXPR = session_errored_predicate("sessions")

_SESSION_ACTIVE_EXPR = session_active_predicate("sessions")

# Read-path status label ({active, idle, archived}). ``archived`` is terminal
# and DOMINATES the active/idle derivation: a soft-archived session
# (``archived_at`` set) — e.g. a workflow ``agent()`` child that reclaimed itself
# on idle (``archive_when_idle``, #831) — never wakes again, so reporting it as
# ``active``/``idle`` would be a lie. This wraps, but does NOT alter,
# ``_SESSION_ACTIVE_EXPR``: the sweep's candidate filter uses that predicate
# directly under its own ``archived_at IS NULL`` WHERE clause and is untouched.
_SESSION_STATUS_EXPR = (
    "CASE WHEN sessions.archived_at IS NOT NULL THEN 'archived' "
    f"WHEN {_SESSION_ACTIVE_EXPR} THEN 'active' ELSE 'idle' END"
)


def _row_to_session(row: asyncpg.Record) -> Session:
    raw_metadata = row["metadata"]
    metadata = raw_metadata
    raw_stop = row["stop_reason"]
    stop_reason = raw_stop
    return Session(
        id=row["id"],
        agent_id=row["agent_id"],
        environment_id=row["environment_id"],
        agent_version=row["agent_version"],
        model=row.get("model"),
        title=row["title"],
        metadata=metadata,
        # ``status`` is derived ({active, idle}) from the event log via
        # ``_SESSION_STATUS_EXPR``; ``get_session``/``list_sessions`` select it
        # as a ``status`` column. There is no persisted status column. Reads
        # that don't derive it (INSERT ... RETURNING, clone) default to "idle"
        # — a fresh session is idle, and internal paths never surface it.
        status=row.get("status") or "idle",
        stop_reason=stop_reason,
        last_event_seq=row["last_event_seq"],
        usage=SessionUsage(
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            cache_read_input_tokens=row["cache_read_input_tokens"],
            cache_creation_input_tokens=row["cache_creation_input_tokens"],
        ),
        created_by=actor_from_row(row),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        archived_at=row["archived_at"],
        focal_channel=row["focal_channel"],
        focal_locked=row["focal_locked"],
        # Soft reads: present in every ``SELECT *`` / ``RETURNING *`` feeder
        # (the worker step's ``get_session_bare`` is one), default for the few
        # explicit-column reads that don't select them.
        origin=row.get("origin") or "foreground",
        parent_run_id=row.get("parent_run_id"),
        surface_frozen=bool(row.get("surface_frozen")),
        archive_when_idle=bool(row.get("archive_when_idle")),
        # Soft read: present in every ``SELECT *`` / ``RETURNING *`` feeder;
        # the few explicit-column reads that don't select it fall back to the
        # safe default "off" (#710).
        outbound_suppression=row.get("outbound_suppression") or "off",
        # Present only when the query derives it (list_sessions); other callers
        # (single read, INSERT ... RETURNING) leave it None.
        last_event_at=row.get("last_event_at"),
    )


def _default_workspace_path(account_id: str, session_id: str) -> str:
    """Per-tenant default workspace dir ``workspace_root/{account_id}/{session_id}``.

    The #367 isolation convention (a stray bind-mount can't reach across tenants;
    per-tenant quotas/backups scope to one dir) — one source of truth for the
    session-insert paths.
    """
    from aios.config import get_settings

    return str(get_settings().workspace_root / account_id / session_id)


async def insert_session(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    agent_id: str,
    environment_id: str,
    agent_version: int | None,
    title: str | None,
    metadata: dict[str, Any],
    workspace_path: str | None = None,
    env: dict[str, str] | None = None,
    focal_channel: str | None = None,
    focal_locked: bool = False,
    archive_when_idle: bool = False,
    outbound_suppression: str = "off",
    frozen_surface: Surface | None = None,
    frozen_litellm_extra: dict[str, Any] | None = None,
) -> Session:
    """Insert a fresh session row.

    ``workspace_path`` defaults to ``settings.workspace_root / session_id``.
    Caller sets up vault bindings via :func:`set_session_vaults` after.
    Raises :class:`NotFoundError` if either the agent or environment FK
    is unsatisfied.

    ``focal_channel`` + ``focal_locked`` are written atomically with
    the row insert so the focal-locked invariant (see
    ``switch_channel``'s rejection of mutations on per_chat sessions)
    holds from creation. Per-chat-spawned sessions pass
    ``focal_locked=True`` to start life locked on the spawning
    chat's channel.
    """
    new_id = make_id(SESSION)
    if workspace_path is None:
        workspace_path = _default_workspace_path(account_id, new_id)
    try:
        row = await conn.fetchrow(
            """
            INSERT INTO sessions (
                id, agent_id, environment_id, agent_version, title, metadata,
                workspace_volume_path, env,
                focal_channel, focal_locked, account_id, archive_when_idle,
                outbound_suppression, created_by_type, created_by_ref,
                tools, mcp_servers, http_servers, surface_frozen, litellm_extra
            )
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8::jsonb, $9, $10, $11, $12, $13, $14, $15,
                    $16::jsonb, $17::jsonb, $18::jsonb, $19, $20::jsonb)
            RETURNING *
            """,
            new_id,
            agent_id,
            environment_id,
            agent_version,
            title,
            json.dumps(metadata),
            workspace_path,
            json.dumps(env or {}),
            focal_channel,
            focal_locked,
            account_id,
            archive_when_idle,
            outbound_suppression,
            *actor_columns(),
            json.dumps([item.model_dump() for item in frozen_surface.tools])
            if frozen_surface
            else None,
            json.dumps([item.model_dump() for item in frozen_surface.mcp_servers])
            if frozen_surface
            else None,
            json.dumps([item.model_dump() for item in frozen_surface.http_servers])
            if frozen_surface
            else None,
            frozen_surface is not None,
            json.dumps(frozen_litellm_extra or {}) if frozen_surface else None,
        )
    except asyncpg.ForeignKeyViolationError as exc:
        raise NotFoundError(
            "agent or environment not found",
            detail={
                "agent_id": agent_id,
                "environment_id": environment_id,
            },
        ) from exc
    assert row is not None
    return _row_to_session(row)


async def insert_child_session(
    conn: asyncpg.Connection[Any],
    *,
    session_id: str,
    account_id: str,
    agent_id: str | None,
    environment_id: str,
    agent_version: int | None,
    model: str | None,
    parent_run_id: str | None,
    tools: list[ToolSpec],
    mcp_servers: list[McpServerSpec],
    http_servers: list[HttpServerSpec],
    litellm_extra: dict[str, Any] | None = None,
    workspace_path: str | None = None,
) -> Session | None:
    """Insert a workflow ``agent()`` child under a deterministic ``session_id``.

    ``INSERT ... ON CONFLICT (id) DO NOTHING RETURNING *`` — returns the new
    :class:`Session` on first spawn, or ``None`` on conflict (a replay found the
    existing row, so the caller harvests instead of re-spawning). ``origin`` is
    ``'background'`` and ``agent_version`` is the **pinned** int resolved at
    spawn (never ``None`` — a child must not track "latest").

    ``tools``/``mcp_servers``/``http_servers`` are the child's **frozen, run-attenuated
    surface** (the ``attenuate(agent, run)`` meet result) and ``surface_frozen`` is set
    ``TRUE`` — read back by ``load_for_session`` instead of the live agent, and pinned
    under ``ON CONFLICT DO NOTHING`` so a replay can never re-freeze a shifted surface.
    ``litellm_extra`` is the child's **frozen, clamped model identity** (#823 — the
    second authority axis: ``api_base`` foremost), likewise read back by
    ``load_for_session`` and pinned so a later ``update_agent`` can't shift the child's
    inference endpoint on replay.
    The caller delivers the agent input and copies the run's vaults in the same
    transaction.
    """
    workspace_path = workspace_path or _default_workspace_path(account_id, session_id)
    try:
        row = await conn.fetchrow(
            """
            INSERT INTO sessions (
                id, agent_id, environment_id, agent_version, model, title, metadata,
                workspace_volume_path, env, focal_channel, focal_locked,
                account_id, parent_run_id, origin, archive_when_idle,
                tools, mcp_servers, http_servers, surface_frozen, litellm_extra,
                tools_vocab_epoch
            )
            VALUES ($1, $2, $3, $4, $5, NULL, '{}'::jsonb, $6, '{}'::jsonb,
                    NULL, FALSE, $7, $8, 'background', TRUE,
                    $9::jsonb, $10::jsonb, $11::jsonb, TRUE, $12::jsonb, $13)
            ON CONFLICT (id) DO NOTHING
            RETURNING *
            """,
            session_id,
            agent_id,
            environment_id,
            agent_version,
            model,
            workspace_path,
            account_id,
            parent_run_id,
            json.dumps([t.model_dump() for t in tools]),
            json.dumps([s.model_dump() for s in mcp_servers]),
            json.dumps([s.model_dump() for s in http_servers]),
            json.dumps(litellm_extra or {}),
            TOOLS_VOCAB_EPOCH,
        )
    except asyncpg.ForeignKeyViolationError as exc:
        raise NotFoundError(
            "agent or environment not found",
            detail={"agent_id": agent_id, "environment_id": environment_id},
        ) from exc
    return _row_to_session(row) if row is not None else None


async def get_session_env(
    conn: asyncpg.Connection[Any], session_id: str, *, account_id: str
) -> dict[str, str]:
    row = await conn.fetchrow(
        "SELECT env FROM sessions WHERE id = $1 AND account_id = $2",
        session_id,
        account_id,
    )
    if row is None:
        raise NotFoundError("session not found", detail={"session_id": session_id})
    return dict(row["env"] or {})


async def freeze_session_surface(
    conn: asyncpg.Connection[Any],
    session_id: str,
    surface: Surface,
    *,
    account_id: str,
    litellm_extra: dict[str, Any] | None = None,
) -> None:
    """Pin a newly spawned session's authority surface and model identity."""
    result = await conn.execute(
        "UPDATE sessions SET tools = $3::jsonb, mcp_servers = $4::jsonb, "
        "http_servers = $5::jsonb, litellm_extra = $6::jsonb, surface_frozen = TRUE "
        "WHERE id = $1 AND account_id = $2",
        session_id,
        account_id,
        json.dumps([item.model_dump() for item in surface.tools]),
        json.dumps([item.model_dump() for item in surface.mcp_servers]),
        json.dumps([item.model_dump() for item in surface.http_servers]),
        json.dumps(litellm_extra or {}),
    )
    if result == "UPDATE 0":
        raise NotFoundError("session not found", detail={"session_id": session_id})


async def copy_session_github_resources(
    conn: asyncpg.Connection[Any],
    parent_session_id: str,
    child_session_id: str,
    bindings: list[tuple[str, str]],
    *,
    account_id: str,
) -> None:
    """Copy selected parent repository bindings without accepting new credentials."""
    if not bindings:
        return
    rows = await conn.fetch(
        "SELECT * FROM session_github_repositories "
        "WHERE session_id = $1 AND account_id = $2 ORDER BY rank",
        parent_session_id,
        account_id,
    )
    wanted = set(bindings)
    rank = 0
    for row in rows:
        if (row["repo_url"], row["mount_path"]) not in wanted:
            continue
        await conn.execute(
            """
            INSERT INTO session_github_repositories
                (id, session_id, rank, repo_url, mount_path, ciphertext, nonce,
                 git_user_name, git_user_email, account_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """,
            make_id(GITHUB_REPOSITORY),
            child_session_id,
            rank,
            row["repo_url"],
            row["mount_path"],
            row["ciphertext"],
            row["nonce"],
            row["git_user_name"],
            row["git_user_email"],
            account_id,
        )
        rank += 1


async def copy_session_resources(
    conn: asyncpg.Connection[Any],
    parent_session_id: str,
    child_session_id: str,
    *,
    account_id: str,
) -> None:
    """Copy parent resource bindings, including encrypted repository credentials."""
    await conn.execute(
        """
        INSERT INTO session_memory_stores
            (session_id, memory_store_id, rank, access, instructions,
             name_at_attach, description_at_attach, account_id)
        SELECT $2, memory_store_id, rank, access, instructions,
               name_at_attach, description_at_attach, account_id
          FROM session_memory_stores
         WHERE session_id = $1 AND account_id = $3
        """,
        parent_session_id,
        child_session_id,
        account_id,
    )
    rows = await conn.fetch(
        "SELECT * FROM session_github_repositories "
        "WHERE session_id = $1 AND account_id = $2 ORDER BY rank",
        parent_session_id,
        account_id,
    )
    for row in rows:
        await conn.execute(
            """
            INSERT INTO session_github_repositories
                (id, session_id, rank, repo_url, mount_path, ciphertext, nonce,
                 git_user_name, git_user_email, account_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """,
            make_id(GITHUB_REPOSITORY),
            child_session_id,
            row["rank"],
            row["repo_url"],
            row["mount_path"],
            row["ciphertext"],
            row["nonce"],
            row["git_user_name"],
            row["git_user_email"],
            account_id,
        )


async def get_session_frozen_surface(
    conn: asyncpg.Connection[Any], session_id: str, *, account_id: str
) -> Surface | None:
    """A workflow child's frozen surface, or ``None`` if the row is not ``surface_frozen``.

    ``None`` is the load-bearing signal that this is *not* a frozen child (a foreground
    session, or — for a ``parent_run_id`` child — a corrupt/unmigrated row the caller
    must fail closed on). Raises ``NotFoundError`` if the session is absent.
    """
    row = await conn.fetchrow(
        "SELECT tools, mcp_servers, http_servers, surface_frozen "
        "FROM sessions WHERE id = $1 AND account_id = $2",
        session_id,
        account_id,
    )
    if row is None:
        raise NotFoundError("session not found", detail={"session_id": session_id})
    if not row["surface_frozen"]:
        return None
    return Surface(
        tools=load_tool_specs(row["tools"]),
        mcp_servers=[McpServerSpec.model_validate(s) for s in row["mcp_servers"]],
        http_servers=[HttpServerSpec.model_validate(s) for s in row["http_servers"]],
    )


async def get_session_frozen_litellm_extra(
    conn: asyncpg.Connection[Any], session_id: str, *, account_id: str
) -> dict[str, Any]:
    """A workflow child's frozen model identity (``litellm_extra``), or ``{}`` (#823).

    The model-identity dual of :func:`get_session_frozen_surface`: a named-agent
    child's ``litellm_extra`` (``api_base`` foremost) clamped + frozen at spawn, read
    back verbatim every wake so replay never re-resolves a since-changed agent's
    endpoint. ``{}`` covers an agentless generic child, a child whose agent carried no
    ``litellm_extra``, and a grandfathered pre-column child (``NULL`` only when its
    ``agent_versions`` row was missing) — all "no redirect, default endpoint", the safe
    reading. Raises ``NotFoundError`` if the session is absent.
    """
    row = await conn.fetchrow(
        "SELECT litellm_extra FROM sessions WHERE id = $1 AND account_id = $2",
        session_id,
        account_id,
    )
    if row is None:
        raise NotFoundError("session not found", detail={"session_id": session_id})
    raw = row["litellm_extra"]
    return raw if raw is not None else {}


async def get_session_workflow_context(
    conn: asyncpg.Connection[Any], session_id: str
) -> tuple[str, str | None] | None:
    """``(account_id, parent_run_id)`` for a session, or ``None`` if it's absent.

    Unscoped (internal/trusted — the ``return``/``error`` completion tools run
    inside the child's own dispatch and need the parent run to wake). A non-null
    ``parent_run_id`` is what marks the session a workflow child.
    """
    row = await conn.fetchrow(
        "SELECT account_id, parent_run_id FROM sessions WHERE id = $1", session_id
    )
    return (row["account_id"], row["parent_run_id"]) if row is not None else None


async def get_wake_priority_context(
    conn: asyncpg.Connection[Any], session_id: str
) -> tuple[str, bool] | None:
    """``(account_id, is_background)`` for a session's wake priority, or ``None``.

    The single source of truth for :func:`aios.services.wake.defer_wake`'s
    foreground/background demotion. Re-keys the demotion off the **triggering
    edge's up-link** (#1123's ``request_opened`` ``caller``) rather than the
    run-only ``parent_run_id`` column, so every caller kind (api/session/run)
    demotes uniformly when its ancestor is background:

    - ``caller.kind = 'run'`` → background (a workflow run is always background;
      this preserves the legacy ``parent_run_id`` run path behavior-for-behavior).
    - ``caller.kind = 'session'`` → background iff the caller session is itself
      background-rooted (``origin = 'background'``); a foreground/user-rooted
      session-invoke stays foreground.
    - ``caller.kind = 'api'`` → foreground.
    - no ``request_opened`` edge (an ordinary root / fg-user session) → foreground.

    Derives the demotion from the **latest still-open** edge — the most-recent
    ``request_opened`` not yet answered by a ``request_response``, composed from
    the shared :func:`open_request_anti_join` fragment with ``awaited_only=False``
    (**intentional here**, unlike :func:`get_open_request_ids`). The demotion keys
    off *any* triggering up-link, including a background fan-out's **unawaited**
    ``Tell(NewSession)`` child: a ``Tell``-spawned child of a background run is a
    fan-out descendant and *should* wake at background, so this open set is
    deliberately a **superset** of :func:`get_open_request_ids`'s (which excludes
    the unawaited ``Tell`` edges) — it is every still-open edge, awaited or not,
    not the response-owing subset. A session
    that has served several requests wakes at the priority of the edge that
    triggered *this* wake, not its oldest-ever edge — reachable since ``invoke``
    with ``target_kind='session'`` (#1128) appends a second edge to a live session:
    keying on the oldest would pin a re-invoked background child at background while
    it serves a foreground api request (and a fg-first session at foreground while a
    background descendant drives it). An answered-and-gone request contributes
    nothing, so a quiescent session falls back to the foreground default. A ``None``
    result is the deleted-session race — :func:`defer_wake` maps it to the
    foreground default, then the wake no-ops harmlessly. Unscoped (internal/trusted),
    mirroring :func:`get_session_workflow_context`.
    """
    row = await conn.fetchrow(
        "SELECT s.account_id, "
        "  COALESCE(("
        "    SELECT CASE req.data->'caller'->>'kind' "
        "      WHEN 'run' THEN TRUE "
        "      WHEN 'session' THEN ("
        "        SELECT caller_s.origin = 'background' "
        "        FROM sessions caller_s "
        "        WHERE caller_s.id = req.data->'caller'->>'id') "
        "      ELSE FALSE "
        "    END "
        "    FROM events req "
        "    WHERE "
        + open_request_anti_join(sid="s.id", acct="s.account_id", awaited_only=False)
        + "    ORDER BY req.seq DESC LIMIT 1"
        "  ), FALSE) AS is_background "
        "FROM sessions s WHERE s.id = $1",
        session_id,
    )
    return (row["account_id"], bool(row["is_background"])) if row is not None else None


async def read_request_response(
    conn: asyncpg.Connection[Any], session_id: str, *, account_id: str, request_id: str
) -> Outcome | None:
    """A specific request's written **response** (`request_response` event), or
    ``None`` if none has been written.

    The response-only primitive: it reflects only an explicit `return`/`error`,
    harness-erroring, or no_return write — NOT liveness. It backs
    :func:`write_response_if_absent`'s exactly-once absent-recheck (its only caller);
    the run-step harvest instead uses :func:`derive_response`, which additionally
    folds in `child_gone` liveness. Keeping this response-only is load-bearing — a
    liveness check here would make the recheck treat a never-answered gone session
    as already-answered and suppress the totality write. ``data`` is
    ``{event, request_id, is_error, result, error}``. Exactly one exists per request
    (see :func:`write_response_if_absent`), so ``LIMIT 1`` returns it directly; the
    ``events_request_response_idx`` partial index (mig 0069) keeps this a point
    lookup rather than a per-wake history scan.
    """
    row = await conn.fetchrow(
        "SELECT data FROM events WHERE session_id = $1 AND account_id = $2 "
        "AND kind = 'lifecycle' AND data->>'event' = 'request_response' "
        "AND data->>'request_id' = $3 ORDER BY seq DESC LIMIT 1",
        session_id,
        account_id,
        request_id,
    )
    return queries.outcome_from_jsonb(row["data"]) if row is not None else None


async def get_open_request_ids(
    conn: asyncpg.Connection[Any], session_id: str, *, account_id: str
) -> list[str]:
    """The ``request_id``s of the session's still-open requests, oldest first.

    A request is a trusted ``request_opened`` lifecycle event (the *ask* half of
    the request edge, #1123 — appended only by the launch-path creation sites; see
    :func:`append_request_opened`); it is *open* until a ``request_response`` event
    (the symmetric *answer* half) answers it. The open set is
    ``asked(request_opened) MINUS answered(request_response)``, so an ordinary
    session (no edge) returns ``[]`` and the quiescence guard no-ops for it. v1
    injects one request per child, so this is ``[]`` or one id; the set shape is
    ready for multi-request `invoke_session`.

    Only **awaited** edges (``awaited=true``, the ``Ask`` arm) count: a
    ``Tell(NewSession)`` fire-and-forget spawn writes a real ``request_opened``
    row with ``awaited=false`` (#1197) — it carries lineage/depth/surface but owes
    **no** response, so the asked set filters it out. An absent ``awaited`` reads
    as ``true`` (additive/legacy), so pre-#1197 rows keep their obligation. This
    is one of the three awaited-triad readers (with the #1131 totality gate and
    the #1132 cap) that must apply the same filter in lockstep.

    Reads the trusted ``request_opened`` frame rather than the forgeable
    ``metadata.request`` user-message blob (which #1123 still dual-writes for the
    legacy run path until a future contract migration retires it). Both halves
    are partial-indexed (``events_request_opened_idx`` /
    ``events_request_response_idx``) so this stays
    a point lookup rather than a per-wake history scan.

    #1747: ``floor_bounded=True`` — the anti-join scan is restricted to
    ``req.seq >= sessions.open_request_scan_floor``, bounding the per-step cost
    to O(open edges + request-edges-since-floor) instead of O(lifetime asks
    served). See :func:`advance_open_request_scan_floor` for the invariant that
    makes this safe (the floor never advances past a genuinely-open edge) and
    the SEQ-PREFIX INVARIANT it depends on.
    """
    rows: list[asyncpg.Record] = await conn.fetch(
        "SELECT req.data->>'request_id' AS rid FROM events req WHERE "
        + open_request_anti_join(sid="$1", acct="$2", awaited_only=True, floor_bounded=True)
        + "ORDER BY req.seq ASC",
        session_id,
        account_id,
    )
    return [r["rid"] for r in rows]


async def get_open_obligations(
    conn: asyncpg.Connection[Any], session_id: str, *, account_id: str
) -> list[Obligation]:
    """The session's still-open **awaited** obligations, oldest first (#1413).

    Modeled 1:1 on :func:`get_open_request_ids` — same
    ``asked(request_opened) MINUS answered(request_response)`` anti-join (the
    shared :func:`open_request_anti_join` fragment, ``awaited_only=True`` so a
    fire-and-forget ``Tell`` edge is excluded), same ``ORDER BY req.seq ASC``
    (oldest-first, deterministic). But
    instead of bare ``request_id``s it projects a full :class:`Obligation` per open
    edge so the tail-injected obligations block (and the ``obligations`` read
    model) can render it: ``caller_kind`` (``req.data->'caller'->>'kind'`` — the
    **trusted** frame, not the forgeable ``metadata.request`` blob), ``opened_at``
    (``req.created_at``, for age), and a short ``summary`` (``req.data->>'summary'``,
    additive — absent on pre-#1413 frames -> ``None`` -> an id-only render line, no
    migration).

    #1522 widens the projection to also carry ``output_schema``
    (``req.data->'output_schema'``) — the JSON Schema the request demands of its
    response ``value`` (the **acceptance contract**), the SAME datum
    :func:`get_request_output_schema` reads off the trusted ``request_opened``
    frame. It is already persisted on the frame, so this is a SELECT widening, not
    new persistence: additive, ``None`` when the request demands no schema or on a
    pre-#1522 frame — no migration. This makes one shared owed-read-model carry
    "outstanding obligations + each one's contract" so every consumer (the
    obligations tail, the quiescence-attempt surfacing, the ``list_obligations``
    tool) can render "what you owe **and the format**" from one source.

    This is the data source for the always-on obligations reminder that survives
    context-windowing erasure of the original request user message (the defect
    #1413 fixes): a full-log query, not a slate-derived render-time marker.

    NOTE: the cheap quiescence GUARD decision ("do I owe anything at all?") stays
    on :func:`get_open_request_ids` (bare ids) — it must NOT pay for the schema
    projection. Only the **content render** uses this widened model.

    #1747: ``floor_bounded=True`` — same seq-range restriction as
    :func:`get_open_request_ids`; see its docstring and
    :func:`advance_open_request_scan_floor` for the invariant.
    """
    rows: list[asyncpg.Record] = await conn.fetch(
        "SELECT req.data->>'request_id' AS rid, "
        "req.data->'caller'->>'kind' AS caller_kind, "
        "req.data->'caller'->>'id' AS caller_id, "
        "req.created_at AS opened_at, "
        "req.data->>'summary' AS summary, "
        "req.data->'output_schema' AS output_schema "
        "FROM events req WHERE "
        + open_request_anti_join(sid="$1", acct="$2", awaited_only=True, floor_bounded=True)
        + "ORDER BY req.seq ASC",
        session_id,
        account_id,
    )
    return [
        Obligation(
            request_id=r["rid"],
            caller_kind=r["caller_kind"] or "",
            caller_id=r["caller_id"],
            opened_at=r["opened_at"],
            summary=r["summary"],
            # ``output_schema`` is a JSONB column (``data->'output_schema'``) —
            # the jsonb codec decodes it the same way get_request_output_schema does;
            # absent on a no-schema/pre-#1522 frame -> None (additive, no migration).
            output_schema=r["output_schema"] if r["output_schema"] is not None else None,
        )
        for r in rows
    ]


async def advance_open_request_scan_floor(
    conn: asyncpg.Connection[Any], session_id: str, *, account_id: str
) -> None:
    """Ratchet ``sessions.open_request_scan_floor`` forward (#1747) — the
    monotone scan-floor that bounds the per-step open-obligations anti-join
    (:func:`get_open_request_ids` / :func:`get_open_obligations`,
    ``floor_bounded=True``) to O(open edges + request-edges-since-floor)
    instead of O(lifetime asks served).

    ONE data-modifying-CTE statement — single snapshot, race-safe by the
    SEQ-PREFIX INVARIANT (documented on :func:`aios.db.queries.open_request_anti_join`
    and at ``append_event``: within a session, commit order == seq order and
    events are never deleted/re-sequenced/truncated, so any statement
    snapshot sees a seq-prefix of the session's history).

    ``open_edges`` reuses :func:`~aios.db.queries.open_request_anti_join`
    VERBATIM (``awaited_only=True, floor_bounded=True``) — no hand-copied
    predicate, so this can never independently drift from what the readers
    treat as "open". The new floor is:

      * if any awaited open edge exists at/after the current floor: the seq
        of the OLDEST such edge (``min``, not ``min + 1`` — the oldest open
        edge sits AT the new floor, and readers use an inclusive
        ``req.seq >= floor`` bound, so it must stay visible);
      * else (every request edge at/after the floor is answered or
        unawaited, i.e. nothing is open): one past the newest **request**
        edge in the whole session (``max(seq) + 1`` over ALL
        ``request_opened`` rows, open+answered+unawaited) — an O(1)
        index-max on the partial ``request_opened`` index. Deliberately
        NOT ``sessions.last_event_seq``: that column advances on every
        ordinary event, which would force a floor **write** (the
        monotone-``<`` guard only skips a no-op *value*, not a no-op
        *statement*) on every step of every session, including the ~99% of
        sessions that never had a perf problem (write amplification on the
        common path). Request-edge-max is NULL — and so triggers no write
        at all — for a session with zero request edges, and is otherwise
        stable between asks.

    Safety of the empty branch under the SEQ-PREFIX INVARIANT: ``open_edges``
    empty (with the floor bound already applied) means no open awaited edge
    has ``seq >= current floor``; since the floor never exceeds the oldest
    open edge (this function's own invariant, inductively), that means NO
    awaited edge is open at all. So every request edge with ``seq <= max`` is
    either answered or unawaited, and ``max + 1`` skips only settled edges;
    any FUTURE awaited edge necessarily has ``seq > max``, i.e.
    ``seq >= new_floor``, so the readers' inclusive bound keeps it visible
    the moment it is appended.

    The ``WHERE ... AND open_request_scan_floor < (SELECT v FROM f)`` monotone
    guard is what makes concurrent advances safe: two racing advances (e.g.
    an #1747 step-path call racing a reconciliation-sweep read, or two
    overlapping steps for the same re-entrant session) each recompute the same
    deterministic value from their own snapshot, and only a strictly-forward
    move commits — never a regression, and a stale loser is a no-op, not an
    error.

    Best-effort by contract: the ONLY use of this value is a perf bound, so a
    lost/rolled-back advance leaves the floor exactly where it was — never
    invisible-obligation risk. Callers on the hot step path MUST wrap this in
    a SAVEPOINT and swallow+count any exception (a raw swallow would leave a
    failed UPDATE's implicit abort poisoning the rest of the step's
    transaction in PG) — see ``compute_step_prelude`` in
    ``harness/step_context.py`` for the call site and its counted-swallow
    contract.
    """
    await conn.execute(
        "WITH open_edges AS ("
        "    SELECT req.seq FROM events req WHERE "
        + open_request_anti_join(sid="$1", acct="$2", awaited_only=True, floor_bounded=True)
        + "), f AS ("
        "    SELECT COALESCE("
        "        (SELECT min(seq) FROM open_edges),"
        "        (SELECT max(seq) + 1 FROM events e"
        "           WHERE e.session_id = $1 AND e.account_id = $2"
        "             AND e.kind = 'lifecycle' AND e.data->>'event' = 'request_opened')"
        "    ) AS v"
        ")"
        "UPDATE sessions"
        "   SET open_request_scan_floor = (SELECT v FROM f)"
        " WHERE id = $1 AND account_id = $2"
        "   AND (SELECT v FROM f) IS NOT NULL"
        "   AND open_request_scan_floor < (SELECT v FROM f)",
        session_id,
        account_id,
    )


async def get_open_obligations_batch(
    conn: asyncpg.Connection[Any], session_ids: list[str], *, account_id: str
) -> dict[str, list[Obligation]]:
    """Batched dual of :func:`get_open_obligations` (#1561).

    One indexed anti-join across the whole batch (``req.session_id = ANY($1)``)
    instead of one round-trip per session, then in-memory grouping by
    ``session_id``. Same ``asked(request_opened) MINUS answered(request_response)``
    open set (the shared :func:`open_request_anti_join` fragment,
    ``awaited_only=True``) and same per-row :class:`Obligation` projection as the
    single-session reader, oldest-first within each session (``ORDER BY
    req.session_id, req.seq ASC``). Used by
    :func:`services.sessions.compute_obligations` so the list-sessions /
    get-session obligations view pays exactly one DB round-trip for a batch of N
    sessions.

    #1747: deliberately stays **unbounded** (``floor_bounded`` is NOT set) — its
    ``sid`` is the batch array form ``ANY($1::text[])``, which is structurally
    incompatible with the floor's scalar-subquery bound (see
    :func:`~aios.db.queries.open_request_anti_join`'s guard, which raises if this
    is ever flipped without also rewriting the bound as a LATERAL join). The
    list-sessions accounting surface staying unbounded-by-design is intentional:
    under any (guarded-against) floor bug, this reader is the independent
    off-substrate ground truth the reconciliation sweep diffs against, and it
    is not on the hot per-step path this issue targets (API-side, one
    round-trip per batch — #1561 out of scope). A future LATERAL rewrite to
    bound this path is a possible follow-up, not part of this change.
    """
    if not session_ids:
        return {}
    rows: list[asyncpg.Record] = await conn.fetch(
        "SELECT req.session_id AS sid, "
        "req.data->>'request_id' AS rid, "
        "req.data->'caller'->>'kind' AS caller_kind, "
        "req.data->'caller'->>'id' AS caller_id, "
        "req.created_at AS opened_at, "
        "req.data->>'summary' AS summary, "
        "req.data->'output_schema' AS output_schema "
        "FROM events req WHERE "
        + open_request_anti_join(sid="ANY($1::text[])", acct="$2", awaited_only=True)
        + "ORDER BY req.session_id, req.seq ASC",
        session_ids,
        account_id,
    )
    out: dict[str, list[Obligation]] = {}
    for r in rows:
        out.setdefault(r["sid"], []).append(
            Obligation(
                request_id=r["rid"],
                caller_kind=r["caller_kind"] or "",
                caller_id=r["caller_id"],
                opened_at=r["opened_at"],
                summary=r["summary"],
                output_schema=r["output_schema"] if r["output_schema"] is not None else None,
            )
        )
    return out


async def get_request_caller(
    conn: asyncpg.Connection[Any], session_id: str, *, request_id: str
) -> dict[str, Any] | None:
    """The trusted ``caller`` of a request — ``{kind, id}`` — off its ``request_opened``
    edge, or ``None`` if no such open-edge exists.

    Reads the #1123 ``request_opened`` lifecycle frame (the trusted half), NEVER the
    forgeable ``metadata.request`` user-message blob, so the caller provenance can be
    trusted to route a response wake (#1127): ``kind == "run"`` → the run harvest path
    (fused ``child_done`` marker); ``kind == "session"`` → wake the caller session;
    ``kind == "api"`` → the ephemeral HTTP awaiter (no wake — it long-polls). Like
    :func:`get_request_output_schema`, ``session_id`` is a unique PK so it is sufficient
    scope on its own (the caller already account-scoped the session).
    """
    caller = await conn.fetchval(
        "SELECT req.data->'caller' FROM events req "
        "WHERE req.session_id = $1 "
        "AND req.kind = 'lifecycle' AND req.data->>'event' = 'request_opened' "
        "AND req.data->>'request_id' = $2 "
        "ORDER BY req.seq ASC LIMIT 1",
        session_id,
        request_id,
    )
    return caller if caller is not None else None


async def get_request_output_schema(
    conn: asyncpg.Connection[Any], session_id: str, *, request_id: str
) -> dict[str, Any] | None:
    """The JSON Schema a request demands of its response ``value``, or ``None``.

    Reads ``output_schema`` off the trusted ``request_opened`` edge (#1131 — no longer
    the forgeable ``metadata.request`` user-message blob). Keyed on
    ``(session_id, request_id)`` — a child can owe several requests, each with its own
    schema — so the ``return`` enforcement validates a ``value`` against *its* request's
    schema. Like :func:`get_session_workflow_context` (the other return-path read),
    ``session_id`` is a unique PK and so is sufficient scope on its own — no
    ``account_id`` predicate, which lets the tool path resolve the schema without first
    looking up the session's account. ``None`` when the request demands no schema (the
    common case) or the id matches nothing.
    """
    schema = await conn.fetchval(
        "SELECT req.data->'output_schema' FROM events req "
        "WHERE req.session_id = $1 "
        "AND req.kind = 'lifecycle' AND req.data->>'event' = 'request_opened' "
        "AND req.data->>'request_id' = $2 "
        "ORDER BY req.seq ASC LIMIT 1",  # oldest-first, like get_open_request_ids — deterministic
        session_id,
        request_id,
    )
    return schema if schema is not None else None


async def get_closed_request(
    conn: asyncpg.Connection[Any], session_id: str, *, request_id: str
) -> tuple[Outcome, Any] | None:
    """The terminal ``(outcome, closed_at)`` of a request that has ALREADY been
    answered, or ``None`` if it's still open (or ``request_id`` names no request
    at all — the caller's own ``unknown_request`` path handles that).

    The #1773 liveness-first primitive: ``return``/``error`` must check this
    BEFORE doing any other work (schema validation included) so a child blindly
    answering a request that closed out from under it (e.g. the run's await
    deadline wrote a timeout response first) gets one clear terminal stop signal
    instead of an endless schema-bounce loop that never mentions the request is
    dead. Reads the same ``request_response`` lifecycle event
    :func:`read_request_response` does, plus its ``created_at`` (the closing
    timestamp the stop message quotes). Like :func:`get_request_output_schema`,
    ``session_id`` is a unique PK and so is sufficient scope on its own — no
    ``account_id`` predicate, so the tool path resolves this without first
    looking up the session's account.
    """
    row = await conn.fetchrow(
        "SELECT data, created_at FROM events WHERE session_id = $1 "
        "AND kind = 'lifecycle' AND data->>'event' = 'request_response' "
        "AND data->>'request_id' = $2 ORDER BY seq DESC LIMIT 1",
        session_id,
        request_id,
    )
    if row is None:
        return None
    return queries.outcome_from_jsonb(row["data"]), row["created_at"]


async def count_request_nudges(
    conn: asyncpg.Connection[Any], session_id: str, *, account_id: str, request_id: str
) -> int:
    """How many times this request has been nudged **since the most recent
    activity turn** — the count of nudge user messages whose
    ``metadata.nudged_request_ids`` include it and whose ``seq`` is greater than
    the latest assistant tool-call turn in this session.

    This is a **consecutive-inaction** count, not a lifetime one (#1412): the
    request-totality quiescence guard short-circuits on any assistant turn that
    carries ``tool_calls`` (``services/sessions.py``), so that tool-call turn is
    the natural reset anchor. Counting nudges only *after* it turns the retry
    budget from a **loop-limiter** (bounds total idle work → fights always-on)
    into a **stuck-detector** (bounds *consecutive* idle turns → enables
    always-on): an agent that interleaves tool calls never trips the budget,
    while one stuck doing nothing N turns running still gets ``no_return``'d.

    The reset anchor is **session-wide** (the latest tool-call turn), but the
    nudge count is **per-request** (filtered by ``nudged_request_ids``): a
    session working one obligation does not reset a stuck sibling's count —
    each open request keeps its own consecutive count, so a stuck sibling still
    hits ``no_return`` (#1412, multi-request reset = per-request).

    The budget stays *derived* from the log, not stored — so it is crash-safe
    and needs no counter to keep in sync (the same stance as
    ``_count_consecutive_rescheduling`` for model-error backoff). The
    ``data ? 'tool_calls'`` / ``role = 'assistant'`` anchor subquery matches
    ``events_assistant_tool_calls_idx`` (migration 0011/0023); the extra
    ``jsonb_array_length > 0`` guard excludes a present-but-empty array so the
    anchor agrees byte-for-byte with the guard's ``assistant_msg.get('tool_calls')``
    truthiness short-circuit.

    The anchor is additionally floored at this request's ``request_opened`` seq:
    a nudge for ``request_id`` is only ever appended while it is open (the nudge
    append follows ``get_open_request_ids``), so every counted row already has
    ``seq >`` that request's opening seq by construction — the floor is a
    semantic no-op on all naturally-occurring event orders. It matters only when
    no tool-call turn exists (a tool-call-free session), where the anchor would
    otherwise be 0 and the count would scan every user message in the session's
    history instead of just those since the request opened.
    """
    n: int | None = await conn.fetchval(
        "SELECT count(*) FROM events WHERE session_id = $1 AND account_id = $2 "
        "AND kind = 'message' AND role = 'user' "
        "AND data->'metadata'->'nudged_request_ids' @> to_jsonb($3::text) "
        "AND seq > GREATEST("
        "    COALESCE("
        "        (SELECT max(seq) FROM events "
        "         WHERE session_id = $1 AND account_id = $2 "
        "         AND kind = 'message' AND role = 'assistant' "
        "         AND data ? 'tool_calls' "
        "         AND jsonb_array_length(data->'tool_calls') > 0), 0),"
        "    COALESCE("
        "        (SELECT req.seq FROM events req "
        "         WHERE req.session_id = $1 AND req.account_id = $2 "
        "         AND req.kind = 'lifecycle' AND req.data->>'event' = 'request_opened' "
        "         AND req.data->>'request_id' = $3 "
        "         ORDER BY req.seq ASC LIMIT 1), 0)"
        ")",
        session_id,
        account_id,
        request_id,
    )
    return n or 0


async def derive_response(
    conn: asyncpg.Connection[Any], session_id: str, *, account_id: str, request_id: str
) -> Outcome | None:
    """A request's **terminal outcome**, or ``None`` if it is still pending.

    The single resolver the run harvest asks "what became of this ``agent()``?" — a
    pure, monotonic function of the target's state (the dual of Block-0's
    ``_SESSION_STATUS_EXPR``):

    * a **written response** (``return``/``error``, model-failure, or the no_return
      backstop) → that outcome;
    * else, if the target is **gone** (archived or deleted before answering, so it
      can never respond) → a Failed ``child_gone`` outcome;
    * else → ``None`` (alive and unanswered — still pending).

    The response and the liveness are read in **one query / one snapshot**, so a
    response always dominates "gone" even if the target is archived the instant
    after answering — there is no read-between-reads window. Each branch is
    terminal-or-pending and never reverts (a response is permanent; gone is
    permanent), so the harvest can journal the returned dict as the ``call_result``
    payload directly.
    """
    row = await conn.fetchrow(
        "SELECT (SELECT data FROM events e WHERE e.session_id = $1 AND e.account_id = $2 "
        "        AND e.kind = 'lifecycle' AND e.data->>'event' = 'request_response' "
        "        AND e.data->>'request_id' = $3 ORDER BY e.seq DESC LIMIT 1) AS response, "
        "       EXISTS (SELECT 1 FROM sessions s WHERE s.id = $1 AND s.account_id = $2 "
        "               AND s.archived_at IS NULL) AS live",
        session_id,
        account_id,
        request_id,
    )
    assert row is not None
    if row["response"] is not None:
        return queries.outcome_from_jsonb(row["response"])
    if not row["live"]:
        return Err(error={"kind": "child_gone"})
    return None


async def read_session_watermarks(
    conn: asyncpg.Connection[Any], session_id: str, *, account_id: str
) -> tuple[int, int] | None:
    """(last_reacted_seq, last_stimulus_seq) for a scoped session, or None if absent.

    The Mode-2 scalar read for await_session: the monotonic quiescence columns, read
    directly with no event-log status derivation."""
    row = await conn.fetchrow(
        "SELECT last_reacted_seq, last_stimulus_seq FROM sessions "
        "WHERE id = $1 AND account_id = $2",
        session_id,
        account_id,
    )
    if row is None:
        return None
    return (row["last_reacted_seq"], row["last_stimulus_seq"])


async def derive_session_status(
    conn: asyncpg.Connection[Any], session_id: str, *, account_id: str
) -> str:
    """The session's derived ``{active, idle}`` status via the single
    ``_SESSION_STATUS_EXPR`` source of truth.

    The quiescence guard calls this in the same transaction as the just-appended
    assistant event — so it sees that event — to decide whether the session would
    now go idle while still owing a request response.
    """
    status: str | None = await conn.fetchval(
        f"SELECT ({_SESSION_STATUS_EXPR}) FROM sessions WHERE id = $1 AND account_id = $2",
        session_id,
        account_id,
    )
    return status or "idle"


async def reclaim_session_if_idle(
    conn: asyncpg.Connection[Any], session_id: str, *, account_id: str
) -> bool:
    """Soft-archive the session **iff it is currently idle** — the archive-on-quiescence reclaim.

    One conditional UPDATE reusing ``_SESSION_ACTIVE_EXPR`` (the same predicate every reader and
    the sweep use): a stimulus arriving as the session idles flips the predicate to active → no
    row matches → no-op, so a late user/tool message always wins over reclaim. Idempotent via
    ``archived_at IS NULL``. Returns ``True`` iff this call archived the row.

    The caller gates on the session's immutable ``archive_when_idle`` launch flag; this query
    enforces the idle condition atomically and must be the **last** session write of the step
    (no write may follow — ``append_event`` fences on ``archived_at IS NULL``).
    """
    row = await conn.fetchrow(
        "UPDATE sessions SET archived_at = now(), updated_at = now() "
        f"WHERE id = $1 AND account_id = $2 AND archived_at IS NULL AND NOT {_SESSION_ACTIVE_EXPR} "
        "RETURNING id",
        session_id,
        account_id,
    )
    return row is not None


async def write_response_if_absent(
    conn: asyncpg.Connection[Any],
    session_id: str,
    *,
    account_id: str,
    request_id: str,
    outcome: Outcome,
) -> bool:
    """Write a request's response, **exactly once per request** (first-writer-wins).

    A session answers a request via `return`/`error`; concurrent or repeated writes
    for the same ``request_id`` (`return`+`error` in one assistant batch, a model
    double-call, a late `return` racing the model-failure path or the no_return
    backstop) must still yield **exactly one** response. ``FOR UPDATE`` on the
    session row + a per-``request_id`` absent-recheck makes the first writer win;
    the rest no-op. Returns ``True`` iff this call wrote the response.

    The guard is per ``request_id``, so a session with several open requests can
    answer each independently without clobbering the others.
    """
    async with conn.transaction():
        await conn.execute(
            "SELECT 1 FROM sessions WHERE id = $1 AND account_id = $2 FOR UPDATE",
            session_id,
            account_id,
        )
        if (
            await read_request_response(
                conn, session_id, account_id=account_id, request_id=request_id
            )
            is not None
        ):
            return False
        await queries.append_event(
            conn,
            account_id=account_id,
            session_id=session_id,
            kind="lifecycle",
            data={
                "event": "request_response",
                "request_id": request_id,
                **queries.outcome_to_jsonb(outcome),
            },
        )
    return True


async def append_request_opened(
    conn: asyncpg.Connection[Any],
    *,
    session_id: str,
    account_id: str,
    request_id: str,
    caller: dict[str, Any],
    depth: int,
    environment_id: str,
    frozen_surface: dict[str, Any],
    vault_ids: list[str],
    awaited: bool = True,
    output_schema: dict[str, Any] | None = None,
    summary: str | None = None,
    rlm_token_budget: int | None = None,
) -> None:
    """Append the trusted ``request_opened`` lifecycle event — the *ask* half of
    the request edge (#1123).

    The symmetric counterpart of :func:`write_response_if_absent` (the *answer*
    half, ``request_response``): a typed ``kind='lifecycle'`` frame in the
    append-only ``events`` log carrying ``{event:'request_opened', request_id,
    caller:{kind,id}, depth, environment_id, frozen_surface, vault_ids}``. Trust
    derives from the writer being **service-code-only**: this is called from
    exactly the launch-path creation sites (``create_run``,
    ``create_child_session`` / ``insert_child_session``, ``_open_agent_capability``)
    inside the same transaction as the servicer they open — never from
    harness/model-facing code.

    ``caller`` is ``{kind:'api'|'session'|'run', id}`` — the caller-kind-agnostic
    provenance the run-only ``parent_run_id`` gate generalizes to. ``depth`` is
    **carried, not enforced** here (enforcement is #1124); the run→child path
    sources it from the existing depth computation so #1124 inherits a correct
    value. ``frozen_surface`` / ``vault_ids`` are the #794 clamp results computed
    at the launch site.

    ``awaited`` (#1197) carries the ``Ask | Tell`` distinction on the edge: an
    ``Ask`` opens an **awaited** edge (``awaited=true``) the target must answer
    (``return``/``error``); a ``Tell(NewSession)`` fire-and-forget spawn opens an
    **unawaited** edge (``awaited=false``) — a real ``request_opened`` row (so it
    still carries lineage / depth / the #794-frozen surface), but with **no
    response obligation**. It is a first-class, *explicitly-passed* field — never
    inferred from ``request_id``-presence — so the discriminated-union discipline
    holds at the call site. The triad of readers that derive "open request" from
    this frame (this issue's quiescence guard nudge/``no_return`` loop, the #1131
    totality gate, the #1132 cap) all filter ``awaited=true``; an absent field
    reads as ``awaited=true`` (additive/legacy), so existing rows keep their
    obligation.

    Idempotency is the **caller's** responsibility: the ``create_child_session``
    path gates this behind its first-spawn ``ON CONFLICT`` check, so a replayed
    wake opens the edge exactly once (see :func:`get_open_request_ids` for the
    asked-minus-answered derivation this feeds).

    ``summary`` (#1413) is a short (~60-char) truncated preview of the request
    input, carried so the always-on tail-injected obligations block can render a
    human-readable line for an obligation whose original request user message has
    been windowed out of context. **Purely additive**: omitted (the legacy/None
    case) it is simply not written to the frame, and the obligations reader treats
    an absent field as ``None`` -> an id-only render line (no migration, #1131-proof
    -- frame-extension rather than a LEFT-JOIN on the soon-retired ``metadata`` blob).
    """
    data: dict[str, Any] = {
        "event": "request_opened",
        "request_id": request_id,
        "caller": caller,
        "depth": depth,
        "environment_id": environment_id,
        "frozen_surface": frozen_surface,
        "vault_ids": vault_ids,
        "awaited": awaited,
    }
    if output_schema is not None:
        # The per-request schema the answer's ``value`` must satisfy (#1131): carried
        # on the trusted edge so ``return`` enforcement no longer reads the forgeable
        # ``metadata.request`` user-message blob.
        data["output_schema"] = output_schema
    if summary is not None:
        data["summary"] = summary
    if rlm_token_budget is not None:
        # rlm spawns only (docs/rlm.md): the child's inherited child-token
        # allowance, read back by the child's own rlm dispatch admission. Its
        # presence is also the discriminant marking an rlm spawn edge.
        data["rlm_token_budget"] = rlm_token_budget
    await queries.append_event(
        conn,
        account_id=account_id,
        session_id=session_id,
        kind="lifecycle",
        data=data,
    )


async def get_session(
    conn: asyncpg.Connection[Any], session_id: str, *, account_id: str
) -> Session:
    rec = await conn.fetchrow(
        f"SELECT sessions.*, ({_SESSION_STATUS_EXPR}) AS status "
        "FROM sessions WHERE id = $1 AND account_id = $2",
        session_id,
        account_id,
    )
    if rec is None:
        raise NotFoundError(f"session {session_id} not found", detail={"id": session_id})
    return _row_to_session(rec)


async def get_session_bare(
    conn: asyncpg.Connection[Any], session_id: str, *, account_id: str
) -> Session:
    """Session read WITHOUT deriving ``status`` (defaults to ``idle``).

    For callers that need only core columns and never surface the status label
    — existence checks, the worker step path, connection lookups — so they
    don't pay for the ``_SESSION_STATUS_EXPR`` event-log derivation.
    """
    return await _get_scoped(
        conn,
        table="sessions",
        id_=session_id,
        account_id=account_id,
        row=_row_to_session,
        noun="session",
    )


async def get_session_workspace_path(
    conn: asyncpg.Connection[Any], session_id: str, *, account_id: str
) -> str:
    """Return the host-side workspace path stored on the session row."""
    val: str | None = await conn.fetchval(
        "SELECT workspace_volume_path FROM sessions WHERE id = $1 AND account_id = $2",
        session_id,
        account_id,
    )
    if val is None:
        raise NotFoundError(f"session {session_id} not found", detail={"id": session_id})
    return val


async def get_session_focal_channel(
    conn: asyncpg.Connection[Any], session_id: str, *, account_id: str
) -> str | None:
    """Return the session's current ``focal_channel`` (or NULL = phone down)."""
    focal: str | None = await conn.fetchval(
        "SELECT focal_channel FROM sessions WHERE id = $1 AND account_id = $2",
        session_id,
        account_id,
    )
    return focal


async def is_session_focal_locked(
    conn: asyncpg.Connection[Any], session_id: str, *, account_id: str
) -> bool:
    """Return whether the session's focal channel is locked.

    The flag is set at session creation by per_chat-mode spawns (and any
    future spawner that wants to pin a session to a single channel).
    ``switch_channel`` rejects any mutation when this returns ``True``.

    Raises :class:`NotFoundError` if the session row doesn't exist —
    callers in the harness should never reach this with an invalid
    session id, so a missing row is a real bug, not a permission state.
    """
    locked: bool | None = await conn.fetchval(
        "SELECT focal_locked FROM sessions WHERE id = $1 AND account_id = $2",
        session_id,
        account_id,
    )
    if locked is None:
        raise NotFoundError(f"session {session_id} not found", detail={"id": session_id})
    return locked


async def set_session_focal_channel(
    conn: asyncpg.Connection[Any],
    session_id: str,
    focal: str | None,
    *,
    account_id: str,
) -> None:
    """Mutate the session's ``focal_channel``.  Only ``switch_channel``
    should call this — it's the single source of truth for the agent's
    focal attention.
    """
    await conn.execute(
        "UPDATE sessions SET focal_channel = $1 WHERE id = $2 AND account_id = $3",
        focal,
        session_id,
        account_id,
    )


async def set_session_channels(
    conn: asyncpg.Connection[Any],
    session_id: str,
    channels: list[str],
    *,
    account_id: str,
) -> None:
    """Overwrite the session's maintained ``channels`` array (issue #1742).

    Used ONLY by the repair-on-mismatch fallback (``switch_channel``'s
    membership check, the POST ``metadata.channel`` bound-check) to fix a
    stored set that's drifted from :func:`aios.db.queries.events.recompute_session_channels`'s
    ground truth — never on the hot per-step loop path, which only reads.
    """
    await conn.execute(
        "UPDATE sessions SET channels = $1 WHERE id = $2 AND account_id = $3",
        channels,
        session_id,
        account_id,
    )


async def get_session_provisioning(
    conn: asyncpg.Connection[Any],
    session_id: str,
    *,
    account_id: str,
) -> tuple[str, dict[str, str], int, str | None]:
    """Return ``(workspace_volume_path, env, spec_version, snapshot_ref)`` for
    provisioning a session's container.

    ``spec_version`` (issue #713) is the snapshot the backend stamps onto
    the :class:`SandboxHandle` so the registry can detect drift on a warm
    hit and recycle the cached sandbox. ``snapshot_ref`` (durable session
    sandboxes) is the DB snapshot pointer the provision path resolves
    through the store to decide resume-source vs cold-start; ``None`` ⇒ no
    snapshot.
    """
    row = await conn.fetchrow(
        "SELECT workspace_volume_path, env, spec_version, snapshot_ref "
        "FROM sessions WHERE id = $1 AND account_id = $2",
        session_id,
        account_id,
    )
    if row is None:
        raise NotFoundError(f"session {session_id} not found", detail={"id": session_id})
    raw_env = row["env"]
    env: dict[str, str] = raw_env
    return row["workspace_volume_path"], env, row["spec_version"], row["snapshot_ref"]


async def list_sessions(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    agent_id: str | None = None,
    status: SessionStatus | None = None,
    parent_run_id: str | None = None,
    limit: int = 50,
    after: str | None = None,
) -> list[Session]:
    """Keyset-paginated session list with derived ``status`` ({active, idle}).

    ``status`` is a derived expression, not a column, so it's passed to
    ``_list_scoped`` as an expression-filter (a static literal, never user
    input) rather than a column name. That keeps the keyset/limit/has-more
    boilerplate in one place while still applying the filter in SQL — so
    pagination stays correct (post-filtering the page would conflate "few
    matches in this window" with "no more results"). ``status`` and
    ``last_event_at`` are derived in ``extra_select``.

    ``parent_run_id`` is a plain column filter (the session is a workflow run's
    ``agent()`` child) — the session-side analog of filtering runs by their
    parent, so a run can list the agent sessions it spawned.

    Soft-archived rows are normally invisible, but a workflow ``agent()`` child
    reclaims itself on idle (``archive_when_idle``), so an account could never
    enumerate its spent judgment nodes or sum their token spend (#831). Two
    listings therefore drop the ``archived_at IS NULL`` clause: filtering by
    ``parent_run_id`` (a run enumerating its children, alive or terminal) and
    ``status="archived"`` (the explicit terminal-status query). All other
    listings stay archive-blind. The ``({_SESSION_STATUS_EXPR})`` filter then
    matches ``'archived'`` like any other status, so ``status="archived"``
    returns exactly the archived rows.
    """
    include_archived = parent_run_id is not None or status == "archived"
    return await _list_scoped(
        conn,
        table="sessions",
        account_id=account_id,
        row=_row_to_session,
        limit=limit,
        after=after,
        include_archived=include_archived,
        filters=[
            ("agent_id", agent_id),
            (f"({_SESSION_STATUS_EXPR})", status),
            ("parent_run_id", parent_run_id),
        ],
        # last_event_at: timestamp of the session's newest event. ``ORDER BY
        # seq DESC LIMIT 1`` rides the (session_id, seq) index — O(log n) even
        # for huge sessions (unlike MAX(created_at), which has no index).
        extra_select=(
            f"({_SESSION_STATUS_EXPR}) AS status, "
            "(SELECT e.created_at FROM events e WHERE e.session_id = sessions.id "
            "ORDER BY e.seq DESC LIMIT 1) AS last_event_at"
        ),
    )


async def lock_active_session_for_update(
    conn: asyncpg.Connection[Any], session_id: str, *, account_id: str
) -> None:
    """Enforce the active-session precondition: ``SELECT FOR UPDATE`` the
    session row, raise :class:`NotFoundError` on miss / wrong account,
    :class:`ConflictError` when status is ``errored`` (a user message is
    required to resume).

    Must be called inside an outer ``conn.transaction()`` block — the
    row lock is what serialises concurrent retries on the same session.
    """
    row = await conn.fetchrow(
        f"SELECT ({_SESSION_ERRORED_EXPR}) AS errored "
        "FROM sessions WHERE id = $1 AND account_id = $2 FOR UPDATE",
        session_id,
        account_id,
    )
    if row is None:
        raise NotFoundError(
            f"session {session_id} not found",
            detail={"session_id": session_id},
        )
    if row["errored"]:
        raise ConflictError(
            f"session {session_id} is errored; post a user message to resume",
            detail={"session_id": session_id, "status": "errored"},
        )


async def lock_writable_session_for_update(
    conn: asyncpg.Connection[Any], session_id: str, *, account_id: str
) -> None:
    """Row-lock an unarchived session without rejecting its derived status.

    User messages are the recovery path for errored sessions, so their
    idempotency lock must serialize retries while still allowing that append.
    Archived sessions remain unwritable, matching :func:`append_event`.
    """
    row = await conn.fetchrow(
        "SELECT 1 FROM sessions "
        "WHERE id = $1 AND account_id = $2 AND archived_at IS NULL FOR UPDATE",
        session_id,
        account_id,
    )
    if row is None:
        raise NotFoundError(
            f"session {session_id} not found",
            detail={"session_id": session_id},
        )


async def decrement_open_tool_call_count(
    conn: asyncpg.Connection[Any], session_id: str, *, account_id: str
) -> None:
    """Compensate the id-blind +1 that ``append_event`` applied for a
    tool_call when its result append later dedup-skips (issue #841).

    ``append_event`` increments ``open_tool_call_count`` by ``len(tool_calls)``
    when the assistant turn lands — id-blind. Every decrement path (the
    ``role:"tool"`` append) is short-circuited when a ``tool_call_id`` already
    has a result, so a reused/duplicate id leaks a permanent +1: the session
    stays a wake candidate (``_SESSION_ACTIVE_EXPR`` / ``CANDIDATE_ROWS_SQL``)
    forever. Both dedup-skip sites call this to apply the matching -1 in the
    SAME session-row-lock transaction. ``GREATEST(..., 0)`` clamps the floor.

    By design this fires on EVERY dedup-skip — including a genuine idempotent
    retry of an already-resolved call (network blip, worker-API race), not only
    the reused-id case. When a SIBLING tool_call is still genuinely open at that
    moment, the unconditional -1 can transiently drive the counter below the
    true open count. That undercount is safe and self-healing: ``GREATEST(...,
    0)`` keeps it non-negative, and the independent ``last_stimulus_seq >
    last_reacted_seq`` wake path re-activates the session when the sibling's real
    result lands — so the counter is never the sole thing keeping the session
    alive and the session is never permanently stalled.

    Must be called inside the caller's transaction, while it holds the
    session row lock (``SELECT ... FOR UPDATE`` /
    ``lock_active_session_for_update``).
    """
    await conn.execute(
        "UPDATE sessions "
        "SET open_tool_call_count = GREATEST(open_tool_call_count - 1, 0) "
        "WHERE id = $1 AND account_id = $2",
        session_id,
        account_id,
    )


async def set_session_stop_reason(
    conn: asyncpg.Connection[Any],
    session_id: str,
    stop_reason: dict[str, Any],
    *,
    account_id: str,
) -> None:
    """Record why the most recent step ended. ``status`` is derived from the
    event log (see ``_SESSION_STATUS_EXPR``), so this writes only
    ``stop_reason`` — the console renders the Errored pill from
    ``status == 'idle' AND stop_reason.type == 'error'``.

    ``archived_at IS NULL`` guards the archive race: every caller is
    worker-internal and a silent no-op is the right contract.
    """
    await conn.execute(
        "UPDATE sessions SET stop_reason = $1::jsonb, updated_at = now() "
        "WHERE id = $2 AND account_id = $3 AND archived_at IS NULL",
        json.dumps(stop_reason),
        session_id,
        account_id,
    )


async def increment_session_usage(
    conn: asyncpg.Connection[Any],
    session_id: str,
    *,
    account_id: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    cost_microusd: int = 0,
) -> int:
    """Atomically add token and spend counts; return the account spend total."""
    async with conn.transaction():
        await conn.execute(
            "UPDATE sessions SET "
            "input_tokens = input_tokens + $2, "
            "output_tokens = output_tokens + $3, "
            "cache_read_input_tokens = cache_read_input_tokens + $4, "
            "cache_creation_input_tokens = cache_creation_input_tokens + $5, "
            "cost_microusd = cost_microusd + $6 "
            "WHERE id = $1 AND account_id = $7",
            session_id,
            input_tokens,
            output_tokens,
            cache_read_input_tokens,
            cache_creation_input_tokens,
            cost_microusd,
            account_id,
        )
        spent = await conn.fetchval(
            "UPDATE accounts SET spent_microusd = spent_microusd + $1 "
            "WHERE id = $2 RETURNING spent_microusd",
            cost_microusd,
            account_id,
        )
    return int(spent or 0)


async def get_session_model(
    conn: asyncpg.Connection[Any], session_id: str, *, account_id: str
) -> str:
    """Return the bound model for ``session_id`` in one round trip.

    Pinned ``agent_version`` wins when set; otherwise the live agent's
    current ``model`` is returned.

    Filters ``s.account_id``, ``a.account_id``, and ``av.account_id``
    against the same caller account — ``insert_session`` only relies on
    the ``agent_id REFERENCES agents(id)`` FK and does not validate
    cross-account ownership, so a session row can carry an ``agent_id``
    from a different tenant. Without these predicates this read would
    surface the foreign tenant's bound model (which may itself be a
    sensitive routing target) [security]. The ``av.account_id``
    predicate lives in the LEFT JOIN's ON clause so an unpinned session
    (av-side NULL) still resolves to the agent's current model.
    """
    row = await conn.fetchrow(
        """
        SELECT COALESCE(s.model, av.model, a.model) AS model
          FROM sessions s
     LEFT JOIN agents a
            ON a.id = s.agent_id
           AND a.account_id = $2
     LEFT JOIN agent_versions av
            ON av.agent_id = s.agent_id
           AND av.version = s.agent_version
           AND av.account_id = $2
         WHERE s.id = $1
           AND s.account_id = $2
        """,
        session_id,
        account_id,
    )
    if row is None or row["model"] is None:
        raise NotFoundError(f"session {session_id} not found", detail={"id": session_id})
    return str(row["model"])


async def list_attachment_paths_for_sessions(
    conn: asyncpg.Connection[Any],
    session_ids: list[str],
) -> dict[str, set[str]]:
    """Return ``in_sandbox_path`` values referenced by each session's events.

    Returns a map keyed by session_id; sessions with no attachment
    references appear with an empty set so callers can distinguish
    "no events with attachments" from "session unknown".

    Note: this query infers reference state purely from event rows, so
    a ``session_id`` whose row was deleted (or whose events were
    purged) returns an empty set indistinguishable from "session
    exists but has no attachments". The orphan GC sweep relies on
    this and will treat every on-disk file under such a session's
    ``_attachments/<session_id>/`` dir as orphaned — which is the
    intended behavior, but worth being explicit about.
    """
    result: dict[str, set[str]] = {sid: set() for sid in session_ids}
    if not session_ids:
        return result
    # Each attachment record carries the original at ``in_sandbox_path``
    # plus, when staging produced a downsampled inline copy, a sibling
    # at ``inline.in_sandbox_path``.  The GC sweep deletes any file under
    # the session's ``_attachments`` dir whose sandbox path isn't in the
    # returned set, so both paths must be returned or the inline siblings
    # will be reaped as orphans.  ``->>`` returns NULL when the key is
    # missing — those rows drop out at the application-level filter below.
    rows = await conn.fetch(
        """
        SELECT session_id,
               attachment->>'in_sandbox_path' AS path,
               attachment->'inline'->>'in_sandbox_path' AS inline_path
          FROM events,
               jsonb_array_elements(data->'metadata'->'attachments') AS attachment
         WHERE session_id = ANY($1::text[])
           AND data->'metadata' ? 'attachments'
           AND jsonb_typeof(data->'metadata'->'attachments') = 'array'
        """,
        session_ids,
    )
    for row in rows:
        path = row["path"]
        if path is not None:
            result[row["session_id"]].add(path)
        inline_path = row["inline_path"]
        if inline_path is not None:
            result[row["session_id"]].add(inline_path)
    return result


async def update_session(
    conn: asyncpg.Connection[Any],
    session_id: str,
    *,
    account_id: str,
    agent_id: str | None = None,
    agent_version: int | None | EllipsisType = ...,
    title: str | None | EllipsisType = ...,
    metadata: dict[str, Any] | None = None,
    outbound_suppression: str | None = None,
) -> Session:
    # Refuse updates to archived sessions: read paths
    # (``list_sessions``, the worker, the resolver) all filter
    # ``archived_at IS NULL``, so a rewrite of an archived row has no
    # observable effect — but the bare UPDATE below would still commit
    # the new values and the RETURNING-built response would lie back
    # to the caller as if the update took.  Mirrors the symmetric
    # raise on archived rows in ``update_agent`` / ``update_environment``
    # / ``update_session_template`` (PR #547) / ``update_vault``
    # (PR #554).
    #
    # Load-bearing for the resource-attachment writes inside
    # ``service.update_session`` (vault_ids / memory / github
    # resources): those callers run in the same transaction but their
    # query-layer functions don't independently check ``archived_at``,
    # so this raise is the only synchronous barrier against rewriting
    # attachments on an archived session.
    current = await queries.get_session(conn, session_id, account_id=account_id)
    if current.archived_at is not None:
        raise ConflictError(
            f"session {session_id} is archived",
            detail={"id": session_id},
        )

    args: list[Any] = [session_id]  # $1 = session_id
    fields: list[tuple[str, Any, str | None]] = []
    if agent_id is not None:
        fields.append(("agent_id", agent_id, None))
    if agent_version is not ...:
        fields.append(("agent_version", agent_version, None))
    if title is not ...:
        fields.append(("title", title, None))
    if metadata is not None:
        fields.append(("metadata", metadata, "jsonb"))
    if outbound_suppression is not None:
        fields.append(("outbound_suppression", outbound_suppression, None))
    sets = _build_set_assignments(fields, args)
    if not sets:
        return current

    sets.append("updated_at = now()")
    args.append(account_id)
    sql = (
        f"UPDATE sessions SET {', '.join(sets)} "
        f"WHERE id = $1 AND account_id = ${len(args)} AND archived_at IS NULL RETURNING *"
    )
    row = await conn.fetchrow(sql, *args)
    if row is None:
        # The upfront read already raised on missing rows, so a no-row
        # UPDATE here means an archive committed between read and UPDATE.
        raise ConflictError(f"session {session_id} is archived", detail={"id": session_id})
    return _row_to_session(row)


async def archive_session(
    conn: asyncpg.Connection[Any], session_id: str, *, account_id: str, idempotent: bool = False
) -> Session:
    row = await _archive_scoped(
        conn,
        table="sessions",
        id_=session_id,
        account_id=account_id,
        noun="session",
        idempotent=idempotent,
    )
    return _row_to_session(row)


async def delete_session(
    conn: asyncpg.Connection[Any], session_id: str, *, account_id: str
) -> None:
    async with conn.transaction():
        row = await conn.fetchrow(
            "SELECT 1 FROM sessions WHERE id = $1 AND account_id = $2",
            session_id,
            account_id,
        )
        if row is None:
            raise NotFoundError(
                f"session {session_id} not found",
                detail={"id": session_id},
            )
        await conn.execute(
            "DELETE FROM session_vaults WHERE session_id = $1 AND account_id = $2",
            session_id,
            account_id,
        )
        await conn.execute(
            "DELETE FROM events WHERE session_id = $1 AND account_id = $2",
            session_id,
            account_id,
        )
        # bindings.session_id now carries ON DELETE CASCADE (migration
        # 0110 restored the cascade dropped in the 0033 redesign), so
        # binding rows are cleaned up by Postgres as a side effect of the
        # `DELETE FROM sessions` below — no hand-DELETE required.
        await conn.execute(
            "DELETE FROM sessions WHERE id = $1 AND account_id = $2",
            session_id,
            account_id,
        )


def _mint_clone_ingest_token_hash() -> str:
    """A fresh SHA-256 ingest-token hash for an external_event trigger clone.

    The clone can't carry the parent's hash (the ``triggers_ingest_token_hash``
    UNIQUE index) so each external_event trigger gets a fresh one.  The
    plaintext is dropped — the owner rotates via ``update_trigger`` to obtain a
    usable secret, exactly as the session-create attach path's
    ``mint_ingest_token_hash`` does.  Kept db-layer-local to avoid a cycle back
    through ``services.triggers``; the format is opaque (only the hash is
    stored) so matching the service's plaintext scheme byte-for-byte isn't
    required — just that it's a unique 64-hex sha256.
    """
    return hashlib.sha256(secrets.token_bytes(32)).hexdigest()


async def clone_session(
    conn: asyncpg.Connection[Any],
    parent_session_id: str,
    *,
    account_id: str,
    workspace_path: str | None = None,
) -> Session:
    """Clone a session into a new one with the same prefix of events.

    Which columns are copied, reset, or minted fresh is no longer enumerated
    inline — the four session-copy INSERTs are GENERATED from the per-column
    clone policies in ``clone_policy.py`` (see ``build_projection``), so a
    migration that adds a column MUST classify it or CI's completeness gate
    (``test_clone_policy_completeness.py``) fails.  This closes the #1676 class
    of bug where a new column was silently dropped on every clone — most
    consequentially the 0127 ``cumulative_*`` class-mass counters, whose loss
    ran the #1609 R_eff context-window blend composition-blind on clones.

    The salient policy decisions the generator encodes:

    * the clone inherits the parent's surface (``tools`` / ``mcp_servers`` /
      ``http_servers`` / ``surface_frozen``), authority pins (``model`` /
      ``litellm_extra``), ``env``, vault bindings, memory-store and
      github-repository attachments, event prefix (with all ``cumulative_*``
      counters), ``focal_channel`` and ``focal_locked`` — so its next forward
      step sees a context byte-identical to the parent's.  ``focal_locked``
      MUST follow ``focal_channel``: a per_chat parent cloned without its lock
      would inherit the bound channel but bypass the
      ``is_session_focal_locked`` gate on ``switch_channel``, escaping per_chat
      isolation.  (``status`` derives from the event log, so the clone is idle
      like the parent it was required to be.)
    * ``parent_run_id`` / ``origin`` RESET: a clone must NOT join a live run's
      sweep/cancel cascade nor masquerade as that run's child.
    * cumulative ``input_tokens`` / ``output_tokens`` / ``cost_microusd`` and
      ``snapshot_ref`` RESET: usage was paid on the parent, and the snapshot
      pointer belongs to the parent's workspace.
    * global-PK attachment ids (github-repository) are minted fresh; an
      external_event trigger's ``ingest_token_hash`` is minted fresh (the
      UNIQUE index forbids copying) while a non-external trigger's stays NULL
      (the iff CHECK), and per-trigger runtime state (``running_since``,
      ``consecutive_failures``, last-fire fields) RESETs.

    Workspace volume defaults to a fresh ``workspace_root / new_id`` path so
    clones don't fight over the same files.  Pass ``workspace_path`` to
    override (e.g. share a read-only volume between clones).

    Refuses parents that aren't ``idle`` or ``terminated``: a ``running``
    parent has tool tasks in flight whose results would land only on its
    own session_id, leaving the clone's expected event stream undefined.
    The clone primitive locks the parent row for the copy, so concurrent
    appenders serialize behind it and the copied seq range is gapless.

    Since ``enabled``/``next_fire`` are copied verbatim, a clone of a
    trigger-bearing session is itself a trigger-CREATING write path, so it
    honors ``Settings.triggers_per_account_max``. EVERY clone acquires the
    same ``acquire_account_triggers_lock`` advisory lock ``add_trigger`` /
    ``update_trigger``'s re-enable / session-attach use — unconditionally, at
    the very top of the transaction, BEFORE the parent ``FOR UPDATE`` — to
    match the siblings' advisory-first / session-row-second lock order (they
    take the advisory lock, then the session row via their INSERT's FK). A
    clone that locked the session row first would invert that order and
    deadlock against a concurrent ``add_trigger``. With the lock held, the
    parent's enabled-trigger count is exact through the copy, so when the
    parent owns any triggers this enforces the cap against the account's
    existing enabled count plus the clone's — raising ``RateLimitedError``
    (and rolling back the whole clone) on breach. A trigger-free parent still
    takes the lock (for ordering) but skips the cap CHECK; its empty row set
    is self-consistent with the copy INSERT, which copies nothing.
    """
    new_id = make_id(SESSION)
    if workspace_path is None:
        workspace_path = _default_workspace_path(account_id, new_id)

    async with conn.transaction():
        # Take the per-account trigger advisory lock FIRST — before the parent
        # session FOR UPDATE below — because the sibling trigger-creating paths
        # (add_trigger / update_trigger re-enable / session-attach) all acquire
        # it before the session-row lock their trigger INSERT's FK takes. A
        # clone that took the session row lock first, then this advisory lock,
        # would invert that order and deadlock (AB/BA) against a concurrent
        # add_trigger on the same account+session. Unconditional for every clone
        # (not gated on trigger_count): the lock ordering is what makes the cap
        # contractual, so its acquisition can't depend on the racy trigger set.
        # It is xact-scoped, so it's held across the whole clone — the deliberate
        # cost of correct ordering; clones are infrequent and the lock is
        # per-account.
        await queries.acquire_account_triggers_lock(conn, account_id)
        row = await conn.fetchrow(
            f"SELECT archived_at, ({_SESSION_ACTIVE_EXPR}) AS active, "
            f"({_SESSION_ERRORED_EXPR}) AS errored "
            "FROM sessions WHERE id = $1 AND account_id = $2 FOR UPDATE",
            parent_session_id,
            account_id,
        )
        if row is None:
            raise NotFoundError(
                f"session {parent_session_id} not found",
                detail={"id": parent_session_id},
            )
        # Refuse archived parents: cloning would resurrect the parent's
        # event log into a live new session, defeating the archive
        # intent.  Same family as PR #573 / #547 / #554 / #587 —
        # archive must hold across every mutation/copy surface.
        if row["archived_at"] is not None:
            raise ConflictError(
                f"session {parent_session_id} is archived",
                detail={"id": parent_session_id},
            )
        # Only clone idle (quiescent, non-errored) parents: an active parent
        # has tool tasks in flight whose results would land only on its own
        # session_id, leaving the clone's expected event stream undefined; an
        # errored parent isn't a clean base. (Derived ``status == 'idle'`` =
        # not active; errored is checked explicitly for a precise message.)
        if row["active"] or row["errored"]:
            state = "errored" if row["errored"] else "active"
            raise ConflictError(
                f"can only clone idle sessions; parent {parent_session_id} is {state}",
                detail={"id": parent_session_id, "status": state},
            )

        # The four session-copy INSERTs below are GENERATED from the
        # per-column clone policies in ``clone_policy.py`` — the column
        # projection is no longer hand-enumerated, so a migration that adds a
        # column without classifying it fails the completeness gate in CI
        # (tests/integration/test_clone_policy_completeness.py) rather than
        # silently dropping the column on every clone (#1676).  The parent row
        # is aliased ``s``; ``$1`` is the fresh session id, ``$2`` the
        # workspace path, ``$3`` the parent id.
        sessions_proj = build_projection(
            SESSIONS_POLICY,
            source_alias="",  # single-row copy: bare column names, no alias
            new_id_expr="$1",
            session_id_param="$1",  # sessions has no REMAP_SESSION column
            new_value_exprs={"workspace_volume_path": "$2"},
        )
        new_row = await conn.fetchrow(
            f"""
            INSERT INTO sessions ({sessions_proj.insert_columns_sql})
            SELECT {sessions_proj.select_list_sql}
              FROM sessions WHERE id = $3
            RETURNING *
            """,
            new_id,
            workspace_path,
            parent_session_id,
        )
        assert new_row is not None

        await conn.execute(
            "INSERT INTO session_vaults (session_id, vault_id, rank, account_id) "
            "SELECT $1, vault_id, rank, account_id FROM session_vaults WHERE session_id = $2",
            new_id,
            parent_session_id,
        )

        # Resource attachments.  ``session_memory_stores`` has a
        # composite PK so a direct INSERT/SELECT works.
        # ``session_github_repositories.id`` is a global PK, so each
        # row needs a fresh ULID minted via the same ordinal-join
        # pattern the events copy below uses.  Direct INSERT/SELECT
        # bypasses the archival check the normal attach path enforces
        # — by design: a clone snapshots the parent's attachment
        # state at clone time, including references to stores
        # archived after the parent attached them.
        mem_proj = build_projection(
            SESSION_MEMORY_STORES_POLICY,
            source_alias="",
            new_id_expr="$1",  # no MINT_ID column on this composite-PK relation
            session_id_param="$1",
        )
        await conn.execute(
            f"""
            INSERT INTO session_memory_stores ({mem_proj.insert_columns_sql})
            SELECT {mem_proj.select_list_sql}
              FROM session_memory_stores WHERE session_id = $2
            """,
            new_id,
            parent_session_id,
        )
        gh_count: int = await conn.fetchval(
            "SELECT COUNT(*) FROM session_github_repositories WHERE session_id = $1",
            parent_session_id,
        )
        new_gh_ids = [make_id(GITHUB_REPOSITORY) for _ in range(gh_count)]
        gh_proj = build_projection(
            SESSION_GITHUB_REPOSITORIES_POLICY,
            source_alias="s",
            new_id_expr="i.id",
            session_id_param="$2",
        )
        await conn.execute(
            f"""
            INSERT INTO session_github_repositories ({gh_proj.insert_columns_sql})
            SELECT {gh_proj.select_list_sql}
              FROM (
                SELECT *, row_number() OVER (ORDER BY rank) AS rn
                  FROM session_github_repositories WHERE session_id = $1
              ) s
              JOIN unnest($3::text[]) WITH ORDINALITY AS i(id, rn) USING (rn)
            """,
            parent_session_id,
            new_id,
            new_gh_ids,
        )

        # triggers.id is a global PK — mint fresh per clone. Runtime state
        # (running_since / last_fire_* / consecutive_failures) is RESET on
        # the clone: it inherits source + source_spec + action + next_fire so
        # it continues firing on the parent's cadence, but starts with fresh
        # failure counters and no in-flight claim. Carrying the FULL
        # source_spec (not just the old `schedule` column) is what lets a
        # clone of a session owning a one-shot trigger succeed — the
        # pre-rename INSERT copied `schedule` but not `fire_at` and tripped
        # the old XOR CHECK, aborting the whole clone transaction (§6.1).
        trigger_count: int = await conn.fetchval(
            "SELECT COUNT(*) FROM triggers WHERE owner_session_id = $1",
            parent_session_id,
        )
        if trigger_count:
            # Clone copies triggers with enabled=COPY (TRIGGERS_POLICY), so it is a
            # trigger-CREATING write path and must honor the per-account active
            # cap like add_trigger / update_trigger re-enable / session-attach do.
            # The account advisory lock is already held (acquired at the top of
            # this transaction), so this enabled count is exact — or stricter,
            # since unlocked disables only reduce it — through the INSERT below;
            # a concurrent locked re-enable can't flip a row between the count and
            # the copy. The cap CHECK is gated on trigger_count: a trigger-free
            # clone copies nothing and needs no check (the lock is still taken
            # unconditionally above, for ordering).
            enabled_to_clone: int = await conn.fetchval(
                "SELECT COUNT(*) FROM triggers WHERE owner_session_id = $1 AND enabled",
                parent_session_id,
            )
            if enabled_to_clone:
                from aios.config import get_settings

                cap = get_settings().triggers_per_account_max
                existing_account = await queries.count_account_triggers(
                    conn, account_id=account_id, enabled_only=True
                )
                if existing_account + enabled_to_clone > cap:
                    raise RateLimitedError(
                        "cloning this session would exceed the active-trigger cap: "
                        f"{existing_account} existing + {enabled_to_clone} cloned > {cap}. "
                        "Disable or remove triggers to free slots before cloning."
                    )
        new_trigger_ids = [make_id(TRIGGER) for _ in range(trigger_count)]
        # A fresh ingest-token hash per trigger row, used ONLY by the
        # source-conditional MINT_INGEST_TOKEN arm: an external_event trigger
        # can't carry the parent's hash (the triggers_ingest_token_hash UNIQUE
        # index), and a non-external row must have NULL (the
        # triggers_ingest_token_iff_external_event CHECK).  Non-external rows
        # discard their generated hash.  The plaintext is dropped — the owner
        # rotates via update_trigger to obtain a usable secret (same posture as
        # the session-create attach path's mint_ingest_token_hash).
        new_ingest_hashes = [_mint_clone_ingest_token_hash() for _ in range(trigger_count)]
        trig_proj = build_projection(
            TRIGGERS_POLICY,
            source_alias="s",
            new_id_expr="i.id",
            session_id_param="$2",
            new_value_exprs={
                "ingest_token_hash": (
                    "CASE WHEN s.source = 'external_event' THEN i.ingest_token_hash ELSE NULL END"
                )
            },
        )
        await conn.execute(
            f"""
            INSERT INTO triggers ({trig_proj.insert_columns_sql})
            SELECT {trig_proj.select_list_sql}
              FROM (
                SELECT *, row_number() OVER (ORDER BY created_at) AS rn
                  FROM triggers WHERE owner_session_id = $1
              ) s
              JOIN unnest($3::text[], $4::text[]) WITH ORDINALITY
                     AS i(id, ingest_token_hash, rn) USING (rn)
            """,
            parent_session_id,
            new_id,
            new_trigger_ids,
            new_ingest_hashes,
        )

        # Events are gapless 1..last_event_seq per session, so we pre-generate
        # exactly that many fresh evt_ ids and join by ordinal.  Event ids are
        # PRIMARY KEY so they must change; everything else is preserved
        # verbatim — context builder semantics depend on it.
        new_event_ids = [make_id(EVENT) for _ in range(new_row["last_event_seq"])]
        events_proj = build_projection(
            EVENTS_POLICY,
            source_alias="s",
            new_id_expr="i.id",
            session_id_param="$2",
        )
        await conn.execute(
            f"""
            INSERT INTO events ({events_proj.insert_columns_sql})
            SELECT {events_proj.select_list_sql}
              FROM (
                SELECT *, row_number() OVER (ORDER BY seq) AS rn
                  FROM events WHERE session_id = $1
              ) s
              JOIN unnest($3::text[]) WITH ORDINALITY AS i(id, rn) USING (rn)
            """,
            parent_session_id,
            new_id,
            new_event_ids,
        )

    return _row_to_session(new_row)
