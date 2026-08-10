"""Context-variable queries (docs/rlm.md).

A subsystem module of the ``aios.db.queries`` package — see ``__init__`` for the
shared scoping helpers and the package-level re-export contract. Raw SQL against
asyncpg, same conventions as the rest of the package.

Read-path visibility for a session is the three-way union resolved by
:func:`resolve_readable_variable` / :func:`list_readable_variables`:

1. the session's own ``scope='session'`` variables,
2. its agent's ``scope='agent'`` variables (the world-model plane),
3. variables **granted** to it via ``context_variable_grants`` (written at
   ``rlm_query`` spawn time; read-only by construction — the write path
   never consults grants).

Precedence on name collision is that same order. Listing projections never
select ``content`` — metadata stays cheap no matter how large values grow.
"""

from __future__ import annotations

import json
from typing import Any

import asyncpg

from aios.actors import actor_columns, actor_from_row
from aios.db.queries import _archive_scoped, _build_set_assignments
from aios.errors import NotFoundError
from aios.ids import CONTEXT_VARIABLE, make_id
from aios.models.context_variables import ContextVariable, VariableKind, VariableScope

# Every projection that skips content. ``NULL AS content`` keeps the row
# mapper uniform between metadata-only and with-content reads.
_META_COLUMNS = (
    "id, scope, session_id, agent_id, name, kind, description, schema_tag, "
    "NULL AS content, content_sha256, content_size_bytes, metadata, "
    "created_by_type, created_by_ref, created_at, updated_at, archived_at"
)

# The three-way readability predicate. Placeholders: $1 account_id,
# $2 session_id, $3 agent_id (NULL for agentless children).
_READABLE_WHERE = """
    v.account_id = $1 AND v.archived_at IS NULL AND (
        (v.scope = 'session' AND v.session_id = $2)
        OR (v.scope = 'agent' AND v.agent_id = $3)
        OR EXISTS (
            SELECT 1 FROM context_variable_grants g
            WHERE g.session_id = $2 AND g.context_variable_id = v.id
        )
    )
"""

# Collision precedence: own session scope, then agent scope, then grants.
_READABLE_PRECEDENCE = """
    CASE
        WHEN v.scope = 'session' AND v.session_id = $2 THEN 0
        WHEN v.scope = 'agent' THEN 1
        ELSE 2
    END
"""


def _row_to_context_variable(row: asyncpg.Record) -> ContextVariable:
    return ContextVariable(
        id=row["id"],
        scope=row["scope"],
        session_id=row["session_id"],
        agent_id=row["agent_id"],
        name=row["name"],
        kind=row["kind"],
        description=row["description"],
        schema_tag=row["schema_tag"],
        content=row["content"],
        content_sha256=row["content_sha256"],
        content_size_bytes=row["content_size_bytes"],
        metadata=row["metadata"],
        created_by=actor_from_row(row),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        archived_at=row["archived_at"],
    )


async def upsert_context_variable(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    scope: VariableScope,
    session_id: str | None,
    agent_id: str | None,
    name: str,
    kind: VariableKind,
    description: str,
    schema_tag: str | None,
    content: str,
    content_sha256: str,
    content_size_bytes: int,
    metadata: dict[str, Any],
) -> ContextVariable:
    """Create-or-overwrite a variable at its scope key.

    The conflict target is the scope's partial unique index, so an existing
    live handle is overwritten in place (id preserved) and an archived one
    never blocks a fresh write. ``kind``/``description``/``schema_tag``/
    ``metadata`` follow the write — a variable is one value, not a merge.
    """
    conflict = (
        "(session_id, name) WHERE archived_at IS NULL AND scope = 'session'"
        if scope == "session"
        else "(account_id, agent_id, name) WHERE archived_at IS NULL AND scope = 'agent'"
    )
    row = await conn.fetchrow(
        f"""
        INSERT INTO context_variables (
            id, account_id, scope, session_id, agent_id, name, kind, description,
            schema_tag, content, content_sha256, content_size_bytes, metadata,
            created_by_type, created_by_ref
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13::jsonb, $14, $15)
        ON CONFLICT {conflict} DO UPDATE SET
            kind = EXCLUDED.kind,
            description = EXCLUDED.description,
            schema_tag = EXCLUDED.schema_tag,
            content = EXCLUDED.content,
            content_sha256 = EXCLUDED.content_sha256,
            content_size_bytes = EXCLUDED.content_size_bytes,
            metadata = EXCLUDED.metadata,
            updated_at = now()
        RETURNING *
        """,
        make_id(CONTEXT_VARIABLE),
        account_id,
        scope,
        session_id,
        agent_id,
        name,
        kind,
        description,
        schema_tag,
        content,
        content_sha256,
        content_size_bytes,
        json.dumps(metadata),
        *actor_columns(),
    )
    assert row is not None
    return _row_to_context_variable(row)


async def get_context_variable(
    conn: asyncpg.Connection[Any],
    variable_id: str,
    *,
    account_id: str,
    include_content: bool = True,
) -> ContextVariable:
    """Fetch one variable by id (operator plane — no readability filter)."""
    select = "*" if include_content else _META_COLUMNS
    row = await conn.fetchrow(
        f"SELECT {select} FROM context_variables WHERE id = $1 AND account_id = $2",
        variable_id,
        account_id,
    )
    if row is None:
        raise NotFoundError(f"context variable {variable_id} not found", detail={"id": variable_id})
    return _row_to_context_variable(row)


async def resolve_readable_variable(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    session_id: str,
    agent_id: str | None,
    name: str,
    include_content: bool = True,
) -> ContextVariable | None:
    """Resolve ``name`` for a reading session, or ``None`` when nothing matches."""
    if include_content:
        select = "v.*"
    else:
        # Qualify every column for the aliased table.
        select = ", ".join(
            c if " AS " in c else f"v.{c.strip()}" for c in _META_COLUMNS.split(", ")
        )
    row = await conn.fetchrow(
        f"""
        SELECT {select} FROM context_variables v
        WHERE {_READABLE_WHERE} AND v.name = $4
        ORDER BY {_READABLE_PRECEDENCE}
        LIMIT 1
        """,
        account_id,
        session_id,
        agent_id,
        name,
    )
    return _row_to_context_variable(row) if row is not None else None


async def list_readable_variables(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    session_id: str,
    agent_id: str | None,
    scope: VariableScope | None = None,
    kind: VariableKind | None = None,
    limit: int = 200,
) -> list[ContextVariable]:
    """Metadata-only listing of everything the session can read, newest first."""
    select = ", ".join(c if " AS " in c else f"v.{c.strip()}" for c in _META_COLUMNS.split(", "))
    args: list[Any] = [account_id, session_id, agent_id]
    where = [_READABLE_WHERE]
    if scope is not None:
        args.append(scope)
        where.append(f"v.scope = ${len(args)}")
    if kind is not None:
        args.append(kind)
        where.append(f"v.kind = ${len(args)}")
    args.append(limit)
    rows = await conn.fetch(
        f"""
        SELECT {select} FROM context_variables v
        WHERE {" AND ".join(where)}
        ORDER BY v.updated_at DESC
        LIMIT ${len(args)}
        """,
        *args,
    )
    return [_row_to_context_variable(r) for r in rows]


async def list_context_variables(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    session_id: str | None = None,
    agent_id: str | None = None,
    scope: VariableScope | None = None,
    kind: VariableKind | None = None,
    limit: int = 50,
    after: str | None = None,
) -> list[ContextVariable]:
    """Operator-plane listing (metadata only), keyset-paginated by id."""
    args: list[Any] = [account_id]
    where = ["account_id = $1", "archived_at IS NULL"]
    for column, value in (
        ("session_id", session_id),
        ("agent_id", agent_id),
        ("scope", scope),
        ("kind", kind),
    ):
        if value is None:
            continue
        args.append(value)
        where.append(f"{column} = ${len(args)}")
    if after is not None:
        args.append(after)
        where.append(f"id < ${len(args)}")
    args.append(limit)
    rows = await conn.fetch(
        f"SELECT {_META_COLUMNS} FROM context_variables WHERE {' AND '.join(where)} "
        f"ORDER BY id DESC LIMIT ${len(args)}",
        *args,
    )
    return [_row_to_context_variable(r) for r in rows]


async def update_context_variable(
    conn: asyncpg.Connection[Any],
    variable_id: str,
    *,
    account_id: str,
    content: str | None = None,
    content_sha256: str | None = None,
    content_size_bytes: int | None = None,
    kind: VariableKind | None = None,
    description: str | None = None,
    schema_tag: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ContextVariable:
    """Partial update by id (operator plane). ``content`` travels with its
    caller-computed sha/size triple or not at all."""
    assert (content is None) == (content_sha256 is None) == (content_size_bytes is None)
    args: list[Any] = []
    fields: list[tuple[str, Any, str | None]] = []
    if content is not None:
        fields.append(("content", content, None))
        fields.append(("content_sha256", content_sha256, None))
        fields.append(("content_size_bytes", content_size_bytes, None))
    if kind is not None:
        fields.append(("kind", kind, None))
    if description is not None:
        fields.append(("description", description, None))
    if schema_tag is not None:
        fields.append(("schema_tag", schema_tag, None))
    if metadata is not None:
        fields.append(("metadata", metadata, "jsonb"))
    sets = _build_set_assignments(fields, args)
    if not sets:
        return await get_context_variable(conn, variable_id, account_id=account_id)
    args.extend([variable_id, account_id])
    row = await conn.fetchrow(
        f"""
        UPDATE context_variables SET {", ".join(sets)}, updated_at = now()
        WHERE id = ${len(args) - 1} AND account_id = ${len(args)} AND archived_at IS NULL
        RETURNING *
        """,
        *args,
    )
    if row is None:
        raise NotFoundError(f"context variable {variable_id} not found", detail={"id": variable_id})
    return _row_to_context_variable(row)


async def archive_context_variable(
    conn: asyncpg.Connection[Any],
    variable_id: str,
    *,
    account_id: str,
) -> ContextVariable:
    row = await _archive_scoped(
        conn,
        table="context_variables",
        id_=variable_id,
        account_id=account_id,
        noun="context variable",
        idempotent=True,
    )
    return _row_to_context_variable(row)


async def insert_context_variable_grants(
    conn: asyncpg.Connection[Any],
    *,
    session_id: str,
    variable_ids: list[str],
) -> None:
    """Grant a (child) session read access to the given variables.

    Idempotent; written inside the child-spawn transaction so a replayed
    spawn re-asserts the same grants.
    """
    if not variable_ids:
        return
    await conn.executemany(
        """
        INSERT INTO context_variable_grants (session_id, context_variable_id)
        VALUES ($1, $2)
        ON CONFLICT DO NOTHING
        """,
        [(session_id, vid) for vid in variable_ids],
    )


async def spill_tool_result_variable(
    executor: asyncpg.Pool[Any] | asyncpg.Connection[Any],
    *,
    session_id: str,
    name: str,
    content: str,
    content_sha256: str,
    content_size_bytes: int,
    tool_call_id: str,
) -> None:
    """Write an oversized tool result into a session-scoped ``kind='spill'``
    variable (the inverted spill — ``sandbox/tool_result_spill.py``).

    Self-contained: derives ``account_id`` from the session row so both
    result sinks (worker pool path, API conn path) can call it without
    threading extra context. Idempotent on the scope key — the spill write
    runs before the tool-result dedup lock, so a losing appender's write
    simply overwrites with identical content. No-ops (leaving no orphan)
    when the session does not exist.
    """
    await executor.execute(
        """
        INSERT INTO context_variables (
            id, account_id, scope, session_id, agent_id, name, kind, description,
            schema_tag, content, content_sha256, content_size_bytes, metadata
        )
        SELECT $1, s.account_id, 'session', s.id, NULL, $3, 'spill',
               'Spilled oversized tool result', NULL, $4, $5, $6,
               jsonb_build_object('tool_call_id', $7::text)
        FROM sessions s WHERE s.id = $2
        ON CONFLICT (session_id, name) WHERE archived_at IS NULL AND scope = 'session'
        DO UPDATE SET
            content = EXCLUDED.content,
            content_sha256 = EXCLUDED.content_sha256,
            content_size_bytes = EXCLUDED.content_size_bytes,
            metadata = EXCLUDED.metadata,
            updated_at = now()
        """,
        make_id(CONTEXT_VARIABLE),
        session_id,
        name,
        content,
        content_sha256,
        content_size_bytes,
        tool_call_id,
    )
