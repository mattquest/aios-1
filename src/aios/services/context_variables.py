"""Context-variable service (docs/rlm.md).

Owns pool acquisition and the content hashing/size discipline; the queries
module owns SQL. The ``ctx_*`` tools and the operator router both come
through here so the byte cap and the sha256/size triple are computed in
exactly one place.
"""

from __future__ import annotations

import hashlib
from typing import Any

import asyncpg

from aios.db import queries
from aios.errors import NotFoundError, ValidationError
from aios.models.context_variables import (
    ContextVariable,
    ContextVariableCreate,
    ContextVariableUpdate,
    VariableKind,
    VariableScope,
    check_content_size,
)


def content_digest(content: str) -> tuple[str, int]:
    """``(sha256_hex, utf8_byte_size)`` of ``content``, raising past the cap."""
    size = check_content_size(content)
    return hashlib.sha256(content.encode("utf-8")).hexdigest(), size


async def _check_scope_target(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    scope: VariableScope,
    session_id: str | None,
    agent_id: str | None,
) -> None:
    """Reject a scope target the account does not own (404, same as a miss)."""
    if scope == "session":
        owned = await conn.fetchval(
            "SELECT 1 FROM sessions WHERE id = $1 AND account_id = $2", session_id, account_id
        )
        if owned is None:
            raise NotFoundError(f"session {session_id} not found", detail={"id": session_id})
    else:
        owned = await conn.fetchval(
            "SELECT 1 FROM agents WHERE id = $1 AND account_id = $2", agent_id, account_id
        )
        if owned is None:
            raise NotFoundError(f"agent {agent_id} not found", detail={"id": agent_id})


async def write_variable(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    scope: VariableScope,
    session_id: str | None,
    agent_id: str | None,
    name: str,
    content: str,
    kind: VariableKind = "data",
    description: str = "",
    schema_tag: str | None = None,
    metadata: dict[str, Any] | None = None,
    append: bool = False,
) -> ContextVariable:
    """Create-or-overwrite (or append to) a variable at its scope key.

    The ``ctx_write`` write path. ``append=True`` concatenates onto the
    existing content under the session-row-free upsert (read + rewrite in
    one transaction); the byte cap applies to the combined result.
    """
    async with pool.acquire() as conn, conn.transaction():
        await _check_scope_target(
            conn, account_id=account_id, scope=scope, session_id=session_id, agent_id=agent_id
        )
        if append:
            existing = await conn.fetchrow(
                """
                SELECT content, kind, description, schema_tag FROM context_variables
                WHERE account_id = $1 AND archived_at IS NULL AND scope = $2
                  AND session_id IS NOT DISTINCT FROM $3
                  AND agent_id IS NOT DISTINCT FROM $4
                  AND name = $5
                FOR UPDATE
                """,
                account_id,
                scope,
                session_id,
                agent_id,
                name,
            )
            if existing is not None:
                content = existing["content"] + content
        sha, size = content_digest(content)
        return await queries.upsert_context_variable(
            conn,
            account_id=account_id,
            scope=scope,
            session_id=session_id,
            agent_id=agent_id,
            name=name,
            kind=kind,
            description=description,
            schema_tag=schema_tag,
            content=content,
            content_sha256=sha,
            content_size_bytes=size,
            metadata=metadata or {},
        )


async def create_variable(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    body: ContextVariableCreate,
) -> ContextVariable:
    """Operator-plane create (``POST /v1/context-variables``)."""
    return await write_variable(
        pool,
        account_id=account_id,
        scope=body.scope,
        session_id=body.session_id,
        agent_id=body.agent_id,
        name=body.name,
        content=body.content,
        kind=body.kind,
        description=body.description,
        schema_tag=body.schema_tag,
        metadata=body.metadata,
    )


async def get_variable(
    pool: asyncpg.Pool[Any],
    variable_id: str,
    *,
    account_id: str,
    include_content: bool = True,
) -> ContextVariable:
    async with pool.acquire() as conn:
        return await queries.get_context_variable(
            conn, variable_id, account_id=account_id, include_content=include_content
        )


async def list_variables(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    session_id: str | None = None,
    agent_id: str | None = None,
    scope: VariableScope | None = None,
    kind: VariableKind | None = None,
    limit: int = 50,
    after: str | None = None,
) -> list[ContextVariable]:
    async with pool.acquire() as conn:
        return await queries.list_context_variables(
            conn,
            account_id=account_id,
            session_id=session_id,
            agent_id=agent_id,
            scope=scope,
            kind=kind,
            limit=limit,
            after=after,
        )


async def update_variable(
    pool: asyncpg.Pool[Any],
    variable_id: str,
    *,
    account_id: str,
    body: ContextVariableUpdate,
) -> ContextVariable:
    sha: str | None = None
    size: int | None = None
    if body.content is not None:
        sha, size = content_digest(body.content)
    async with pool.acquire() as conn:
        return await queries.update_context_variable(
            conn,
            variable_id,
            account_id=account_id,
            content=body.content,
            content_sha256=sha,
            content_size_bytes=size,
            kind=body.kind,
            description=body.description,
            schema_tag=body.schema_tag,
            metadata=body.metadata,
        )


async def archive_variable(
    pool: asyncpg.Pool[Any],
    variable_id: str,
    *,
    account_id: str,
) -> ContextVariable:
    async with pool.acquire() as conn:
        return await queries.archive_context_variable(conn, variable_id, account_id=account_id)


async def resolve_readable(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    session_id: str,
    agent_id: str | None,
    name: str,
    include_content: bool = True,
) -> ContextVariable:
    """Resolve ``name`` through a session's read visibility, 404 on miss."""
    async with pool.acquire() as conn:
        variable = await queries.resolve_readable_variable(
            conn,
            account_id=account_id,
            session_id=session_id,
            agent_id=agent_id,
            name=name,
            include_content=include_content,
        )
    if variable is None:
        raise NotFoundError(f"context variable {name!r} not found", detail={"name": name})
    return variable


async def list_readable(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    session_id: str,
    agent_id: str | None,
    scope: VariableScope | None = None,
    kind: VariableKind | None = None,
    limit: int = 200,
) -> list[ContextVariable]:
    async with pool.acquire() as conn:
        return await queries.list_readable_variables(
            conn,
            account_id=account_id,
            session_id=session_id,
            agent_id=agent_id,
            scope=scope,
            kind=kind,
            limit=limit,
        )


def require_valid_scope_args(
    scope: str, session_id: str | None, agent_id: str | None
) -> VariableScope:
    """Shared scope/target argument validation for tool handlers."""
    if scope == "session":
        if session_id is None:
            raise ValidationError("scope 'session' requires a session target")
    elif scope == "agent":
        if agent_id is None:
            raise ValidationError(
                "scope 'agent' requires the session to have an agent", detail={"scope": scope}
            )
    else:
        raise ValidationError(f"unknown scope {scope!r}", detail={"scope": scope})
    return scope  # type: ignore[return-value]
