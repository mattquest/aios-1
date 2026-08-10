"""Context-variable CRUD endpoints (RLM metadata plane, docs/rlm.md)."""

from __future__ import annotations

from fastapi import APIRouter, status

from aios.api.deps import AccountIdDep, PoolDep
from aios.models.common import ListResponse
from aios.models.context_variables import (
    ContextVariable,
    ContextVariableCreate,
    ContextVariableUpdate,
    VariableKind,
    VariableScope,
)
from aios.models.pagination import PageLimit, page_cursor, resolve_page_limit
from aios.services import context_variables as service

router = APIRouter(prefix="/v1/context-variables", tags=["context-variables"])


@router.post("", operation_id="create_context_variable", status_code=status.HTTP_201_CREATED)
async def create(
    body: ContextVariableCreate, pool: PoolDep, account_id: AccountIdDep
) -> ContextVariable:
    """Create (or overwrite) a variable at its scope key.

    Exactly one of ``session_id`` / ``agent_id`` must be set, matching
    ``scope``. Writing to an existing live ``(scope target, name)`` handle
    overwrites it in place — a variable is one value, not a merge. Content
    is capped at 8 MiB; the response echoes the row including content.
    """
    return await service.create_variable(pool, account_id=account_id, body=body)


@router.get("", operation_id="list_context_variables")
async def list_(
    pool: PoolDep,
    account_id: AccountIdDep,
    cursor: str | None = None,
    session_id: str | None = None,
    agent_id: str | None = None,
    scope: VariableScope | None = None,
    kind: VariableKind | None = None,
    after: str | None = None,
    limit: PageLimit = None,
) -> ListResponse[ContextVariable]:
    """List context variables, newest first, keyset-paginated. Metadata only —
    ``content`` is ``None`` on every row regardless of size; fetch a single
    variable to read its content.

    First page: filters (``?session_id=``, ``?agent_id=``, ``?scope=``,
    ``?kind=``), ``?after=`` (keyset bound: rows with id before it), and
    ``?limit=``. Subsequent pages: ``?cursor=<next_cursor>`` alone.
    """
    st = page_cursor(
        cursor,
        {
            "session_id": session_id,
            "agent_id": agent_id,
            "scope": scope,
            "kind": kind,
            "after": after,
            "limit": limit,
        },
    )
    page_limit = resolve_page_limit(st, limit)
    if st is not None:
        after = str(st.cursor)
        session_id = st.filters.get("session_id")
        agent_id = st.filters.get("agent_id")
        scope = st.filters.get("scope")
        kind = st.filters.get("kind")
    items = await service.list_variables(
        pool,
        account_id=account_id,
        session_id=session_id,
        agent_id=agent_id,
        scope=scope,
        kind=kind,
        limit=page_limit + 1,
        after=after,
    )
    return ListResponse[ContextVariable].paginate(
        items,
        page_limit,
        cursor=lambda x: x.id,
        filters={"session_id": session_id, "agent_id": agent_id, "scope": scope, "kind": kind},
    )


@router.get("/{variable_id}", operation_id="get_context_variable")
async def get(variable_id: str, pool: PoolDep, account_id: AccountIdDep) -> ContextVariable:
    """Fetch one variable by id, including its content."""
    return await service.get_variable(
        pool, variable_id, account_id=account_id, include_content=True
    )


@router.post("/{variable_id}", operation_id="update_context_variable")
async def update(
    variable_id: str, body: ContextVariableUpdate, pool: PoolDep, account_id: AccountIdDep
) -> ContextVariable:
    """Partially update a variable by id. Omitted fields are preserved;
    ``content`` (when sent) replaces the stored value and recomputes the
    sha256/size pair."""
    return await service.update_variable(pool, variable_id, account_id=account_id, body=body)


@router.delete(
    "/{variable_id}",
    operation_id="archive_context_variable",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def archive(variable_id: str, pool: PoolDep, account_id: AccountIdDep) -> None:
    """Archive a context variable: sets ``archived_at`` and hides it from lists.

    The row persists for audit; the ``(scope target, name)`` handle is freed
    for a fresh write. Idempotent — archiving an already-archived variable
    succeeds.
    """
    await service.archive_variable(pool, variable_id, account_id=account_id)
