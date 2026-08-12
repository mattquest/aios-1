"""Account management endpoints."""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import APIRouter, Header, Query, status

from aios.api.deps import AuthDep, PoolDep, _extract_bearer_token
from aios.config import get_settings
from aios.db import queries
from aios.errors import NotFoundError, UnauthorizedError
from aios.logging import get_logger
from aios.models.accounts import (
    Account,
    AccountKeySummary,
    AccountPurgeMode,
    AccountUsage,
    BootstrapRequest,
    BootstrapResponse,
    MintAccountRequest,
    MintAccountResponse,
    MintKeyRequest,
    MintKeyResponse,
    UpdateAccountRequest,
)
from aios.services import accounts as service

router = APIRouter(prefix="/v1/accounts", tags=["accounts"])

log = get_logger("aios.api.accounts")


@router.post(
    "/bootstrap",
    operation_id="bootstrap_root_account",
    status_code=status.HTTP_201_CREATED,
    # Operator-side ceremony; not part of the agent-facing management plane.
    openapi_extra={"x-codegen": {"targets": ["sdk"]}},
)
async def bootstrap(
    body: BootstrapRequest,
    pool: PoolDep,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> BootstrapResponse:
    """Mint the root account and its first API key.

    T2 decision (#1463): ``bootstrap`` is deliberately retained over
    ``create_root_account`` — the one-time, token-gated, self-closing
    operator ceremony semantic is load-bearing (the endpoint behaves like it
    doesn't exist once a root is in place), and a generic ``create`` would
    obscure that. Documented exception; not part of the agent-facing plane.

    Gated by ``AIOS_BOOTSTRAP_TOKEN`` (env var). When that env is unset
    or empty, the endpoint is 401 regardless of header value.

    Root-exists check fires before the token check so a probe with no/
    wrong token can't distinguish "no bootstrap" (404) from "wrong token
    but bootstrap is still open" (401). Once a root is in place the
    endpoint behaves like it doesn't exist at all.

    ``plaintext_key`` in the response is the only time the operator key
    is returned in plaintext.
    """
    async with pool.acquire() as conn:
        if await queries.has_active_root_account(conn):
            raise NotFoundError("bootstrap endpoint closed: root account already exists")
    token = _extract_bearer_token(authorization)
    expected_secret = get_settings().bootstrap_token
    expected = expected_secret.get_secret_value() if expected_secret is not None else ""
    if not expected or not secrets.compare_digest(token, expected):
        raise UnauthorizedError("invalid bootstrap token")
    response = await service.bootstrap_root(pool, display_name=body.display_name)
    log.info(
        "account.operation",
        actor_account_id=None,
        actor_key_id=None,
        target_account_id=response.account_id,
        action="account.bootstrap",
        outcome="success",
    )
    return response


# ─── management plane — caller-or-child scoped (#367 PR 7) ──────────────────


@router.get("/me", operation_id="get_my_account")
async def get_my_account(pool: PoolDep, auth: AuthDep) -> Account:
    """Return the account the bearer token resolved to."""
    account_id, _key_id, _can_mint = auth
    async with pool.acquire() as conn:
        row = await queries.get_account(conn, account_id)
    if row is None:
        # The auth dep just resolved this id; a missing row here means
        # someone archived the account between auth and route. Surface as
        # 404 rather than 500 — operator-side cleanup is the right next step.
        raise NotFoundError(f"account {account_id} not found", detail={"id": account_id})
    return row


@router.get("/children", operation_id="list_my_children")
async def list_my_children(pool: PoolDep, auth: AuthDep) -> list[Account]:
    """List direct child accounts under the caller."""
    account_id, _key_id, _can_mint = auth
    return await service.list_children(pool, parent_account_id=account_id)


@router.post(
    "/children",
    operation_id="create_child_account",
    status_code=status.HTTP_201_CREATED,
)
async def create_child(
    body: MintAccountRequest, pool: PoolDep, auth: AuthDep
) -> MintAccountResponse:
    """Create a direct child account under the caller and its first API key.

    Requires the caller's ``can_mint_children`` to be true. Returns the
    new account id, the first key's id, and the plaintext bearer (the
    only time that plaintext is recoverable).
    """
    account_id, key_id, can_mint = auth
    response = await service.mint_child(
        pool,
        caller_account_id=account_id,
        caller_can_mint_children=can_mint,
        display_name=body.display_name,
        can_mint_children=body.can_mint_children,
    )
    log.info(
        "account.operation",
        actor_account_id=account_id,
        actor_key_id=key_id,
        target_account_id=response.account_id,
        action="account.mint_child",
        outcome="success",
    )
    return response


@router.get("/by-path", operation_id="resolve_account_by_path")
async def resolve_by_path(path: str, pool: PoolDep, auth: AuthDep) -> Account:
    """Resolve a slash-separated path of ``display_name`` segments under the
    caller's account.

    Examples:
    * ``?path=`` → the caller's account row.
    * ``?path=tenant-a`` → the direct child named ``tenant-a``.
    * ``?path=tenant-a/team-1`` → the grandchild ``team-1`` under
      ``tenant-a``.

    Paths that walk outside the caller's subtree 404 (no existence leak).
    """
    account_id, _key_id, _can_mint = auth
    return await service.resolve_by_path(pool, caller_account_id=account_id, path=path)


@router.get("/{target_id}", operation_id="get_account")
async def get_account(target_id: str, pool: PoolDep, auth: AuthDep) -> Account:
    """Read a specific account that's the caller or a direct child."""
    account_id, _key_id, _can_mint = auth
    return await service.get_account_in_scope(pool, target_id, caller_account_id=account_id)


# T2 verb-normalization decision (#1463): accounts deliberately keep PATCH
# here rather than moving to the project-wide PUT-for-update convention.
# ``update_account`` is a genuine *partial* update — omitted fields are
# preserved and "all fields null" is a valid no-op — which is exactly
# PATCH's RFC 5789 semantics. PUT would imply whole-resource replacement,
# which is wrong for the account row (it carries operator-managed columns
# the management plane must never clobber). This is the one documented
# account-only exception to the UPDATE=PUT rule.
@router.patch("/{target_id}", operation_id="update_account")
async def update_account(
    target_id: str, body: UpdateAccountRequest, pool: PoolDep, auth: AuthDep
) -> Account:
    """Partial-update ``display_name`` / ``can_mint_children`` on a
    caller-or-direct-child account.

    Omitted fields are preserved. Both fields null is a valid no-op
    that returns the current row.

    Uses PATCH (not the project-wide PUT-for-update convention) as a
    deliberate, account-only choice: this is a true partial update, so
    PATCH's partial-replace semantics are correct and PUT's
    whole-resource-replace semantics would be wrong. See #1463.
    """
    account_id, key_id, can_mint = auth
    updated = await service.update_account(
        pool,
        target_account_id=target_id,
        caller_account_id=account_id,
        caller_can_mint_children=can_mint,
        display_name=body.display_name,
        can_mint_children=body.can_mint_children,
        config=body.config,
    )
    log.info(
        "account.operation",
        actor_account_id=account_id,
        actor_key_id=key_id,
        target_account_id=target_id,
        action="account.update",
        outcome="success",
    )
    return updated


@router.delete(
    "/{target_id}",
    operation_id="archive_account",
)
async def archive_account(target_id: str, pool: PoolDep, auth: AuthDep) -> Account:
    """Archive a direct child of the caller.

    Refuses if the child has non-archived children of its own. Returns
    the archived account row (idempotent — calling twice returns the
    same row with the original ``archived_at``).
    """
    account_id, key_id, _can_mint = auth
    archived = await service.archive_child(
        pool, target_account_id=target_id, caller_account_id=account_id
    )
    log.info(
        "account.operation",
        actor_account_id=account_id,
        actor_key_id=key_id,
        target_account_id=target_id,
        action="account.archive",
        outcome="success",
    )
    return archived


@router.post(
    "/{target_id}/purge",
    operation_id="purge_account",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def purge_account(
    target_id: str,
    pool: PoolDep,
    auth: AuthDep,
    mode: Annotated[AccountPurgeMode, Query()] = "strict",
) -> None:
    """Hard-delete a direct child that has already been soft-archived.

    T2 decision (#1463): ``purge`` is deliberately retained as the
    explicitly-named two-step hard-delete ceremony (it refuses unless the
    account is already archived, childless, and resourceless). It is *not*
    renamed to ``delete_account`` — under the T2 convention the bare DELETE
    verb is the soft-archive, and ``/purge`` is the consistent name for the
    irreversible hard-delete across families. accounts is the baseline here.

    Two-step ceremony: \
    1. ``DELETE /v1/accounts/{id}`` soft-archives (sets ``archived_at``).\
    2. ``POST /v1/accounts/{id}/purge`` hard-deletes the row.

    ``mode=strict`` is the legacy default: it refuses populated accounts and
    allows any direct parent. ``mode=cascade`` is root-only and durably erases
    a childless direct child's database resources and canonical host artifacts;
    exact retries resume a stored cleanup receipt after partial failure.
    """
    account_id, key_id, _can_mint = auth
    await service.purge_account(
        pool,
        target_account_id=target_id,
        caller_account_id=account_id,
        mode=mode,
    )
    log.info(
        "account.operation",
        actor_account_id=account_id,
        actor_key_id=key_id,
        target_account_id=target_id,
        action=f"account.purge.{mode}",
        outcome="success",
    )


@router.get("/{target_id}/usage", operation_id="get_account_usage")
async def get_account_usage(target_id: str, pool: PoolDep, auth: AuthDep) -> AccountUsage:
    """Per-resource non-archived counts for a caller-or-direct-child account."""
    account_id, _key_id, _can_mint = auth
    return await service.get_usage(pool, target_account_id=target_id, caller_account_id=account_id)


@router.post(
    "/{target_id}/keys",
    operation_id="mint_account_key",
    status_code=status.HTTP_201_CREATED,
)
async def mint_account_key(
    target_id: str, body: MintKeyRequest, pool: PoolDep, auth: AuthDep
) -> MintKeyResponse:
    """Mint an additional API key on a caller-or-child account."""
    account_id, actor_key_id, _can_mint = auth
    response = await service.mint_key(
        pool,
        target_account_id=target_id,
        caller_account_id=account_id,
        label=body.label,
    )
    log.info(
        "account.operation",
        actor_account_id=account_id,
        actor_key_id=actor_key_id,
        target_account_id=target_id,
        target_key_id=response.key_id,
        action="account.mint_key",
        outcome="success",
    )
    return response


@router.get(
    "/{target_id}/keys",
    operation_id="list_account_keys",
)
async def list_account_keys(
    target_id: str, pool: PoolDep, auth: AuthDep
) -> list[AccountKeySummary]:
    """List key summaries (sans hash) for a caller-or-child account."""
    account_id, _key_id, _can_mint = auth
    return await service.list_keys(pool, target_account_id=target_id, caller_account_id=account_id)


@router.delete(
    "/{target_id}/keys/{key_id}",
    operation_id="revoke_account_key",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def revoke_account_key(target_id: str, key_id: str, pool: PoolDep, auth: AuthDep) -> None:
    """Revoke an API key on a caller-or-child account. Idempotent.

    T2 decision (#1463): ``revoke`` is deliberately retained over
    ``delete_account_key`` — "revoke a key" is the load-bearing security-
    domain idiom and conveys the irreversible credential-invalidation
    semantic more precisely than a generic delete. Documented exception.
    """
    account_id, actor_key_id, _can_mint = auth
    await service.revoke_key(
        pool,
        target_account_id=target_id,
        key_id=key_id,
        caller_account_id=account_id,
    )
    log.info(
        "account.operation",
        actor_account_id=account_id,
        actor_key_id=actor_key_id,
        target_account_id=target_id,
        target_key_id=key_id,
        action="account.revoke_key",
        outcome="success",
    )
