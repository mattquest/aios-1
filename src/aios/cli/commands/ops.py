"""Operator subcommands: ``api``, ``worker``, ``migrate``.

These are lifted almost verbatim from the old ``__main__.py`` implementation.
They start long-running processes (uvicorn, procrastinate worker) or run
migrations — they do NOT talk to the HTTP API.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import asyncpg
import typer

from aios.crypto.vault import CryptoBox, EncryptedBlob
from aios.errors import CryptoDecryptError


def _run_api() -> int:
    import uvicorn

    from aios.cli.commands.dev import assert_not_shared_in_worktree
    from aios.config import get_settings

    # Abort before any DB connection if a linked worktree is about to use the
    # shared dev DB (#349). No-op in the main checkout / when bootstrapped.
    assert_not_shared_in_worktree()

    settings = get_settings()
    uvicorn.run(
        "aios.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        log_config=None,  # we configure structlog ourselves
        # uvicorn's access log is redundant with our own RequestLoggingMiddleware
        # (one api.request line per request) AND, with log_config=None, its
        # `uvicorn.access` logger propagates to the root %(message)s handler and
        # would log the raw request line — leaking the per-trigger ingest bearer
        # token in `POST /v1/triggers/ingest/<token>` that the middleware/error
        # log sites redact. Disable it so the redaction is not defeated.
        access_log=False,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
    return 0


def _run_worker() -> int:
    from aios.cli.commands.dev import assert_not_shared_in_worktree
    from aios.harness.worker import worker_main
    from aios.logging import get_logger

    # Abort before spawning any container / DB connection if a linked worktree
    # is about to use the shared dev DB (#349). No-op in the main checkout.
    assert_not_shared_in_worktree()

    try:
        asyncio.run(worker_main())
    except KeyboardInterrupt:
        pass
    except SystemExit:
        raise
    except BaseException:
        get_logger("aios.worker").exception("worker.unexpected_exit")
        raise
    return 0


@dataclass(frozen=True, slots=True)
class _EncryptedColumn:
    table: str
    ciphertext: str
    nonce: str
    where: str
    account_expr: str = "account_id"


_REKEY_COLUMNS = (
    _EncryptedColumn("vault_credentials", "ciphertext", "nonce", "archived_at IS NULL"),
    _EncryptedColumn(
        "connections", "secrets_ciphertext", "secrets_nonce", "secrets_ciphertext IS NOT NULL"
    ),
    _EncryptedColumn("session_github_repositories", "ciphertext", "nonce", "TRUE"),
    _EncryptedColumn(
        "accounts",
        "placeholder_salt_ciphertext",
        "placeholder_salt_nonce",
        "placeholder_salt_ciphertext IS NOT NULL",
        "id",
    ),
    # In-flight Connect OAuth flow state (#818 vault-oauth): the blob carries the
    # PKCE code_verifier + client_secret, encrypted with the same per-account
    # subkey as vault_credentials. Rows are short-lived (deleted on complete,
    # pruned at ~10min) and ciphertext is NOT NULL — so "TRUE" rekeys every
    # live row. Omitting it left an active flow undecryptable across a rotation.
    _EncryptedColumn("oauth_flows", "ciphertext", "nonce", "TRUE"),
    # Per-account model-provider config (#model_providers): the encrypted
    # api_key, keyed by the owning account's subkey. archived rows have their
    # ciphertext zeroed (not NULL), so the same "archived_at IS NULL" filter
    # vault_credentials uses keeps a rekey from re-encrypting scrubbed bytes.
    _EncryptedColumn("model_providers", "ciphertext", "nonce", "archived_at IS NULL"),
)


def _decrypt_with_current_then_previous(
    current: CryptoBox, previous: CryptoBox | None, account_id: str, blob: EncryptedBlob
) -> str:
    for box in (current, previous):
        if box is None:
            continue
        try:
            return box.derive_account_subkey(account_id).decrypt(blob)
        except CryptoDecryptError:
            if box is current and previous is not None:
                continue
            raise
    raise RuntimeError("unreachable")


async def _run_rekey_async() -> int:
    from aios.config import get_settings
    from aios.db.pool import create_pool

    settings = get_settings()
    previous_secret = settings.vault_key_previous
    if previous_secret is None:
        raise RuntimeError("AIOS_VAULT_KEY_PREVIOUS must be set while running `aios rekey`")
    current = CryptoBox.from_base64(settings.vault_key.get_secret_value())
    previous = CryptoBox.from_base64(
        previous_secret.get_secret_value(), env_name="AIOS_VAULT_KEY_PREVIOUS"
    )
    pool = await create_pool(settings.db_url, min_size=1, max_size=1)
    total = 0
    try:
        async with pool.acquire() as conn, conn.transaction():
            for col in _REKEY_COLUMNS:
                total += await _rekey_column(conn, col, current, previous)
    finally:
        await pool.close()
    typer.echo(f"aios rekey re-encrypted {total} encrypted rows with AIOS_VAULT_KEY")
    return 0


async def _rekey_column(
    conn: asyncpg.Connection[Any], col: _EncryptedColumn, current: CryptoBox, previous: CryptoBox
) -> int:
    rows = await conn.fetch(
        f"SELECT id, {col.account_expr} AS account_id, {col.ciphertext} AS ct, {col.nonce} AS nn "
        f"FROM {col.table} WHERE {col.where}"
    )
    for row in rows:
        plaintext = _decrypt_with_current_then_previous(
            current, previous, row["account_id"], EncryptedBlob(row["ct"], row["nn"])
        )
        blob = current.derive_account_subkey(row["account_id"]).encrypt(plaintext)
        await conn.execute(
            f"UPDATE {col.table} SET {col.ciphertext} = $1, {col.nonce} = $2 WHERE id = $3",
            blob.ciphertext,
            blob.nonce,
            row["id"],
        )
    return len(rows)


def _run_rekey() -> int:
    return asyncio.run(_run_rekey_async())


def _run_migrate() -> int:
    from aios.config import get_settings
    from aios.db.migrations import apply_procrastinate_schema, migration_lock, upgrade_to_head
    from aios.logging import configure_logging

    settings = get_settings()
    # Configure logging before running migrations so migration-emitted audit
    # records are visible. Without this, `import alembic` has attached a
    # NullHandler to the 'alembic' parent logger, which satisfies the
    # handler-search and thus shadows logging.lastResort — so a migration's
    # logger.warning() (e.g. 0130's auto-disable audit) is swallowed silently
    # on the prod `aios migrate` path. logging.py's own docstring prescribes
    # calling configure_logging at the migrate command's start.
    configure_logging(settings.log_level)
    db_url = settings.db_url
    with migration_lock(db_url):
        upgrade_to_head(db_url)
        asyncio.run(apply_procrastinate_schema(db_url, verbose=True))
    return 0


def register(app: typer.Typer) -> None:
    """Attach the operator commands to the root app."""

    @app.command("api", help="Run the aios HTTP API server (uvicorn).")
    def api() -> None:
        raise typer.Exit(_run_api())

    @app.command("worker", help="Run the aios worker (procrastinate).")
    def worker() -> None:
        raise typer.Exit(_run_worker())

    @app.command(
        "migrate",
        help="Apply alembic migrations and the procrastinate schema if missing.",
    )
    def migrate() -> None:
        raise typer.Exit(_run_migrate())

    @app.command("rekey", help="Re-encrypt encrypted rows after AIOS_VAULT_KEY rotation.")
    def rekey() -> None:
        raise typer.Exit(_run_rekey())
