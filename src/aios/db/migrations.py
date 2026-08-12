"""In-process migration helpers.

Replaces ``subprocess.call(["alembic", "upgrade", ...])`` so callers (the
``aios migrate`` CLI command and the e2e test fixtures) skip the
``uv run`` cold-start tax. ``migrations/env.py`` reads ``AIOS_DB_URL``
from the environment, so we temporarily set it for the duration of the
upgrade.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MIGRATION_LOCK_KEY = "aios_schema_migrations"


@contextmanager
def migration_lock(db_url: str) -> Iterator[None]:
    """Serialize production migration runners against one database.

    Fargate replaces the API and worker independently, so both new tasks can
    reach their migration preflight at once.  Hold a session advisory lock on
    a dedicated connection across Alembic and Procrastinate schema work.  A
    crashed task releases the lock with its connection; a waiting peer then
    re-runs the idempotent upgrade before starting its process.
    """
    import psycopg

    with psycopg.connect(db_url, autocommit=True) as conn:
        conn.execute(
            "SELECT pg_advisory_lock(hashtextextended(%s, 0))",
            (_MIGRATION_LOCK_KEY,),
        )
        try:
            yield
        finally:
            conn.execute(
                "SELECT pg_advisory_unlock(hashtextextended(%s, 0))",
                (_MIGRATION_LOCK_KEY,),
            )


def upgrade_to_head(db_url: str) -> None:
    """In-process equivalent of ``alembic upgrade head`` against ``db_url``."""
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(_REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(_REPO_ROOT / "migrations"))
    with mock.patch.dict(os.environ, {"AIOS_DB_URL": db_url}):
        command.upgrade(cfg, "head")


async def apply_procrastinate_schema(db_url: str, *, verbose: bool = False) -> None:
    """Apply procrastinate's schema (if missing) and the aios lock-release
    trigger against ``db_url``. Idempotent — safe on an already-migrated DB.

    ``apply_schema_async`` isn't idempotent (no ``IF NOT EXISTS``), hence the
    ``to_regclass`` guard; the trigger DDL is.

    Pass ``verbose=True`` from CLI contexts to surface user-facing status.
    """
    import asyncpg
    from procrastinate import App, PsycopgConnector

    from aios.db.procrastinate_extensions import LOCK_RELEASE_TRIGGER_DDL

    conn = await asyncpg.connect(db_url)
    try:
        present = await conn.fetchval("SELECT to_regclass('procrastinate_jobs')")
        if present is None:
            if verbose:
                print("applying procrastinate schema...", file=sys.stderr)
            tmp_app = App(connector=PsycopgConnector(conninfo=db_url))
            await tmp_app.open_async()
            try:
                await tmp_app.schema_manager.apply_schema_async()
            finally:
                await tmp_app.close_async()
            if verbose:
                print("procrastinate schema applied", file=sys.stderr)
        elif verbose:
            print("procrastinate schema already present, skipping", file=sys.stderr)
        await conn.execute(LOCK_RELEASE_TRIGGER_DDL)
        if verbose:
            print("aios lock-release trigger ensured", file=sys.stderr)
    finally:
        await conn.close()
