"""Migration 0161 safely crosses the old TrainIQ migration lineage.

TrainIQ production already has ``events_user_client_message_id_uidx`` from
the former revision 0113, but the reconciled RLM chain reaches the same schema
contract at revision 0161.  The migration must replace a same-named object with
its exact definition instead of failing with DuplicateTable or silently
trusting an invalid/mismatched index.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import asyncpg
import pytest

from tests.conftest import _docker_available
from tests.integration.test_migrations import _alembic_url, _run_alembic

INDEX_NAME = "events_user_client_message_id_uidx"
CANDIDATE_INDEX_NAME = "events_user_client_message_id_uidx_0161_candidate"

CREATE_OLD_LINEAGE_INDEX = f"""
CREATE UNIQUE INDEX {INDEX_NAME}
    ON events (
        account_id,
        session_id,
        ((data->'metadata'->>'client_message_id'))
    )
 WHERE kind = 'message'
   AND role = 'user'
   AND data->'metadata'->>'client_message_id'
       ~ '^[0-9a-f]{{8}}-[0-9a-f]{{4}}-[0-9a-f]{{4}}-[0-9a-f]{{4}}-[0-9a-f]{{12}}$'
"""


@pytest.fixture
def postgres() -> Iterator[object]:
    """Fresh function-scoped Postgres."""
    if not _docker_available():
        pytest.skip("Docker not available")
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("postgres:16-alpine") as pg:
        yield pg


async def _execute(db_url: str, sql: str) -> None:
    conn = await asyncpg.connect(db_url)
    try:
        await conn.execute(sql)
    finally:
        await conn.close()


async def _index_state(db_url: str) -> list[asyncpg.Record]:
    conn = await asyncpg.connect(db_url)
    try:
        rows: list[asyncpg.Record] = await conn.fetch(
            """
            SELECT indexrelid::regclass::text AS name,
                   indisunique,
                   indisvalid,
                   indisready,
                   pg_get_indexdef(indexrelid) AS definition
              FROM pg_index
             WHERE indexrelid IN (
                       to_regclass('public.events_user_client_message_id_uidx'),
                       to_regclass('public.events_user_client_message_id_uidx_0161_candidate')
                   )
             ORDER BY name
            """
        )
        return rows
    finally:
        await conn.close()


@pytest.mark.integration
def test_upgrade_replaces_old_lineage_index(postgres: object) -> None:
    db_url = _alembic_url(postgres)
    to_0160 = _run_alembic(["upgrade", "0160"], db_url)
    assert to_0160.returncode == 0, to_0160.stderr

    asyncio.run(_execute(db_url, CREATE_OLD_LINEAGE_INDEX))

    upgraded = _run_alembic(["upgrade", "0161"], db_url)
    assert upgraded.returncode == 0, upgraded.stderr

    indexes = asyncio.run(_index_state(db_url))
    assert len(indexes) == 1
    index = indexes[0]
    assert index["name"] == INDEX_NAME
    assert index["indisunique"] is True
    assert index["indisvalid"] is True
    assert index["indisready"] is True
    assert "(account_id, session_id" in index["definition"]
    assert "client_message_id" in index["definition"]
    assert CANDIDATE_INDEX_NAME not in index["definition"]


@pytest.mark.integration
def test_upgrade_rejects_mismatched_same_named_index(postgres: object) -> None:
    db_url = _alembic_url(postgres)
    to_0160 = _run_alembic(["upgrade", "0160"], db_url)
    assert to_0160.returncode == 0, to_0160.stderr

    asyncio.run(_execute(db_url, f"CREATE INDEX {INDEX_NAME} ON events (id)"))

    upgraded = _run_alembic(["upgrade", "0161"], db_url)
    assert upgraded.returncode != 0
    assert "mismatched definition" in upgraded.stderr


@pytest.mark.integration
def test_upgrade_recovers_reserved_retry_artifact(postgres: object) -> None:
    db_url = _alembic_url(postgres)
    to_0160 = _run_alembic(["upgrade", "0160"], db_url)
    assert to_0160.returncode == 0, to_0160.stderr

    # Simulate an interrupted attempt that left the reserved candidate name.
    asyncio.run(
        _execute(
            db_url,
            f"CREATE INDEX {CANDIDATE_INDEX_NAME} ON events (id)",
        )
    )

    upgraded = _run_alembic(["upgrade", "0161"], db_url)
    assert upgraded.returncode == 0, upgraded.stderr

    indexes = asyncio.run(_index_state(db_url))
    assert len(indexes) == 1
    assert indexes[0]["name"] == INDEX_NAME
    assert indexes[0]["indisunique"] is True
    assert indexes[0]["indisvalid"] is True
    assert indexes[0]["indisready"] is True


@pytest.mark.integration
def test_upgrade_builds_index_on_fresh_lineage(postgres: object) -> None:
    db_url = _alembic_url(postgres)
    to_0160 = _run_alembic(["upgrade", "0160"], db_url)
    assert to_0160.returncode == 0, to_0160.stderr

    upgraded = _run_alembic(["upgrade", "0161"], db_url)
    assert upgraded.returncode == 0, upgraded.stderr

    indexes = asyncio.run(_index_state(db_url))
    assert len(indexes) == 1
    assert indexes[0]["name"] == INDEX_NAME
    assert indexes[0]["indisunique"] is True
    assert indexes[0]["indisvalid"] is True
    assert indexes[0]["indisready"] is True
