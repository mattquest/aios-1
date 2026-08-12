"""Tests for the database-scoped production migration lock."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from aios.db.migrations import migration_lock


def test_migration_lock_holds_one_session_lock_across_caller_work() -> None:
    connection = MagicMock()
    connection_context = MagicMock()
    connection_context.__enter__.return_value = connection

    with (
        patch("psycopg.connect", return_value=connection_context) as connect,
        migration_lock("postgresql://localhost/aios"),
    ):
        assert connection.execute.call_count == 1

    connect.assert_called_once_with("postgresql://localhost/aios", autocommit=True)
    assert connection.execute.call_args_list[0].args == (
        "SELECT pg_advisory_lock(hashtextextended(%s, 0))",
        ("aios_schema_migrations",),
    )
    assert connection.execute.call_args_list[1].args == (
        "SELECT pg_advisory_unlock(hashtextextended(%s, 0))",
        ("aios_schema_migrations",),
    )


def test_migration_lock_unlocks_when_migration_raises() -> None:
    connection = MagicMock()
    connection_context = MagicMock()
    connection_context.__enter__.return_value = connection

    with patch("psycopg.connect", return_value=connection_context):
        try:
            with migration_lock("postgresql://localhost/aios"):
                raise RuntimeError("migration failed")
        except RuntimeError:
            pass

    assert "pg_advisory_unlock" in connection.execute.call_args_list[-1].args[0]
