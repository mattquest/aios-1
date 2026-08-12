"""Tests for operator subcommands (api, worker, migrate)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from typer.testing import CliRunner

from aios.cli.app import app

runner = CliRunner()


def test_api_command_passes_proxy_headers_to_uvicorn(monkeypatch):
    """The `api` command must pass proxy_headers=True and forwarded_allow_ips="*"
    to uvicorn.run so that X-Forwarded-For headers are trusted when running
    behind a reverse proxy."""
    monkeypatch.setenv("AIOS_API_KEY", "test")
    monkeypatch.setenv("AIOS_DB_URL", "postgresql://localhost/test")
    # The shared-DB guard fires from a linked worktree (the mandated dev
    # workflow) because db name "test" is not aios_dev_*. Disable it here so we
    # exercise uvicorn wiring, not the guard. (CI runs from a main checkout
    # where the guard is a no-op, so it would never catch this regression.)
    monkeypatch.setattr("aios.cli.commands.dev.is_linked_worktree", lambda: False)

    with patch("uvicorn.run") as mock_run:
        runner.invoke(app, ["api"])

    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs.get("proxy_headers") is True, "uvicorn.run must be called with proxy_headers=True"
    assert kwargs.get("forwarded_allow_ips") == "*", (
        'uvicorn.run must be called with forwarded_allow_ips="*"'
    )


def test_api_command_aborts_before_uvicorn_on_shared_worktree(monkeypatch):
    """`api` from a linked worktree on the shared DB aborts before uvicorn.run."""
    monkeypatch.setenv("AIOS_API_KEY", "test")
    monkeypatch.setenv("AIOS_DB_URL", "postgresql://localhost/aios")
    monkeypatch.delenv("AIOS_ALLOW_SHARED_DB", raising=False)
    monkeypatch.setattr("aios.cli.commands.dev.is_linked_worktree", lambda: True)

    with patch("uvicorn.run") as mock_run:
        result = runner.invoke(app, ["api"])

    mock_run.assert_not_called()
    assert result.exit_code == 1


def test_worker_command_aborts_before_worker_main_on_shared_worktree(monkeypatch):
    """`worker` from a linked worktree on the shared DB aborts before worker_main."""
    monkeypatch.setenv("AIOS_API_KEY", "test")
    monkeypatch.setenv("AIOS_DB_URL", "postgresql://localhost/aios")
    monkeypatch.delenv("AIOS_ALLOW_SHARED_DB", raising=False)
    monkeypatch.setattr("aios.cli.commands.dev.is_linked_worktree", lambda: True)

    with patch("aios.harness.worker.worker_main") as mock_worker_main:
        result = runner.invoke(app, ["worker"])

    mock_worker_main.assert_not_called()
    assert result.exit_code == 1


def test_migrate_configures_logging_before_running_migrations(monkeypatch):
    """`migrate` must call configure_logging BEFORE upgrade_to_head so that
    migration-emitted audit records (e.g. the 0130 auto-disable WARNING) are
    visible on the prod path. Without this, `import alembic`'s NullHandler on
    the 'alembic' parent logger satisfies the handler-search and shadows
    logging.lastResort, silently swallowing every migration warning (#1678 F1).
    Assert both that configure_logging is called and that it precedes the
    migration run (order matters — a migration warning emitted before logging
    is configured would still be lost)."""
    monkeypatch.setenv("AIOS_API_KEY", "test")
    monkeypatch.setenv("AIOS_DB_URL", "postgresql://localhost/test")

    # Attach every collaborator to one parent so mock_calls records global order.
    parent = Mock()
    parent.get_settings.return_value = SimpleNamespace(
        db_url="postgresql://localhost/test", log_level="INFO"
    )
    parent.apply_procrastinate_schema = AsyncMock()
    parent.migration_lock = MagicMock()
    parent.migration_lock.return_value.__enter__.return_value = None
    parent.migration_lock.return_value.__exit__.return_value = None

    with (
        patch("aios.config.get_settings", parent.get_settings),
        patch("aios.logging.configure_logging", parent.configure_logging),
        patch("aios.db.migrations.migration_lock", parent.migration_lock),
        patch("aios.db.migrations.upgrade_to_head", parent.upgrade_to_head),
        patch("aios.db.migrations.apply_procrastinate_schema", parent.apply_procrastinate_schema),
    ):
        result = runner.invoke(app, ["migrate"])

    assert result.exit_code == 0, result.output
    parent.configure_logging.assert_called_once_with("INFO")
    parent.migration_lock.assert_called_once_with("postgresql://localhost/test")
    parent.upgrade_to_head.assert_called_once_with("postgresql://localhost/test")
    # configure_logging must run before the migration.
    called = [c[0] for c in parent.mock_calls]
    assert called.index("configure_logging") < called.index("upgrade_to_head"), (
        "configure_logging must be called before upgrade_to_head"
    )
