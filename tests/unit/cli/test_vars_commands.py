"""Tests for ``aios vars ...`` (context variables) via the typer app."""

from __future__ import annotations

from typing import Any

import httpx
from typer.testing import CliRunner

from aios.cli.app import app

runner = CliRunner()

_EMPTY_PAGE = {"data": [], "has_more": False, "next_cursor": None}


def _variable(**overrides: Any) -> dict[str, Any]:
    return {
        "id": "cv_1",
        "type": "context_variable",
        "scope": "session",
        "session_id": "sess_1",
        "agent_id": None,
        "name": "notes",
        "kind": "data",
        "description": "",
        "schema_tag": None,
        "content": None,
        "content_sha256": "0" * 64,
        "content_size_bytes": 5,
        "metadata": {},
        "created_at": "2024-01-01T00:00:00+00:00",
        "updated_at": "2024-01-01T00:00:00+00:00",
        "archived_at": None,
        **overrides,
    }


def test_list_sends_filters_and_pagination(mocked_cli):
    mocked_cli.queue_response(httpx.Response(200, json=_EMPTY_PAGE))
    result = runner.invoke(
        app,
        [
            "vars",
            "list",
            "--session",
            "sess_1",
            "--agent",
            "agt_1",
            "--scope",
            "session",
            "--kind",
            "spill",
            "--limit",
            "42",
            "--after",
            "cv_9",
        ],
    )
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.method == "GET"
    assert mocked_cli.captured.path == "/v1/context-variables"
    assert mocked_cli.captured.query == {
        "session_id": ["sess_1"],
        "agent_id": ["agt_1"],
        "scope": ["session"],
        "kind": ["spill"],
        "after": ["cv_9"],
        "limit": ["42"],
    }


def test_list_omits_unset_filters(mocked_cli):
    mocked_cli.queue_response(httpx.Response(200, json={"data": [_variable()], "has_more": False}))
    result = runner.invoke(app, ["vars", "list"])
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.query == {"limit": ["50"]}


def test_get(mocked_cli):
    mocked_cli.queue_response(httpx.Response(200, json=_variable(content="hello")))
    result = runner.invoke(app, ["vars", "get", "cv_1"])
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.method == "GET"
    assert mocked_cli.captured.path == "/v1/context-variables/cv_1"


def test_create_posts_payload(mocked_cli):
    payload = {
        "name": "notes",
        "scope": "session",
        "session_id": "sess_1",
        "content": "hello",
    }
    mocked_cli.queue_response(httpx.Response(201, json=_variable(content="hello")))
    result = runner.invoke(
        app,
        [
            "vars",
            "create",
            "--data",
            '{"name": "notes", "scope": "session", "session_id": "sess_1", "content": "hello"}',
        ],
    )
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.method == "POST"
    assert mocked_cli.captured.path == "/v1/context-variables"
    assert mocked_cli.captured.body == payload


def test_create_from_file(mocked_cli, tmp_path):
    body_file = tmp_path / "var.json"
    body_file.write_text('{"name": "x", "scope": "agent", "agent_id": "agt_1", "content": "c"}')
    mocked_cli.queue_response(httpx.Response(201, json=_variable()))
    result = runner.invoke(app, ["vars", "create", "--file", str(body_file)])
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.method == "POST"
    assert mocked_cli.captured.path == "/v1/context-variables"
    assert mocked_cli.captured.body == {
        "name": "x",
        "scope": "agent",
        "agent_id": "agt_1",
        "content": "c",
    }


def test_update_posts_to_id(mocked_cli):
    mocked_cli.queue_response(httpx.Response(200, json=_variable(content="new")))
    result = runner.invoke(app, ["vars", "update", "cv_1", "--data", '{"content": "new"}'])
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.method == "POST"
    assert mocked_cli.captured.path == "/v1/context-variables/cv_1"
    assert mocked_cli.captured.body == {"content": "new"}


def test_delete_is_archive(mocked_cli):
    mocked_cli.queue_response(httpx.Response(204))
    result = runner.invoke(app, ["vars", "delete", "cv_1"])
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.method == "DELETE"
    assert mocked_cli.captured.path == "/v1/context-variables/cv_1"
    assert "archived" in result.output
    assert "cv_1" in result.output
