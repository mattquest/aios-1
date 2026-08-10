"""Unit tests for the ``ctx_*`` builtins (docs/rlm.md).

DB-free: the registry shape (transports, closed schemas, ctx_eval's sandbox
execution class) and each handler's branch logic are pinned by patching the
pool/session/service seams. ``ctx_eval``'s sandbox exec path needs Docker and
lives in the e2e tier — here only its argument validation (exactly one of
``code``/``helper``, helper-kind check) is covered, all of which runs before
sandbox acquisition.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

import aios.tools  # noqa: F401 — registers the builtins
from aios.models.context_variables import ContextVariable, VariableKind, VariableScope
from aios.tools import ctx as ctx_tools
from aios.tools.invoke import ToolBail, invoke_builtin

_SESSION = "ses_x"
_ACCOUNT = "acc_x"


def _variable(
    name: str = "notes",
    *,
    content: str = "",
    scope: VariableScope = "session",
    kind: VariableKind = "data",
    description: str = "",
    schema_tag: str | None = None,
) -> ContextVariable:
    now = datetime(2026, 8, 1, 12, 0, 0, tzinfo=UTC)
    return ContextVariable(
        id="ctxvar_1",
        scope=scope,
        session_id=_SESSION if scope == "session" else None,
        agent_id="agt_1" if scope == "agent" else None,
        name=name,
        kind=kind,
        description=description,
        schema_tag=schema_tag,
        content=content,
        content_sha256="0" * 64,
        content_size_bytes=len(content.encode("utf-8")),
        metadata={},
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
def wired(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Patch the pool + session seams; return the mutable session stub."""
    pool = object()
    session = SimpleNamespace(agent_id="agt_1", environment_id="env_1")
    monkeypatch.setattr("aios.harness.runtime.require_pool", lambda: pool)
    monkeypatch.setattr(
        "aios.services.sessions.load_session_account_id", AsyncMock(return_value=_ACCOUNT)
    )
    monkeypatch.setattr("aios.services.sessions.get_session_basic", AsyncMock(return_value=session))
    return SimpleNamespace(pool=pool, session=session)


# ─── registration shape ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("name", "transport"),
    [
        ("ctx_list", "both"),
        ("ctx_peek", "both"),
        ("ctx_grep", "both"),
        ("ctx_write", "both"),
        ("ctx_eval", "agent_tool"),
    ],
)
def test_registration_shape(name: str, transport: str) -> None:
    """Every ctx builtin registers with a closed schema; ctx_eval is model-only
    and executes in the sandbox class (its handler needs the container)."""
    from aios.tools.registry import registry

    definition = registry.get(name)
    assert definition.transport == transport
    assert definition.parameters_schema.get("additionalProperties") is False
    if name == "ctx_eval":
        assert definition.executes == "sandbox"


async def test_schema_gate_rejects_smuggled_key(wired: SimpleNamespace) -> None:
    """A smuggled extra key is rejected by the schema before the handler runs."""
    with pytest.raises(ToolBail):
        await invoke_builtin(_SESSION, "ctx_list", {"account_id": "acc_evil"})


# ─── ctx_list ────────────────────────────────────────────────────────────────


async def test_ctx_list_returns_metadata_entries(
    wired: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    var = _variable("digest", scope="agent", kind="digest", description="d", schema_tag="v1")
    list_mock = AsyncMock(return_value=[var])
    monkeypatch.setattr("aios.services.context_variables.list_readable", list_mock)

    out = await ctx_tools.ctx_list_handler(_SESSION, {"scope": "agent"})

    assert out == {
        "variables": [
            {
                "name": "digest",
                "scope": "agent",
                "kind": "digest",
                "description": "d",
                "schema_tag": "v1",
                "size_bytes": 0,
                "updated_at": var.updated_at.isoformat(),
            }
        ]
    }
    assert list_mock.await_args is not None
    kwargs = list_mock.await_args.kwargs
    assert kwargs["session_id"] == _SESSION
    assert kwargs["agent_id"] == "agt_1"
    assert kwargs["scope"] == "agent"
    assert kwargs["account_id"] == _ACCOUNT


# ─── ctx_peek ────────────────────────────────────────────────────────────────


def _peek_cap(monkeypatch: pytest.MonkeyPatch, cap: int) -> None:
    monkeypatch.setattr(
        "aios.tools.ctx.get_settings", lambda: SimpleNamespace(ctx_peek_max_bytes=cap)
    )


async def test_ctx_peek_slices_by_byte_offset(
    wired: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _peek_cap(monkeypatch, 1024)
    monkeypatch.setattr(
        "aios.services.context_variables.resolve_readable",
        AsyncMock(return_value=_variable(content="hello world")),
    )

    out = await ctx_tools.ctx_peek_handler(_SESSION, {"name": "notes", "offset": 6, "length": 5})

    assert out["content"] == "world"
    assert out["offset"] == 6
    assert out["length"] == 5
    assert out["total_size"] == 11
    assert out["truncated"] is False  # 6 + 5 == 11: exactly the end


async def test_ctx_peek_truncated_when_content_remains(
    wired: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _peek_cap(monkeypatch, 1024)
    monkeypatch.setattr(
        "aios.services.context_variables.resolve_readable",
        AsyncMock(return_value=_variable(content="hello world")),
    )

    out = await ctx_tools.ctx_peek_handler(_SESSION, {"name": "notes", "length": 5})

    assert out["content"] == "hello"
    assert out["truncated"] is True  # 0 + 5 < 11: bytes remain


async def test_ctx_peek_caps_length_at_settings(
    wired: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A requested length over ``ctx_peek_max_bytes`` is silently capped; an
    omitted length defaults to the cap."""
    _peek_cap(monkeypatch, 4)
    monkeypatch.setattr(
        "aios.services.context_variables.resolve_readable",
        AsyncMock(return_value=_variable(content="hello world")),
    )

    capped = await ctx_tools.ctx_peek_handler(_SESSION, {"name": "notes", "length": 100})
    assert capped["content"] == "hell"
    assert capped["length"] == 4
    assert capped["truncated"] is True

    defaulted = await ctx_tools.ctx_peek_handler(_SESSION, {"name": "notes"})
    assert defaulted["content"] == "hell"
    assert defaulted["truncated"] is True


async def test_ctx_peek_past_end_is_empty_not_truncated(
    wired: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _peek_cap(monkeypatch, 1024)
    monkeypatch.setattr(
        "aios.services.context_variables.resolve_readable",
        AsyncMock(return_value=_variable(content="hello world")),
    )

    out = await ctx_tools.ctx_peek_handler(_SESSION, {"name": "notes", "offset": 11})

    assert out["content"] == ""
    assert out["length"] == 0
    assert out["truncated"] is False


# ─── ctx_grep ────────────────────────────────────────────────────────────────


async def test_ctx_grep_bad_regex_bails(wired: SimpleNamespace) -> None:
    with pytest.raises(ToolBail, match="invalid pattern"):
        await ctx_tools.ctx_grep_handler(_SESSION, {"name": "notes", "pattern": "("})


async def test_ctx_grep_line_matches_with_offsets(
    wired: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "aios.services.context_variables.resolve_readable",
        AsyncMock(return_value=_variable(content="alpha\nbeta\ngamma alpha\n")),
    )

    out = await ctx_tools.ctx_grep_handler(_SESSION, {"name": "notes", "pattern": "alpha"})

    assert out["total_matches"] == 2
    assert out["truncated"] is False
    assert out["matches"] == [
        {"line": 1, "byte_offset": 0, "text": "alpha"},
        {"line": 3, "byte_offset": 11, "text": "gamma alpha"},
    ]


async def test_ctx_grep_max_matches_truncation(
    wired: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "aios.services.context_variables.resolve_readable",
        AsyncMock(return_value=_variable(content="alpha\nbeta\ngamma alpha\n")),
    )

    out = await ctx_tools.ctx_grep_handler(
        _SESSION, {"name": "notes", "pattern": "a", "max_matches": 1}
    )

    assert out["total_matches"] == 3  # every line matches; total keeps counting
    assert len(out["matches"]) == 1
    assert out["truncated"] is True


# ─── ctx_write ───────────────────────────────────────────────────────────────


async def test_ctx_write_agent_scope_without_agent_bails(
    wired: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    wired.session.agent_id = None
    write_mock = AsyncMock()
    monkeypatch.setattr("aios.services.context_variables.write_variable", write_mock)

    with pytest.raises(ToolBail, match="no agent"):
        await ctx_tools.ctx_write_handler(
            _SESSION, {"name": "wm", "content": "x", "scope": "agent"}
        )
    write_mock.assert_not_awaited()


async def test_ctx_write_content_over_cap_bails(
    wired: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The service's byte-cap ``ValueError`` surfaces as a model-visible bail."""
    monkeypatch.setattr(
        "aios.services.context_variables.write_variable",
        AsyncMock(side_effect=ValueError("content exceeds 8388608-byte cap")),
    )

    with pytest.raises(ToolBail, match="exceeds"):
        await ctx_tools.ctx_write_handler(_SESSION, {"name": "big", "content": "x"})


async def test_ctx_write_forwards_append_and_scope_routing(
    wired: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    var = _variable("log", content="ab")
    write_mock = AsyncMock(return_value=var)
    monkeypatch.setattr("aios.services.context_variables.write_variable", write_mock)

    out = await ctx_tools.ctx_write_handler(
        _SESSION, {"name": "log", "content": "b", "append": True}
    )

    assert out["name"] == "log"
    assert out["size_bytes"] == 2
    assert write_mock.await_args is not None
    kwargs = write_mock.await_args.kwargs
    assert kwargs["append"] is True
    # scope='session' routes the session id and leaves agent_id unset.
    assert kwargs["session_id"] == _SESSION
    assert kwargs["agent_id"] is None
    assert kwargs["metadata"] == {"writer_session_id": _SESSION}


async def test_ctx_write_agent_scope_routes_agent_id(
    wired: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_mock = AsyncMock(return_value=_variable("wm", scope="agent"))
    monkeypatch.setattr("aios.services.context_variables.write_variable", write_mock)

    await ctx_tools.ctx_write_handler(_SESSION, {"name": "wm", "content": "x", "scope": "agent"})

    assert write_mock.await_args is not None
    kwargs = write_mock.await_args.kwargs
    assert kwargs["session_id"] is None
    assert kwargs["agent_id"] == "agt_1"


# ─── ctx_eval (argument validation only — the exec path needs Docker) ────────


@pytest.mark.parametrize(
    "arguments",
    [
        {},  # neither
        {"code": "print(1)", "helper": "h"},  # both
    ],
)
async def test_ctx_eval_requires_exactly_one_of_code_helper(
    wired: SimpleNamespace, arguments: dict[str, Any]
) -> None:
    with pytest.raises(ToolBail, match="exactly one of code / helper"):
        await ctx_tools.ctx_eval_handler(_SESSION, arguments)


async def test_ctx_eval_helper_of_wrong_kind_bails(
    wired: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A helper name resolving to a non-helper variable bails before any
    sandbox acquisition."""
    monkeypatch.setattr(
        "aios.services.context_variables.resolve_readable",
        AsyncMock(return_value=_variable("h", kind="data", content="print(1)")),
    )

    with pytest.raises(ToolBail, match="not a helper"):
        await ctx_tools.ctx_eval_handler(_SESSION, {"helper": "h"})
