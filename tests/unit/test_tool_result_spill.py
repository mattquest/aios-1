"""Unit coverage for :func:`aios.sandbox.tool_result_spill.cap_tool_result_content`.

Deterministic, no Docker or Postgres: the executor is an AsyncMock standing in
for the pool/conn, so the tests assert the variable-write seam
(``queries.spill_tool_result_variable``) is driven with the exact
name/content/sha/size the stub advertises. Real SQL behavior lives in
``tests/integration/test_tool_result_spill.py``.
"""

from __future__ import annotations

import hashlib
from typing import Any
from unittest.mock import AsyncMock, patch

from aios.db import queries
from aios.models.context_variables import MAX_CONTENT_BYTES
from aios.sandbox.tool_result_spill import (
    PREVIEW_CHARS,
    cap_tool_result_content,
    spill_variable_name,
)

_SESSION_ID = "sess_spill_unit"
_TOOL_CALL_ID = "call_unit_1"


async def test_within_cap_returns_unchanged_and_writes_nothing() -> None:
    executor = AsyncMock()
    content = "z" * 500
    with patch.object(queries, "spill_tool_result_variable", AsyncMock()) as spill:
        result = await cap_tool_result_content(
            executor, _SESSION_ID, _TOOL_CALL_ID, content, max_chars=1_000
        )
    assert result.content == content
    assert result.variable_name is None
    spill.assert_not_awaited()


async def test_exactly_at_cap_returns_unchanged() -> None:
    """Boundary: ``len(content) == max_chars`` is within the cap (``<=``)."""
    executor = AsyncMock()
    content = "a" * 1_000
    with patch.object(queries, "spill_tool_result_variable", AsyncMock()) as spill:
        result = await cap_tool_result_content(
            executor, _SESSION_ID, _TOOL_CALL_ID, content, max_chars=1_000
        )
    assert result.content == content
    assert result.variable_name is None
    spill.assert_not_awaited()


async def test_over_cap_returns_handle_stub_with_preview() -> None:
    executor = AsyncMock()
    content = "b" * 3_000
    with patch.object(queries, "spill_tool_result_variable", AsyncMock()) as spill:
        result = await cap_tool_result_content(
            executor, _SESSION_ID, _TOOL_CALL_ID, content, max_chars=1_000
        )

    name = spill_variable_name(_TOOL_CALL_ID)
    assert result.variable_name == name
    assert result.content.startswith("[Tool result spilled:")
    assert name in result.content
    assert "3,000 characters" in result.content
    assert "ctx_peek" in result.content
    # Deterministic preview: the head of the content, frozen into the stub and
    # shrunk so the stub itself respects max_chars (1000 - ~400 prose here).
    assert result.content.endswith("b" * min(PREVIEW_CHARS, 1_000 - 400))
    assert len(result.content) <= 1_000

    spill.assert_awaited_once()
    assert spill.await_args is not None
    kwargs = spill.await_args.kwargs
    assert kwargs["session_id"] == _SESSION_ID
    assert kwargs["name"] == name
    assert kwargs["content"] == content
    assert kwargs["tool_call_id"] == _TOOL_CALL_ID
    assert kwargs["content_size_bytes"] == 3_000
    assert kwargs["content_sha256"] == hashlib.sha256(content.encode()).hexdigest()


async def test_variable_name_is_pure_and_sanitized() -> None:
    """The handle is a pure function of ``tool_call_id`` (retries and the
    worker-vs-API race key the same variable), sanitized to the variable-name
    charset and bounded to the 128-char name cap."""
    assert spill_variable_name("call_abc") == "tool_result_call_abc"
    assert spill_variable_name("we/ird id!") == "tool_result_we_ird_id_"
    # Truncation folds in a digest of the FULL id so two long ids sharing a
    # prefix can never alias one variable.
    long_a, long_b = "x" * 300, "x" * 299 + "y"
    assert len(spill_variable_name(long_a)) == 128
    assert spill_variable_name(long_a) != spill_variable_name(long_b)
    assert spill_variable_name(long_a).startswith("tool_result_" + "x" * 100)


async def test_pathological_content_bounded_at_variable_cap() -> None:
    """Content beyond the 8 MiB variable cap is cut at a character boundary
    with a deterministic marker — the stored size never exceeds the cap."""
    executor = AsyncMock()
    content = "d" * (MAX_CONTENT_BYTES + 10_000)
    with patch.object(queries, "spill_tool_result_variable", AsyncMock()) as spill:
        await cap_tool_result_content(
            executor, _SESSION_ID, _TOOL_CALL_ID, content, max_chars=1_000
        )
    assert spill.await_args is not None
    kwargs: Any = spill.await_args.kwargs
    assert kwargs["content_size_bytes"] <= MAX_CONTENT_BYTES
    assert kwargs["content"].endswith("[…content truncated at the context-variable byte cap]")
    assert len(kwargs["content"].encode("utf-8")) == kwargs["content_size_bytes"]
