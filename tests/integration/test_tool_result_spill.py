"""Integration repro for #735: an oversized tool result must be capped at
the append (dispatch) boundary so ``cumulative_tokens`` reflects the capped
size and windowing stays correct by construction.

A single tool result larger than the model context budget wedges the
session — windowing can only drop whole events, never shrink one, so a
lone oversized result can't be shed.  The fix caps the result content at
the SERVICE append sink (``sessions_service.append_tool_result``): the
inline body is replaced with a handle+preview stub and the full output is
spilled into a session-scoped ``kind='spill'`` context variable
(``sandbox/tool_result_spill.py``, docs/rlm.md), recoverable via the
``ctx_*`` tools instead of re-entering the prompt wholesale.

This test drives the real service sink against a testcontainer Postgres,
then asserts:

* oversized → the stored tool event content is a ``[Tool result spilled:``
  stub naming the spill variable handle and carrying the deterministic
  preview, AND the full body sits in a session-scoped ``kind='spill'``
  context variable, byte-for-byte with a matching sha256.
* within-cap → the content is stored verbatim, no variable row written.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import AsyncIterator
from typing import Any
from unittest import mock

import asyncpg
import pytest

from aios.config import get_settings
from aios.db import queries
from aios.db.pool import create_pool
from aios.models.agents import ToolSpec
from aios.sandbox.tool_result_spill import PREVIEW_CHARS, spill_variable_name
from aios.services import sessions as sessions_service
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration

# Pinned via AIOS_TOOL_RESULT_MAX_CHARS below so the test controls the cap
# instead of tracking the production default.
_MAX_CHARS = 5_000


def _payload(chars: int) -> str:
    """Distinct numbered lines so the preview assertion is meaningful."""
    lines = [f"row-{i:08d}" for i in range(chars // 13 + 1)]
    return "\n".join(lines)[:chars]


@pytest.fixture
async def spill_session(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    """Yield ``(pool, session_id, tool_call_id)`` for a session whose event
    log contains an assistant message carrying a single ``tool_calls`` entry
    (so ``append_tool_result`` finds a parent), with the inline cap pinned
    to ``_MAX_CHARS``.
    """
    with mock.patch.dict(os.environ, {"AIOS_TOOL_RESULT_MAX_CHARS": str(_MAX_CHARS)}):
        get_settings.cache_clear()
        pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
        try:
            account_id = "acc_spill"
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO accounts
                        (id, parent_account_id, can_mint_children, display_name)
                    VALUES ($1, NULL, TRUE, 'spill-test-root')
                    """,
                    account_id,
                )
            _agent, _env, session = await seed_agent_env_session(
                pool,
                account_id=account_id,
                prefix="spill",
                tools=[ToolSpec(type="bash")],
            )
            tool_call_id = "tc_spill_1"
            async with pool.acquire() as conn:
                await queries.append_event(
                    conn,
                    account_id=account_id,
                    session_id=session.id,
                    kind="message",
                    data={
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": tool_call_id,
                                "type": "function",
                                "function": {"name": "bash", "arguments": "{}"},
                            }
                        ],
                    },
                )
            yield pool, session.id, tool_call_id
        finally:
            await pool.close()
            get_settings.cache_clear()


async def _stored_tool_content(pool: asyncpg.Pool[Any], session_id: str, tool_call_id: str) -> str:
    async with pool.acquire() as conn:
        event = await queries.find_tool_result_event(
            conn, session_id, tool_call_id, account_id="acc_spill"
        )
    assert event is not None
    content = event.data["content"]
    assert isinstance(content, str)
    return content


class TestToolResultSpill:
    async def test_oversized_result_spilled_and_stubbed(
        self,
        spill_session: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        pool, session_id, tool_call_id = spill_session
        original = _payload(4 * _MAX_CHARS)
        assert len(original) > _MAX_CHARS

        async with pool.acquire() as conn:
            await sessions_service.append_tool_result(
                conn,
                account_id="acc_spill",
                session_id=session_id,
                tool_call_id=tool_call_id,
                content=original,
            )

        name = spill_variable_name(tool_call_id)
        stored = await _stored_tool_content(pool, session_id, tool_call_id)
        assert stored.startswith("[Tool result spilled:")
        assert f"{name!r}" in stored  # the stub names the recovery handle
        assert stored.endswith(original[:PREVIEW_CHARS])  # deterministic preview
        assert len(stored) < len(original)

        # The full body lives in the session-scoped spill variable,
        # byte-for-byte, with the matching content sha.
        async with pool.acquire() as conn:
            var = await queries.resolve_readable_variable(
                conn, account_id="acc_spill", session_id=session_id, agent_id=None, name=name
            )
        assert var is not None
        assert var.scope == "session"
        assert var.session_id == session_id
        assert var.kind == "spill"
        assert var.content == original
        assert var.content_sha256 == hashlib.sha256(original.encode("utf-8")).hexdigest()
        assert var.content_size_bytes == len(original.encode("utf-8"))
        assert var.metadata == {"tool_call_id": tool_call_id}

    async def test_within_cap_result_stored_verbatim(
        self,
        spill_session: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        pool, session_id, tool_call_id = spill_session
        original = _payload(100)  # well under the cap

        async with pool.acquire() as conn:
            await sessions_service.append_tool_result(
                conn,
                account_id="acc_spill",
                session_id=session_id,
                tool_call_id=tool_call_id,
                content=original,
            )

        stored = await _stored_tool_content(pool, session_id, tool_call_id)
        assert stored == original

        # No spill: no context-variable row lands for the session.
        async with pool.acquire() as conn:
            var = await queries.resolve_readable_variable(
                conn,
                account_id="acc_spill",
                session_id=session_id,
                agent_id=None,
                name=spill_variable_name(tool_call_id),
            )
            count = await conn.fetchval(
                "SELECT count(*) FROM context_variables WHERE session_id = $1", session_id
            )
        assert var is None
        assert count == 0
