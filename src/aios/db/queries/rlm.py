"""rlm dispatch queries (docs/rlm.md): spawn budgets + the per-session ledger.

A subsystem module of the ``aios.db.queries`` package — see ``__init__`` for the
shared scoping helpers and the package-level re-export contract. Raw SQL against
asyncpg, same conventions as the rest of the package.

The budget model: depth and a child-token allowance ride the trusted
``request_opened`` spawn edge (inherited-and-decremented down the tree); the
per-session ``rlm_ledgers`` row counts spawns per assistant step and harvested
child tokens per turn. Everything here is dispatch-path enforcement — the
prompt never carries a budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import asyncpg


@dataclass(frozen=True)
class RlmSpawnBudget:
    """The budgets carried on a session's inbound rlm spawn edge."""

    depth: int
    token_budget: int


async def get_rlm_spawn_budget(
    conn: asyncpg.Connection[Any],
    session_id: str,
    *,
    account_id: str,
) -> RlmSpawnBudget | None:
    """The budgets on ``session_id``'s inbound rlm spawn edge, or ``None``.

    ``None`` means the session was not rlm-spawned (a root): the caller falls
    back to the operator defaults. The discriminant is the edge's
    ``rlm_token_budget`` field, written only by the rlm spawn path.
    """
    row = await conn.fetchrow(
        "SELECT (data->>'depth')::int AS depth, "
        "       (data->>'rlm_token_budget')::bigint AS token_budget "
        "FROM events "
        "WHERE session_id = $1 AND account_id = $2 AND kind = 'lifecycle' "
        "  AND data->>'event' = 'request_opened' AND data ? 'rlm_token_budget' "
        "ORDER BY seq ASC LIMIT 1",
        session_id,
        account_id,
    )
    if row is None:
        return None
    return RlmSpawnBudget(depth=row["depth"], token_budget=row["token_budget"])


async def get_tool_call_parent_seq(
    conn: asyncpg.Connection[Any],
    session_id: str,
    tool_call_id: str,
    *,
    account_id: str,
) -> int | None:
    """Seq of the assistant event that issued ``tool_call_id`` (the step key)."""
    seq: Any = await conn.fetchval(
        "SELECT seq FROM events "
        "WHERE session_id = $1 "
        "  AND account_id = $3 "
        "  AND kind = 'message' "
        "  AND data->>'role' = 'assistant' "
        "  AND data ? 'tool_calls' "
        "  AND data->'tool_calls' @> jsonb_build_array("
        "    jsonb_build_object('id', $2::text)) "
        "ORDER BY seq DESC LIMIT 1",
        session_id,
        tool_call_id,
        account_id,
    )
    return int(seq) if seq is not None else None


async def admit_rlm_child(
    conn: asyncpg.Connection[Any],
    *,
    session_id: str,
    step_key: int,
    turn_key: int,
) -> tuple[int, int]:
    """Atomically count one child spawn against the ledger.

    Returns ``(children_spawned_this_step, child_tokens_this_turn)`` AFTER the
    increment. Counters reset when their key advances; a refused admission
    leaves its increment in place, so repeated over-budget attempts stay
    refused (the counter counts attempts, which is the conservative side).
    """
    row = await conn.fetchrow(
        """
        INSERT INTO rlm_ledgers (session_id, step_key, children_spawned, turn_key, child_tokens)
        VALUES ($1, $2, 1, $3, 0)
        ON CONFLICT (session_id) DO UPDATE SET
            children_spawned = CASE
                WHEN rlm_ledgers.step_key = EXCLUDED.step_key
                THEN rlm_ledgers.children_spawned + 1 ELSE 1 END,
            step_key = EXCLUDED.step_key,
            child_tokens = CASE
                WHEN rlm_ledgers.turn_key = EXCLUDED.turn_key
                THEN rlm_ledgers.child_tokens ELSE 0 END,
            turn_key = EXCLUDED.turn_key,
            updated_at = now()
        RETURNING children_spawned, child_tokens
        """,
        session_id,
        step_key,
        turn_key,
    )
    assert row is not None
    return row["children_spawned"], row["child_tokens"]


async def add_rlm_child_tokens(
    conn: asyncpg.Connection[Any],
    *,
    session_id: str,
    turn_key: int,
    tokens: int,
) -> None:
    """Accrue a harvested child's tokens onto the spawner's turn accumulator.

    A harvest landing after the turn advanced (``turn_key`` mismatch) is
    dropped — it belongs to a turn whose budget window is closed.
    """
    await conn.execute(
        "UPDATE rlm_ledgers SET child_tokens = child_tokens + $3, updated_at = now() "
        "WHERE session_id = $1 AND turn_key = $2",
        session_id,
        turn_key,
        tokens,
    )


@dataclass(frozen=True)
class SessionUsage:
    """A session's own accumulated inference usage (its row counters)."""

    input_tokens: int
    output_tokens: int
    cost_microusd: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


async def get_session_usage(
    conn: asyncpg.Connection[Any],
    session_id: str,
    *,
    account_id: str,
) -> SessionUsage:
    row = await conn.fetchrow(
        "SELECT input_tokens, output_tokens, cost_microusd FROM sessions "
        "WHERE id = $1 AND account_id = $2",
        session_id,
        account_id,
    )
    if row is None:
        return SessionUsage(input_tokens=0, output_tokens=0, cost_microusd=0)
    return SessionUsage(
        input_tokens=row["input_tokens"] or 0,
        output_tokens=row["output_tokens"] or 0,
        cost_microusd=row["cost_microusd"] or 0,
    )


async def summarize_session_tool_calls(
    conn: asyncpg.Connection[Any],
    session_id: str,
    *,
    account_id: str,
) -> dict[str, int]:
    """``{tool_name: result_count}`` over a session's tool-result events —
    the compact ctx-call trace carried on rlm result provenance."""
    rows = await conn.fetch(
        "SELECT tool_name, count(*) AS n FROM events "
        "WHERE session_id = $1 AND account_id = $2 AND kind = 'message' "
        "  AND tool_name IS NOT NULL AND data->>'role' = 'tool' "
        "GROUP BY tool_name ORDER BY tool_name",
        session_id,
        account_id,
    )
    return {row["tool_name"]: row["n"] for row in rows}


async def get_session_last_user_seq(
    conn: asyncpg.Connection[Any],
    session_id: str,
    *,
    account_id: str,
) -> int:
    """The session's ``last_user_seq`` watermark (the rlm turn key)."""
    value: Any = await conn.fetchval(
        "SELECT last_user_seq FROM sessions WHERE id = $1 AND account_id = $2",
        session_id,
        account_id,
    )
    return int(value or 0)
