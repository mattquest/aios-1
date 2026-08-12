from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from aios.db.queries.events import count_tool_errors_since_last_user


async def test_count_tool_errors_is_account_tool_and_user_turn_scoped() -> None:
    conn = MagicMock()
    conn.fetchval = AsyncMock(return_value=2)

    assert (
        await count_tool_errors_since_last_user(
            conn,
            "sess_1",
            "mcp__kine__propose_workout",
            "missing_required",
            account_id="acc_1",
        )
        == 2
    )

    sql, session_id, account_id, tool_name, error_code = conn.fetchval.await_args.args
    assert "account_id = $2" in sql
    assert "role = 'tool' AND is_error IS TRUE" in sql
    assert "role = 'user'" in sql
    assert "data->'metadata'->>'mcp_error_code' = $4" in sql
    assert (session_id, account_id, tool_name, error_code) == (
        "sess_1",
        "acc_1",
        "mcp__kine__propose_workout",
        "missing_required",
    )
