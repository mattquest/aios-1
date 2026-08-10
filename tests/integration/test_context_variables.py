"""Integration coverage for the context-variable store (docs/rlm.md).

Drives the real query layer (``db/queries/context_variables.py``,
``db/queries/rlm.py``) against a testcontainer Postgres so the pieces the
unit suite cannot see are exercised for real: the partial-unique-index
upsert conflict targets, the three-way readability predicate and its
collision precedence, the grants junction, the session-derived spill
write, and the atomic rlm-ledger counter resets.

Layout mirrors ``test_list_account_triggers.py``: a ``pool`` fixture that
seeds one root account, ``seed_agent_env_session`` for session-shaped
scaffolding, and the shared ``conn_two_accounts`` fixture for the tenant
isolation obligations.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.errors import NotFoundError
from aios.models.context_variables import ContextVariable, VariableKind, VariableScope
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration

ACC = "acc_cv_root"


@pytest.fixture
async def pool(migrated_db_url: str, _reset_db_state: None) -> AsyncIterator[asyncpg.Pool[Any]]:
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                f"VALUES ('{ACC}', NULL, TRUE, 'cv-root')"
            )
        yield pool
    finally:
        await pool.close()


def _digest(content: str) -> tuple[str, int]:
    """The caller-computed ``(sha256_hex, utf8_byte_size)`` triple half."""
    encoded = content.encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), len(encoded)


async def _write(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str = ACC,
    scope: VariableScope = "session",
    session_id: str | None = None,
    agent_id: str | None = None,
    name: str,
    content: str,
    kind: VariableKind = "data",
    description: str = "",
    schema_tag: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ContextVariable:
    sha, size = _digest(content)
    return await queries.upsert_context_variable(
        conn,
        account_id=account_id,
        scope=scope,
        session_id=session_id,
        agent_id=agent_id,
        name=name,
        kind=kind,
        description=description,
        schema_tag=schema_tag,
        content=content,
        content_sha256=sha,
        content_size_bytes=size,
        metadata=metadata or {},
    )


async def _seed_conn_session(
    conn: asyncpg.Connection[Any], *, account_id: str, prefix: str
) -> tuple[str, str]:
    """``(agent_id, session_id)`` seeded on a bare connection (no pool)."""
    agent = await queries.insert_agent(
        conn,
        account_id=account_id,
        name=f"{prefix}-agent",
        model="openrouter/test",
        system="",
        tools=[],
        mcp_servers=[],
        http_servers=[],
        description=None,
        metadata={},
        litellm_extra={},
        window_min=50_000,
        window_max=150_000,
        preempt_policy="wait",
    )
    env = await queries.insert_environment(conn, account_id=account_id, name=f"{prefix}-env")
    session = await queries.insert_session(
        conn,
        account_id=account_id,
        agent_id=agent.id,
        environment_id=env.id,
        agent_version=agent.version,
        title=None,
        metadata={},
    )
    return agent.id, session.id


# ── upsert_context_variable ──────────────────────────────────────────────


async def test_upsert_overwrite_preserves_id_and_updates_fields(pool: asyncpg.Pool[Any]) -> None:
    """A second write at the same ``(session, name)`` key overwrites in place:
    same row id, new content/sha/size/kind/metadata following the write."""
    _, _, session = await seed_agent_env_session(pool, account_id=ACC, prefix="up")
    async with pool.acquire() as conn:
        created = await _write(
            conn, session_id=session.id, name="notes", content="v1", metadata={"rev": 1}
        )
        overwritten = await _write(
            conn,
            session_id=session.id,
            name="notes",
            content="v2-longer",
            kind="digest",
            description="second write",
            schema_tag="tag2",
            metadata={"rev": 2},
        )

    assert overwritten.id == created.id
    assert overwritten.created_at == created.created_at
    assert overwritten.updated_at >= created.updated_at
    assert overwritten.content == "v2-longer"
    sha, size = _digest("v2-longer")
    assert overwritten.content_sha256 == sha
    assert overwritten.content_size_bytes == size
    assert overwritten.kind == "digest"
    assert overwritten.description == "second write"
    assert overwritten.schema_tag == "tag2"
    assert overwritten.metadata == {"rev": 2}

    # One live row at the key — the overwrite did not accrete a sibling.
    async with pool.acquire() as conn:
        listed = await queries.list_context_variables(conn, account_id=ACC, session_id=session.id)
    assert [v.id for v in listed] == [created.id]


async def test_agent_scope_key_independent_from_session_scope(pool: asyncpg.Pool[Any]) -> None:
    """The same name at agent scope ``(account, agent, name)`` and session
    scope ``(session, name)`` are two distinct rows — neither upsert
    overwrites the other."""
    agent, _, session = await seed_agent_env_session(pool, account_id=ACC, prefix="scopes")
    async with pool.acquire() as conn:
        sess_var = await _write(conn, session_id=session.id, name="model", content="session-value")
        agent_var = await _write(
            conn, scope="agent", agent_id=agent.id, name="model", content="agent-value"
        )
        # Overwriting one scope leaves the other untouched.
        await _write(conn, scope="agent", agent_id=agent.id, name="model", content="agent-value-2")
        sess_after = await queries.get_context_variable(conn, sess_var.id, account_id=ACC)
        agent_after = await queries.get_context_variable(conn, agent_var.id, account_id=ACC)

    assert sess_var.id != agent_var.id
    assert sess_after.content == "session-value"
    assert agent_after.content == "agent-value-2"
    assert agent_after.scope == "agent"
    assert agent_after.session_id is None


async def test_archived_row_does_not_block_fresh_write(pool: asyncpg.Pool[Any]) -> None:
    """The conflict target is the partial (live-rows-only) unique index, so an
    archived row at the key never blocks — a fresh write mints a fresh id."""
    _, _, session = await seed_agent_env_session(pool, account_id=ACC, prefix="arch")
    async with pool.acquire() as conn:
        first = await _write(conn, session_id=session.id, name="scratch", content="old")
        archived = await queries.archive_context_variable(conn, first.id, account_id=ACC)
        fresh = await _write(conn, session_id=session.id, name="scratch", content="new")

    assert archived.archived_at is not None
    assert fresh.id != first.id
    assert fresh.content == "new"
    assert fresh.archived_at is None


# ── resolve_readable_variable ────────────────────────────────────────────


async def test_resolve_precedence_session_beats_agent_beats_grant(pool: asyncpg.Pool[Any]) -> None:
    """Name-collision precedence is session scope, then agent scope, then
    granted — asserted with the same name present at every plane."""
    agent, _, reader = await seed_agent_env_session(pool, account_id=ACC, prefix="rdr")
    _, _, donor = await seed_agent_env_session(pool, account_id=ACC, prefix="dnr")
    async with pool.acquire() as conn:
        # "n3" exists at all three planes; "n2" at agent + grant only.
        own = await _write(conn, session_id=reader.id, name="n3", content="own")
        ag3 = await _write(conn, scope="agent", agent_id=agent.id, name="n3", content="agent")
        ag2 = await _write(conn, scope="agent", agent_id=agent.id, name="n2", content="agent")
        g3 = await _write(conn, session_id=donor.id, name="n3", content="granted")
        g2 = await _write(conn, session_id=donor.id, name="n2", content="granted")
        await queries.insert_context_variable_grants(
            conn, session_id=reader.id, variable_ids=[g3.id, g2.id]
        )

        hit3 = await queries.resolve_readable_variable(
            conn, account_id=ACC, session_id=reader.id, agent_id=agent.id, name="n3"
        )
        hit2 = await queries.resolve_readable_variable(
            conn, account_id=ACC, session_id=reader.id, agent_id=agent.id, name="n2"
        )

    assert hit3 is not None and hit3.id == own.id  # session wins over agent + grant
    assert hit3.content == "own"
    assert hit2 is not None and hit2.id == ag2.id  # agent wins over grant
    assert ag3.id not in {hit3.id, hit2.id}


async def test_resolve_grant_only_and_unknown_name(pool: asyncpg.Pool[Any]) -> None:
    """A grant alone resolves (the child-session read plane); a name nobody
    holds resolves to ``None``; ``include_content=False`` keeps content off."""
    agent, _, reader = await seed_agent_env_session(pool, account_id=ACC, prefix="gr")
    _, _, donor = await seed_agent_env_session(pool, account_id=ACC, prefix="gd")
    async with pool.acquire() as conn:
        granted = await _write(conn, session_id=donor.id, name="n1", content="shared-body")
        await queries.insert_context_variable_grants(
            conn, session_id=reader.id, variable_ids=[granted.id]
        )

        hit = await queries.resolve_readable_variable(
            conn, account_id=ACC, session_id=reader.id, agent_id=agent.id, name="n1"
        )
        meta_only = await queries.resolve_readable_variable(
            conn,
            account_id=ACC,
            session_id=reader.id,
            agent_id=agent.id,
            name="n1",
            include_content=False,
        )
        miss = await queries.resolve_readable_variable(
            conn, account_id=ACC, session_id=reader.id, agent_id=agent.id, name="nope"
        )

    assert hit is not None and hit.id == granted.id
    assert hit.session_id == donor.id  # the granted row itself, no copy
    assert hit.content == "shared-body"
    assert meta_only is not None and meta_only.content is None
    assert meta_only.content_sha256 == granted.content_sha256
    assert miss is None


# ── list_readable_variables ──────────────────────────────────────────────


async def _seed_readable_set(
    pool: asyncpg.Pool[Any],
) -> tuple[str, str]:
    """Seed a reader with one variable per plane (+ one unreadable) and pin
    ``updated_at`` so ordering assertions are deterministic.

    Returns ``(agent_id, reader_session_id)``. Seeded names, newest first:
    ``s_data`` (session/data), ``a_data`` (agent/data), ``s_spill``
    (session/spill), ``g_data`` (granted/data). ``u_data`` on the donor is
    NOT granted and must never appear.
    """
    agent, _, reader = await seed_agent_env_session(pool, account_id=ACC, prefix="lr")
    _, _, donor = await seed_agent_env_session(pool, account_id=ACC, prefix="ld")
    async with pool.acquire() as conn:
        s_data = await _write(conn, session_id=reader.id, name="s_data", content="c1")
        s_spill = await _write(
            conn, session_id=reader.id, name="s_spill", content="c2", kind="spill"
        )
        a_data = await _write(conn, scope="agent", agent_id=agent.id, name="a_data", content="c3")
        g_data = await _write(conn, session_id=donor.id, name="g_data", content="c4")
        await _write(conn, session_id=donor.id, name="u_data", content="c5")
        await queries.insert_context_variable_grants(
            conn, session_id=reader.id, variable_ids=[g_data.id]
        )
        for variable_id, age_seconds in (
            (s_data.id, 0),
            (a_data.id, 10),
            (s_spill.id, 20),
            (g_data.id, 30),
        ):
            await conn.execute(
                "UPDATE context_variables "
                "SET updated_at = now() - ($2 * interval '1 second') WHERE id = $1",
                variable_id,
                age_seconds,
            )
    return agent.id, reader.id


async def test_list_readable_metadata_only_and_ordering(pool: asyncpg.Pool[Any]) -> None:
    agent_id, reader_id = await _seed_readable_set(pool)
    async with pool.acquire() as conn:
        listed = await queries.list_readable_variables(
            conn, account_id=ACC, session_id=reader_id, agent_id=agent_id
        )

    # Everything readable, nothing else, newest updated_at first.
    assert [v.name for v in listed] == ["s_data", "a_data", "s_spill", "g_data"]
    # Metadata-only projection: content never travels on a listing, while the
    # cheap columns (sha, size) still do.
    assert all(v.content is None for v in listed)
    assert all(v.content_sha256 for v in listed)
    assert all(v.content_size_bytes > 0 for v in listed)


async def test_list_readable_scope_and_kind_filters(pool: asyncpg.Pool[Any]) -> None:
    agent_id, reader_id = await _seed_readable_set(pool)
    async with pool.acquire() as conn:
        session_scoped = await queries.list_readable_variables(
            conn, account_id=ACC, session_id=reader_id, agent_id=agent_id, scope="session"
        )
        agent_scoped = await queries.list_readable_variables(
            conn, account_id=ACC, session_id=reader_id, agent_id=agent_id, scope="agent"
        )
        spills = await queries.list_readable_variables(
            conn, account_id=ACC, session_id=reader_id, agent_id=agent_id, kind="spill"
        )

    # The scope filter applies to the variable's own scope: a granted row is
    # session-scoped (on its donor), so it stays in the 'session' slice.
    assert {v.name for v in session_scoped} == {"s_data", "s_spill", "g_data"}
    assert [v.name for v in agent_scoped] == ["a_data"]
    assert [v.name for v in spills] == ["s_spill"]


# ── tenant isolation ─────────────────────────────────────────────────────


async def test_tenant_isolation_get_list_resolve(
    conn_two_accounts: asyncpg.Connection[Any],
) -> None:
    """Account B never sees account A's variables — by id (get), by listing,
    or through the readability union (even with a forged cross-account
    grant row, the ``account_id`` predicate holds)."""
    conn = conn_two_accounts
    a_agent, a_session = await _seed_conn_session(conn, account_id="acc_a", prefix="ta")
    b_agent, b_session = await _seed_conn_session(conn, account_id="acc_b", prefix="tb")

    a_sess_var = await _write(
        conn, account_id="acc_a", session_id=a_session, name="secret", content="a-only"
    )
    await _write(
        conn,
        account_id="acc_a",
        scope="agent",
        agent_id=a_agent,
        name="world",
        content="a-world",
    )

    with pytest.raises(NotFoundError):
        await queries.get_context_variable(conn, a_sess_var.id, account_id="acc_b")

    assert await queries.list_context_variables(conn, account_id="acc_b") == []

    # Even a grant pointing B's session at A's variable cannot cross the
    # account boundary: the readable predicate is account-scoped first.
    await queries.insert_context_variable_grants(
        conn, session_id=b_session, variable_ids=[a_sess_var.id]
    )
    resolved = await queries.resolve_readable_variable(
        conn, account_id="acc_b", session_id=b_session, agent_id=b_agent, name="secret"
    )
    assert resolved is None
    assert (
        await queries.list_readable_variables(
            conn, account_id="acc_b", session_id=b_session, agent_id=b_agent
        )
        == []
    )

    # Sanity: A itself resolves its own variable.
    own = await queries.resolve_readable_variable(
        conn, account_id="acc_a", session_id=a_session, agent_id=a_agent, name="secret"
    )
    assert own is not None and own.id == a_sess_var.id


# ── insert_context_variable_grants ───────────────────────────────────────


async def test_grant_insert_idempotent(pool: asyncpg.Pool[Any]) -> None:
    """Re-asserting the same grant pair (a replayed spawn) is a no-op, not an
    error, and an empty id list is a no-op outright."""
    _, _, reader = await seed_agent_env_session(pool, account_id=ACC, prefix="gi")
    _, _, donor = await seed_agent_env_session(pool, account_id=ACC, prefix="gj")
    async with pool.acquire() as conn:
        var = await _write(conn, session_id=donor.id, name="g", content="x")
        await queries.insert_context_variable_grants(
            conn, session_id=reader.id, variable_ids=[var.id]
        )
        await queries.insert_context_variable_grants(
            conn, session_id=reader.id, variable_ids=[var.id]
        )
        await queries.insert_context_variable_grants(conn, session_id=reader.id, variable_ids=[])
        count = await conn.fetchval(
            "SELECT count(*) FROM context_variable_grants "
            "WHERE session_id = $1 AND context_variable_id = $2",
            reader.id,
            var.id,
        )
    assert count == 1


# ── spill_tool_result_variable ───────────────────────────────────────────


async def test_spill_writes_session_scoped_spill_row(pool: asyncpg.Pool[Any]) -> None:
    """The spill write derives ``account_id`` from the session row and lands a
    session-scoped ``kind='spill'`` variable carrying the tool_call_id."""
    _, _, session = await seed_agent_env_session(pool, account_id=ACC, prefix="sp")
    content = "spilled body"
    sha, size = _digest(content)
    await queries.spill_tool_result_variable(
        pool,
        session_id=session.id,
        name="tool_result_tc1",
        content=content,
        content_sha256=sha,
        content_size_bytes=size,
        tool_call_id="tc1",
    )

    async with pool.acquire() as conn:
        var = await queries.resolve_readable_variable(
            conn, account_id=ACC, session_id=session.id, agent_id=None, name="tool_result_tc1"
        )
    assert var is not None
    assert var.scope == "session"
    assert var.session_id == session.id
    assert var.kind == "spill"
    assert var.content == content
    assert var.content_sha256 == sha
    assert var.content_size_bytes == size
    assert var.metadata == {"tool_call_id": "tc1"}


async def test_spill_overwrite_same_name_is_idempotent(pool: asyncpg.Pool[Any]) -> None:
    """A losing appender re-running the spill upserts onto the same row —
    one row at the key, id preserved, content following the last write."""
    _, _, session = await seed_agent_env_session(pool, account_id=ACC, prefix="sq")
    for body in ("first", "second"):
        sha, size = _digest(body)
        await queries.spill_tool_result_variable(
            pool,
            session_id=session.id,
            name="tool_result_tc2",
            content=body,
            content_sha256=sha,
            content_size_bytes=size,
            tool_call_id="tc2",
        )

    async with pool.acquire() as conn:
        rows = await queries.list_context_variables(conn, account_id=ACC, session_id=session.id)
        var = await queries.get_context_variable(conn, rows[0].id, account_id=ACC)
    assert len(rows) == 1
    assert var.content == "second"


async def test_spill_noop_for_missing_session(pool: asyncpg.Pool[Any]) -> None:
    """No session row → the INSERT..SELECT selects nothing: no orphan variable,
    no error."""
    sha, size = _digest("orphan")
    await queries.spill_tool_result_variable(
        pool,
        session_id="sess_does_not_exist",
        name="tool_result_tc3",
        content="orphan",
        content_sha256=sha,
        content_size_bytes=size,
        tool_call_id="tc3",
    )
    async with pool.acquire() as conn:
        count = await conn.fetchval("SELECT count(*) FROM context_variables")
    assert count == 0


# ── rlm ledger (db/queries/rlm.py) ───────────────────────────────────────


async def test_admit_rlm_child_counts_within_step_and_resets_on_advance(
    pool: asyncpg.Pool[Any],
) -> None:
    _, _, session = await seed_agent_env_session(pool, account_id=ACC, prefix="rla")
    async with pool.acquire() as conn:
        admit = queries.admit_rlm_child
        assert await admit(conn, session_id=session.id, step_key=10, turn_key=1) == (1, 0)
        assert await admit(conn, session_id=session.id, step_key=10, turn_key=1) == (2, 0)
        assert await admit(conn, session_id=session.id, step_key=10, turn_key=1) == (3, 0)
        # The step key advancing resets the children counter to this spawn.
        assert await admit(conn, session_id=session.id, step_key=11, turn_key=1) == (1, 0)
        assert await admit(conn, session_id=session.id, step_key=11, turn_key=1) == (2, 0)


async def test_admit_rlm_child_tokens_reset_on_turn_advance(pool: asyncpg.Pool[Any]) -> None:
    _, _, session = await seed_agent_env_session(pool, account_id=ACC, prefix="rlb")
    async with pool.acquire() as conn:
        await queries.admit_rlm_child(conn, session_id=session.id, step_key=1, turn_key=7)
        await queries.add_rlm_child_tokens(conn, session_id=session.id, turn_key=7, tokens=700)
        # Same turn, new step: accumulated child tokens survive.
        assert await queries.admit_rlm_child(
            conn, session_id=session.id, step_key=2, turn_key=7
        ) == (1, 700)
        # Turn advances: the child-token accumulator resets with it.
        assert await queries.admit_rlm_child(
            conn, session_id=session.id, step_key=3, turn_key=8
        ) == (1, 0)


async def test_add_rlm_child_tokens_accrues_and_drops_stale_turn(pool: asyncpg.Pool[Any]) -> None:
    _, _, session = await seed_agent_env_session(pool, account_id=ACC, prefix="rlc")
    async with pool.acquire() as conn:
        await queries.admit_rlm_child(conn, session_id=session.id, step_key=1, turn_key=5)
        await queries.add_rlm_child_tokens(conn, session_id=session.id, turn_key=5, tokens=100)
        await queries.add_rlm_child_tokens(conn, session_id=session.id, turn_key=5, tokens=250)
        accrued = await conn.fetchval(
            "SELECT child_tokens FROM rlm_ledgers WHERE session_id = $1", session.id
        )
        # A harvest landing after the turn advanced is dropped, not misfiled.
        await queries.add_rlm_child_tokens(conn, session_id=session.id, turn_key=4, tokens=999)
        after_stale = await conn.fetchval(
            "SELECT child_tokens FROM rlm_ledgers WHERE session_id = $1", session.id
        )
    assert accrued == 350
    assert after_stale == 350


async def test_get_rlm_spawn_budget_reads_the_request_opened_edge(
    pool: asyncpg.Pool[Any],
) -> None:
    """``None`` with no edge; ``None`` for a non-rlm edge (no
    ``rlm_token_budget``); the ``(depth, token_budget)`` pair once the rlm
    spawn edge exists."""
    _, env, session = await seed_agent_env_session(pool, account_id=ACC, prefix="rld")
    async with pool.acquire() as conn:
        assert await queries.get_rlm_spawn_budget(conn, session.id, account_id=ACC) is None

        # A plain (non-rlm) request edge does not carry a budget and must not
        # be mistaken for one — the discriminant is the rlm_token_budget field.
        await queries.append_request_opened(
            conn,
            session_id=session.id,
            account_id=ACC,
            request_id="req_plain",
            caller={"kind": "api", "id": "op"},
            depth=1,
            environment_id=env.id,
            frozen_surface={},
            vault_ids=[],
        )
        assert await queries.get_rlm_spawn_budget(conn, session.id, account_id=ACC) is None

        await queries.append_request_opened(
            conn,
            session_id=session.id,
            account_id=ACC,
            request_id="req_rlm",
            caller={"kind": "session", "id": "sess_parent"},
            depth=3,
            environment_id=env.id,
            frozen_surface={},
            vault_ids=[],
            rlm_token_budget=44_000,
        )
        budget = await queries.get_rlm_spawn_budget(conn, session.id, account_id=ACC)

    assert budget is not None
    assert budget.depth == 3
    assert budget.token_budget == 44_000


async def test_get_session_usage_zero_default_for_missing_session(pool: asyncpg.Pool[Any]) -> None:
    async with pool.acquire() as conn:
        usage = await queries.get_session_usage(conn, "sess_missing", account_id=ACC)
    assert (usage.input_tokens, usage.output_tokens, usage.cost_microusd) == (0, 0, 0)
    assert usage.total_tokens == 0
