"""RLM context variables: named out-of-context content handles (docs/rlm.md).

A context variable is a named handle whose content lives out of the prompt
window — the substrate for the RLM layer (``ctx_*`` tools, ``rlm_query``
child grants, spill inversion). Content is a ``text`` column with the
``memories`` discipline (sha256 + size + ``octet_length`` CHECK, byte cap
enforced at the model layer); TOAST stores large values out-of-row, so no
large-object machinery is needed.

Two scopes, discriminated by the ``scope`` column:

* ``session`` — keyed ``(session_id, name)``; rows cascade with the session.
* ``agent`` — keyed ``(account_id, agent_id, name)``; ``session_id`` NULL.
  Agent scope is what makes a world model outlive any session.

``context_variable_grants`` is the read-only visibility junction written at
``rlm_query`` spawn time inside the child-creation transaction: a child
session sees granted variables through the ctx read tools without content
copies. Grants carry no access mode — the read path is the only consumer
(a child that should write holds ``ctx_write`` in its surface and writes
its OWN scope, never a granted variable).

Revision ID: 0159
Revises: 0158
Create Date: 2026-08-10
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0159"
down_revision: str = "0158"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(r"""
        CREATE TABLE context_variables (
            id                 text PRIMARY KEY,
            account_id         text NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
            scope              text NOT NULL CHECK (scope IN ('session', 'agent')),
            session_id         text REFERENCES sessions(id) ON DELETE CASCADE,
            agent_id           text REFERENCES agents(id) ON DELETE CASCADE,
            name               text NOT NULL,
            kind               text NOT NULL DEFAULT 'data'
                               CHECK (kind IN ('data', 'helper', 'spill', 'digest')),
            description        text NOT NULL DEFAULT '',
            schema_tag         text,
            content            text NOT NULL,
            content_sha256     text NOT NULL,
            content_size_bytes integer NOT NULL,
            metadata           jsonb NOT NULL DEFAULT '{}'::jsonb,
            created_by_type    text,
            created_by_ref     text,
            created_at         timestamptz NOT NULL DEFAULT now(),
            updated_at         timestamptz NOT NULL DEFAULT now(),
            archived_at        timestamptz,
            CHECK (content_size_bytes = octet_length(content)),
            CHECK (
                (scope = 'session' AND session_id IS NOT NULL)
                OR (scope = 'agent' AND agent_id IS NOT NULL AND session_id IS NULL)
            ),
            CONSTRAINT context_variables_created_by_ck CHECK (
                (created_by_type IS NULL AND created_by_ref IS NULL)
                OR (
                    created_by_type IN ('session_actor', 'api_actor')
                    AND created_by_ref IS NOT NULL
                )
            )
        )
    """)
    op.execute("""
        CREATE UNIQUE INDEX context_variables_session_name_uniq
            ON context_variables (session_id, name)
            WHERE archived_at IS NULL AND scope = 'session'
    """)
    op.execute("""
        CREATE UNIQUE INDEX context_variables_agent_name_uniq
            ON context_variables (account_id, agent_id, name)
            WHERE archived_at IS NULL AND scope = 'agent'
    """)
    op.execute(r"""
        CREATE TABLE context_variable_grants (
            session_id          text NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            context_variable_id text NOT NULL REFERENCES context_variables(id) ON DELETE CASCADE,
            created_at          timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (session_id, context_variable_id)
        )
    """)
    # Per-session rlm dispatch ledger (docs/rlm.md): one row per spawning
    # session. ``step_key`` (the assistant seq that issued the launching tool
    # call) scopes the children counter; ``turn_key`` (last_user_seq at spawn)
    # scopes the harvested child-token accumulator. Counters reset when their
    # key advances — enforced in the atomic upsert, not by a sweeper.
    op.execute(r"""
        CREATE TABLE rlm_ledgers (
            session_id       text PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
            step_key         bigint NOT NULL,
            children_spawned integer NOT NULL,
            turn_key         bigint NOT NULL,
            child_tokens     bigint NOT NULL,
            updated_at       timestamptz NOT NULL DEFAULT now()
        )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS rlm_ledgers")
    op.execute("DROP TABLE IF EXISTS context_variable_grants")
    op.execute("DROP TABLE IF EXISTS context_variables")
