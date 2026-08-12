"""Unique client message ids for idempotent session-message intake.

Pre-feature callers could already place arbitrary ``client_message_id`` values
in event metadata. For duplicate canonical UUIDs, preserve every event and the
lowest-seq event's idempotency key while removing the key from later rows. The
unique index is then built concurrently because ``events`` is live-written.

TrainIQ production previously ran this same contract as revision ``0113`` on a
different migration lineage.  The RLM reconciliation renumbered it to ``0161``
while retaining the index name, so that database arrives here stamped at
``0160`` with the target index already present.  Do not use ``IF NOT EXISTS``:
it would silently trust any same-named object, including an invalid remnant.
Instead, build the exact index under a revision-owned candidate name and compare
PostgreSQL's canonical definitions.  An equivalent valid target is adopted; a
mismatched or invalid same-named object fails closed.  When the target is absent
(fresh database or retry after an interrupted rename), the proven candidate is
renamed into place.

Revision ID: 0113
Revises: 0112
Create Date: 2026-07-13
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0161"
down_revision: str = "0160"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

INDEX_NAME = "events_user_client_message_id_uidx"
CANDIDATE_INDEX_NAME = "events_user_client_message_id_uidx_0161_candidate"

CREATE_CANDIDATE_INDEX = f"""
CREATE UNIQUE INDEX CONCURRENTLY {CANDIDATE_INDEX_NAME}
    ON events (
        account_id,
        session_id,
        ((data->'metadata'->>'client_message_id'))
    )
 WHERE kind = 'message'
   AND role = 'user'
   AND data->'metadata'->>'client_message_id'
       ~ '^[0-9a-f]{{8}}-[0-9a-f]{{4}}-[0-9a-f]{{4}}-[0-9a-f]{{4}}-[0-9a-f]{{12}}$'
"""


def upgrade() -> None:
    op.execute(
        """
        WITH ranked AS (
            SELECT id,
                   row_number() OVER (
                       PARTITION BY account_id, session_id,
                                    (data->'metadata'->>'client_message_id')
                       ORDER BY seq, id
                   ) AS duplicate_rank
              FROM events
             WHERE kind = 'message'
               AND role = 'user'
               AND data->'metadata'->>'client_message_id'
                   ~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        )
        UPDATE events AS event
           SET data = event.data #- '{metadata,client_message_id}'
          FROM ranked
         WHERE event.id = ranked.id
           AND ranked.duplicate_rank > 1
        """
    )
    with op.get_context().autocommit_block():
        # ``CONCURRENTLY`` cannot run inside a transaction. Clear only this
        # revision's reserved retry artifact, then build a canonical comparison
        # object rather than trusting a same-named production index.
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {CANDIDATE_INDEX_NAME}")
        op.execute(CREATE_CANDIDATE_INDEX)
        op.execute(
            f"""
            DO $$
            DECLARE
                target_oid regclass := to_regclass('public.{INDEX_NAME}');
                candidate_oid regclass := to_regclass('public.{CANDIDATE_INDEX_NAME}');
                target_definition text;
                candidate_definition text;
                target_valid boolean;
                target_ready boolean;
            BEGIN
                IF candidate_oid IS NULL THEN
                    RAISE EXCEPTION '0161 candidate index is missing';
                END IF;

                IF target_oid IS NULL THEN
                    EXECUTE 'ALTER INDEX {CANDIDATE_INDEX_NAME} RENAME TO {INDEX_NAME}';
                    RETURN;
                END IF;

                SELECT indisvalid, indisready, pg_get_indexdef(indexrelid)
                  INTO target_valid, target_ready, target_definition
                  FROM pg_index
                 WHERE indexrelid = target_oid;
                SELECT pg_get_indexdef(indexrelid)
                  INTO candidate_definition
                  FROM pg_index
                 WHERE indexrelid = candidate_oid;

                IF NOT target_valid OR NOT target_ready THEN
                    RAISE EXCEPTION '{INDEX_NAME} exists but is not valid and ready';
                END IF;

                target_definition := replace(target_definition, '{INDEX_NAME}', '<index>');
                candidate_definition := replace(
                    candidate_definition,
                    '{CANDIDATE_INDEX_NAME}',
                    '<index>'
                );
                IF target_definition IS DISTINCT FROM candidate_definition THEN
                    RAISE EXCEPTION '{INDEX_NAME} exists with a mismatched definition';
                END IF;
            END
            $$
            """
        )
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {CANDIDATE_INDEX_NAME}")


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {INDEX_NAME}")
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {CANDIDATE_INDEX_NAME}")
