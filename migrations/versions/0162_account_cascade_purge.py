"""Durable direct-child cascade purge receipts and inbound-ack ownership.

Cascade purge removes the target account row before host cleanup can finish.
The receipt retains the exact root caller plus a minimal cleanup manifest so a
crash or partial filesystem failure can resume safely without weakening the
caller/direct-child authorization proof. Successful cleanup scrubs the
manifest, retaining only opaque account ids and timestamps for idempotency.

``connector_inbound_acks`` was the only tenant-writable account-id table with
no FK (its account scope was added in 0098). Add a validated CASCADE FK so an
inbound race cannot leave a post-purge ledger orphan.

Revision ID: 0114
Revises: 0113
Create Date: 2026-07-13
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0162"
down_revision: str = "0161"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_RECEIPT_TABLE = "account_cascade_purge_receipts"
_RECEIPT_CANDIDATE = "account_cascade_purge_receipts_0162_candidate"


def _create_receipt_table(table_name: str, *, if_not_exists: bool = False) -> None:
    guard = "IF NOT EXISTS " if if_not_exists else ""
    op.execute(
        f"""
        CREATE TABLE {guard}{table_name} (
            target_account_id   text PRIMARY KEY,
            caller_account_id   text NOT NULL
                REFERENCES accounts(id) ON DELETE CASCADE,
            manifest            jsonb,
            cleanup_attempts    integer NOT NULL DEFAULT 0,
            last_cleanup_error  text,
            created_at          timestamptz NOT NULL DEFAULT now(),
            db_purged_at        timestamptz NOT NULL DEFAULT now(),
            cleanup_completed_at timestamptz,
            updated_at          timestamptz NOT NULL DEFAULT now(),
            CHECK (cleanup_attempts >= 0),
            CHECK (
                (cleanup_completed_at IS NULL AND manifest IS NOT NULL)
                OR (cleanup_completed_at IS NOT NULL AND manifest IS NULL)
            ),
            CHECK (manifest IS NULL OR jsonb_typeof(manifest) = 'object')
        )
        """
    )


def _adopt_or_create_receipt_table() -> None:
    # 0114 in the legacy TrainIQ lineage already created this table. Build an
    # exact candidate and compare PostgreSQL's canonical column/default and
    # constraint definitions before adopting the existing object. This keeps
    # the crossover idempotent without accepting schema drift.
    op.execute(f"DROP TABLE IF EXISTS {_RECEIPT_CANDIDATE}")
    _create_receipt_table(_RECEIPT_CANDIDATE)
    _create_receipt_table(_RECEIPT_TABLE, if_not_exists=True)
    op.execute(
        f"""
        DO $migration$
        DECLARE
            target_signature jsonb;
            candidate_signature jsonb;
        BEGIN
            SELECT jsonb_build_object(
                'columns', (
                    SELECT jsonb_agg(
                        jsonb_build_object(
                            'name', a.attname,
                            'type', format_type(a.atttypid, a.atttypmod),
                            'not_null', a.attnotnull,
                            'default', pg_get_expr(d.adbin, d.adrelid)
                        ) ORDER BY a.attnum
                    )
                    FROM pg_attribute a
                    LEFT JOIN pg_attrdef d
                      ON d.adrelid = a.attrelid AND d.adnum = a.attnum
                    WHERE a.attrelid = to_regclass('public.{_RECEIPT_TABLE}')
                      AND a.attnum > 0 AND NOT a.attisdropped
                ),
                'constraints', (
                    SELECT jsonb_agg(
                        jsonb_build_object(
                            'definition', pg_get_constraintdef(c.oid, true),
                            'validated', c.convalidated
                        )
                                     ORDER BY pg_get_constraintdef(c.oid, true))
                    FROM pg_constraint c
                    WHERE c.conrelid = to_regclass('public.{_RECEIPT_TABLE}')
                )
            ) INTO target_signature;

            SELECT jsonb_build_object(
                'columns', (
                    SELECT jsonb_agg(
                        jsonb_build_object(
                            'name', a.attname,
                            'type', format_type(a.atttypid, a.atttypmod),
                            'not_null', a.attnotnull,
                            'default', pg_get_expr(d.adbin, d.adrelid)
                        ) ORDER BY a.attnum
                    )
                    FROM pg_attribute a
                    LEFT JOIN pg_attrdef d
                      ON d.adrelid = a.attrelid AND d.adnum = a.attnum
                    WHERE a.attrelid = to_regclass('public.{_RECEIPT_CANDIDATE}')
                      AND a.attnum > 0 AND NOT a.attisdropped
                ),
                'constraints', (
                    SELECT jsonb_agg(
                        jsonb_build_object(
                            'definition', pg_get_constraintdef(c.oid, true),
                            'validated', c.convalidated
                        )
                                     ORDER BY pg_get_constraintdef(c.oid, true))
                    FROM pg_constraint c
                    WHERE c.conrelid = to_regclass('public.{_RECEIPT_CANDIDATE}')
                )
            ) INTO candidate_signature;

            IF target_signature IS DISTINCT FROM candidate_signature THEN
                RAISE EXCEPTION
                    'existing {_RECEIPT_TABLE} does not match revision 0162';
            END IF;

            DROP TABLE {_RECEIPT_CANDIDATE};
        END
        $migration$;
        """
    )


def upgrade() -> None:
    # The pre-FK strict purge path could delete an account while leaving its
    # short-lived dedup marker behind. Such rows have no tenant owner to
    # recover and 0098 established deletion (rather than guessing) as the
    # ledger's orphan policy, so clear them before validating the new FK.
    op.execute(
        "DELETE FROM connector_inbound_acks a "
        "WHERE NOT EXISTS (SELECT 1 FROM accounts WHERE id = a.account_id)"
    )
    op.execute(
        """
        DO $migration$
        DECLARE
            existing_constraint record;
        BEGIN
            SELECT c.* INTO existing_constraint
            FROM pg_constraint c
            WHERE c.conrelid = 'connector_inbound_acks'::regclass
              AND c.conname = 'connector_inbound_acks_account_id_fk';

            IF NOT FOUND THEN
                ALTER TABLE connector_inbound_acks
                  ADD CONSTRAINT connector_inbound_acks_account_id_fk
                  FOREIGN KEY (account_id) REFERENCES accounts(id)
                  ON DELETE CASCADE NOT VALID;
            ELSIF existing_constraint.contype <> 'f'
               OR existing_constraint.confrelid <> 'accounts'::regclass
               OR existing_constraint.confdeltype <> 'c'
               OR existing_constraint.confupdtype <> 'a'
               OR existing_constraint.confmatchtype <> 's'
               OR existing_constraint.condeferrable
               OR existing_constraint.condeferred
               OR existing_constraint.conkey <> ARRAY[(
                    SELECT attnum FROM pg_attribute
                    WHERE attrelid = 'connector_inbound_acks'::regclass
                      AND attname = 'account_id'
               )]::smallint[]
               OR existing_constraint.confkey <> ARRAY[(
                    SELECT attnum FROM pg_attribute
                    WHERE attrelid = 'accounts'::regclass AND attname = 'id'
               )]::smallint[] THEN
                RAISE EXCEPTION
                    'existing connector_inbound_acks_account_id_fk does not match revision 0162';
            END IF;
        END
        $migration$;
        """
    )
    op.execute(
        "ALTER TABLE connector_inbound_acks "
        "VALIDATE CONSTRAINT connector_inbound_acks_account_id_fk"
    )
    _adopt_or_create_receipt_table()


def downgrade() -> None:
    op.execute(f"DROP TABLE IF EXISTS {_RECEIPT_CANDIDATE}")
    op.execute("DROP TABLE IF EXISTS account_cascade_purge_receipts")
    op.execute(
        "ALTER TABLE connector_inbound_acks "
        "DROP CONSTRAINT IF EXISTS connector_inbound_acks_account_id_fk"
    )
