"""Admit agentless rlm children to the sessions agentless check (docs/rlm.md).

``sessions_agentless_workflow_child_ck`` (migration 0095) encoded "an
agentless session is a workflow child": ``agent_id IS NOT NULL OR
(parent_run_id IS NOT NULL AND model IS NOT NULL)``. The rlm spawn path
creates a second kind of agentless session — a session-launched generic
child with ``parent_run_id NULL``, ``surface_frozen`` TRUE, and a stamped
``model`` — which the 0095 predicate rejects (found live by the first
``rlm_verify`` smoke; the unit/integration tiers never exercised the real
INSERT against this constraint).

The widened invariant: an agentless session must be a *spawned generic
child* — born with a frozen surface and a stamped model — whether its
launcher was a workflow run or a session. ``surface_frozen AND model``
covers both arms (every workflow generic child is also ``surface_frozen``
TRUE), and ``load_for_session`` still fails closed on a frozen row with no
snapshot.

Revision ID: 0160
Revises: 0159
Create Date: 2026-08-11
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0160"
down_revision: str = "0159"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE sessions DROP CONSTRAINT sessions_agentless_workflow_child_ck")
    op.execute(
        "ALTER TABLE sessions ADD CONSTRAINT sessions_agentless_workflow_child_ck "
        "CHECK (agent_id IS NOT NULL OR (surface_frozen AND model IS NOT NULL))"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE sessions DROP CONSTRAINT sessions_agentless_workflow_child_ck")
    op.execute(
        "ALTER TABLE sessions ADD CONSTRAINT sessions_agentless_workflow_child_ck "
        "CHECK (agent_id IS NOT NULL OR (parent_run_id IS NOT NULL AND model IS NOT NULL))"
    )
