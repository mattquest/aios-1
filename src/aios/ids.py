"""Prefixed Crockford-base32 ULIDs.

Every aios resource id has the form ``<prefix>_<26-char-ULID>``. ULIDs are
time-ordered, so DB indexes on append-heavy tables (the events table in
particular) stay tight. The prefix makes ids self-describing in logs and
error messages without an extra type lookup.

Example::

    >>> make_id("agent")
    'agent_01HQR2K7VXBZ9MNPL3WYCT8F'
"""

from __future__ import annotations

from typing import Final, Literal

from ulid import ULID

# Canonical prefixes for every resource kind. Centralized so callers don't
# accidentally use a typo'd prefix that the DB would happily accept.
AGENT: Final = "agent"
ENVIRONMENT: Final = "env"
SESSION: Final = "sess"
EVENT: Final = "evt"
CREDENTIAL: Final = "cred"
VAULT: Final = "vlt"
VAULT_CREDENTIAL: Final = "vcr"
SKILL: Final = "skl"
CONNECTION: Final = "conn"
SESSION_TEMPLATE: Final = "stpl"
# Connector subsystem (#328 PR 2+). The aios_connectors module uses
# these prefixes to mint ids via ``make_id``; the PR 2 migration
# also backfills binding ids prefixed with ``bnd_`` (but with a
# random-hex body, not a ULID) — backfilled rows are valid ``text``
# PKs that won't survive ``split_id`` parsing.  No current call site
# does that on a binding id; if one is added later, generate ULIDs
# in the migration or relax ``split_id``.
BINDING: Final = "bnd"
RUNTIME_TOKEN: Final = "rtk"
MEMORY_STORE: Final = "memstore"
MEMORY: Final = "mem"
MEMORY_VERSION: Final = "memver"
GITHUB_REPOSITORY: Final = "ghrepo"
FILE: Final = "file"
MANAGEMENT_CALL: Final = "mgmt"
ACCOUNT: Final = "acc"
ACCOUNT_KEY: Final = "acckey"
SCHEDULED_TASK: Final = "sched"
# Triggers (#818) — the generalization of scheduled tasks (a `source` that
# fires + an `action` that runs). New rows mint `trig_`; legacy `sched_` ids
# from the pre-rename table remain valid (both kept in `_PREFIXES`), so
# `split_id` works on old rows without a data migration.
TRIGGER: Final = "trig"
# Per-fire audit row for a trigger (#819). References triggers by PLAIN id —
# one-shot rows are deleted before firing, so the audit must outlive the
# trigger row (no FK; see migration 0086).
TRIGGER_RUN: Final = "trun"
# Short-lived server-side state for an in-progress interactive OAuth
# authorization-code flow (vault credential "Connect"). Rows are pruned on
# expiry — see ``oauth_flows`` (migration 0061).
OAUTH_FLOW: Final = "oaf"
# Workflows: a deterministic-Python orchestrator (the dual of an agent).
# ``workflows`` are versioned definitions (updated in place); ``wf_runs`` are durable
# execution instances; ``wf_run_events`` is each run's append-only journal.
# (``wf_run_signals`` has a composite PK and mints no id.) See migration 0064.
WORKFLOW: Final = "wf"
WORKFLOW_RUN: Final = "wfr"
WORKFLOW_EVENT: Final = "wfe"
# Request edge (#1123/#1128): the id correlating a request to its response.
# Run→child spawns reuse the workflow ``call_key`` as the request_id; the API
# caller (#1128) has no call_key, so it mints a fresh ``req_`` id here.
REQUEST: Final = "req"
# Per-account model-provider config: an encrypted api_key + plaintext
# api_base for one (account, provider) pair, resolved nearest-ancestor-wins
# up the account tree at model-call time. See migration 0140.
MODEL_PROVIDER: Final = "mp"
# RLM context variables (docs/rlm.md): named out-of-context content handles,
# session- or agent-scoped. See migration 0159.
CONTEXT_VARIABLE: Final = "ctxv"

_PREFIXES: Final = frozenset(
    {
        AGENT,
        ENVIRONMENT,
        SESSION,
        EVENT,
        CREDENTIAL,
        VAULT,
        VAULT_CREDENTIAL,
        SKILL,
        CONNECTION,
        SESSION_TEMPLATE,
        MEMORY_STORE,
        MEMORY,
        MEMORY_VERSION,
        GITHUB_REPOSITORY,
        FILE,
        BINDING,
        RUNTIME_TOKEN,
        MANAGEMENT_CALL,
        ACCOUNT,
        ACCOUNT_KEY,
        SCHEDULED_TASK,
        TRIGGER,
        TRIGGER_RUN,
        OAUTH_FLOW,
        WORKFLOW,
        WORKFLOW_RUN,
        WORKFLOW_EVENT,
        REQUEST,
        MODEL_PROVIDER,
        CONTEXT_VARIABLE,
    }
)


def make_id(prefix: str, *, body: bytes | None = None) -> str:
    """Generate a prefixed ULID id for ``prefix``.

    With no ``body``, returns a fresh random, time-ordered ULID. ``body``
    (exactly 16 bytes) forces a **deterministic** ULID — Crockford-base32
    encoded to the same 26-char body ``split_id`` accepts. Used for
    content-addressed ids (e.g. a workflow's deterministic child session id
    folded from ``(run_id, call_key)``) so a replayed spawn reproduces the
    same id rather than minting a duplicate.

    Raises ``ValueError`` if ``prefix`` isn't one of the canonical aios
    resource prefixes (catches typos before they reach the DB), or if ``body``
    is given but isn't exactly 16 bytes.
    """
    if prefix not in _PREFIXES:
        raise ValueError(f"unknown id prefix {prefix!r}; expected one of {sorted(_PREFIXES)}")
    if body is not None:
        if len(body) != 16:
            raise ValueError(f"deterministic id body must be exactly 16 bytes, got {len(body)}")
        return f"{prefix}_{ULID(body)}"
    return f"{prefix}_{ULID()}"


def is_run_owner_id(owner_id: str) -> bool:
    """Is ``owner_id`` a workflow-run sandbox owner (``wfr_…``)?

    The intrinsic owner-kind discriminator the sandbox registry routes teardown
    on (#995): its ``_handles`` map holds both session (``sess_…``) and
    workflow-run (``wfr_…``) owners, whose teardowns differ (a session snapshots
    its rootfs + stops proxies; a run is a bare ephemeral destroy). Centralizing
    the prefix test here keeps the run-vs-session fork a single source of truth —
    so a future owner prefix added to the map can't silently inherit the session
    path by an inlined ``startswith`` going stale at one call site.
    """
    return owner_id.startswith(f"{WORKFLOW_RUN}_")


def split_id(value: str) -> tuple[str, str]:
    """Split a prefixed id into ``(prefix, ulid)``.

    Useful for assertions and logging. Raises ``ValueError`` for malformed ids.
    """
    prefix, sep, ulid_part = value.partition("_")
    if not sep or prefix not in _PREFIXES or len(ulid_part) != 26:
        raise ValueError(f"malformed prefixed id: {value!r}")
    return prefix, ulid_part


def servicer_kind(value: str) -> Literal["session", "run"]:
    """The task servicer kind of an id — ``session`` (``sess_…``) or ``run`` (``wfr_…``).

    The single source for "what kind of servicer is this id," shared by the unified
    awaiter (``GET /v1/tasks/{task_id}/await``) and the trace CLI to route on a
    servicer reference. Raises ``ValueError`` for a malformed id or a non-servicer prefix
    (an agent/vault/… id is not an awaitable or traceable servicer).
    """
    prefix, _ = split_id(value)  # raises ValueError on a malformed id
    if prefix == WORKFLOW_RUN:
        return "run"
    if prefix == SESSION:
        return "session"
    raise ValueError(f"{value!r} is not a servicer id (expected a session or run id)")
