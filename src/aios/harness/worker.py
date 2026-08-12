"""Worker process entrypoint.

``aios serve worker`` runs :func:`worker_main` in an asyncio event loop. It:

1. Configures structlog
2. Acquires a Postgres advisory lock to refuse a duplicate worker
3. Opens the asyncpg pool
4. Constructs the libsodium CryptoBox
5. Creates the SandboxRegistry, InflightToolRegistry, and McpSessionPool
6. Stashes globals on :mod:`aios.harness.runtime`
7. Opens the procrastinate connector
8. Sweeps orphan attachments, reaps stalled jobs, wakes sessions needing inference
9. Reaps orphaned sandbox containers
10. Starts the container idle-TTL reaper, periodic sweep, interrupt listener, and liveness heartbeat
11. Starts ``app.run_worker_async`` which blocks until SIGTERM/SIGINT

Shutdown: procrastinate's signal handlers stop accepting new jobs and wait
for in-flight jobs. The ``finally`` block then cancels in-flight tool tasks,
releases all containers, closes MCP sessions, and closes connections.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import asyncpg

import aios.harness.tasks
import aios.tools  # noqa: F401  — side-effect: register built-in tools
from aios.config import get_settings
from aios.crypto.vault import CryptoBox
from aios.db import queries
from aios.db.listen import (
    listen_for_github_clone_breaker_clear,
    listen_for_mcp_evict_vault,
    listen_for_session_interrupts,
)
from aios.db.pool import LISTENER_TCP_KEEPALIVE_SETTINGS, create_pool, normalize_dsn
from aios.harness import runtime
from aios.harness.attachment_gc import sweep_orphan_attachments
from aios.harness.exit_diagnostics import install_exit_diagnostics
from aios.harness.host_dir_reaper import sweep_host_dirs
from aios.harness.inflight_tool_registry import InflightToolRegistry
from aios.harness.production_watchdogs import run_production_watchdogs
from aios.harness.reclaimable_prune import sweep_reclaimable_ephemera
from aios.harness.scheduler import _LISTEN_RECONNECT_BACKOFF_SECONDS, event_driven_scheduler
from aios.harness.sweep import (
    reap_stalled_jobs,
    wake_sessions_needing_inference,
)
from aios.harness.trigger_runner import sweep_trigger_fires
from aios.harness.workspace_reaper import sweep_archived_workspaces
from aios.jobs.app import WORKER_POLLED_QUEUES as WORKER_POLLED_QUEUES
from aios.jobs.app import app as procrastinate_app
from aios.logging import configure_logging, get_logger
from aios.mcp.pool import McpSessionPool
from aios.retirements.boot_gate import (
    RetirementsNotAdmissible,
    assert_retirements_admissible,
)
from aios.sandbox.backends import select_sandbox_backend
from aios.sandbox.github_clone_breaker import GithubCloneBreaker
from aios.sandbox.network import ensure_sandbox_network, is_running_in_container
from aios.sandbox.registry import GcPressureResult, SandboxRegistry
from aios.sandbox.tool_broker import ToolBroker
from aios.sandbox.workspace_ownership import repair_workspace_ownership


def _consume_snapshot_pressure(registry: SandboxRegistry, pressure: GcPressureResult) -> None:
    """Apply one GC pressure report to subsequent provisioning admission."""
    registry.set_provisioning_pressure(pressure)
    if pressure.pressured:
        get_logger("aios.worker").error(
            "worker.sandbox_capacity_pressure_alarm",
            pool_used_bytes=pressure.pool_used_bytes,
            pool_budget_bytes=pressure.pool_budget_bytes,
            pressured_accounts=sorted(pressure.pressured_accounts),
        )


# Hashed (via Postgres ``hashtextextended($1, 0)``) into the 64-bit
# advisory-lock key enforcing the worker-process singleton. The string
# is a historical magic value, preserved verbatim so a rolling deploy
# computes the same lock number across old and new workers.
_WORKER_SINGLETON_LOCK_KEY_TEXT = "aios_worker_connector_supervisor"

# Path the worker touches periodically to signal liveness; the Dockerfile
# HEALTHCHECK reads its mtime. tmpfs in containers, so touch/unlink are
# sub-microsecond and don't justify ``asyncio.to_thread`` (which would
# add more latency than it saves).
#
# ``/var/run`` is only writable by the container user, so on the host
# (lean mode, ``uv run aios worker``) touching it raises EACCES. That
# both spammed the log every 15 s and left the liveness file missing —
# anything modelling the compose HEALTHCHECK would have seen a dead
# worker. So pick the path at startup: the container path in a
# container (the Dockerfile HEALTHCHECK stats it), a user-writable temp
# path otherwise.
_HEARTBEAT_FILENAME = "aios-worker-alive"
_CONTAINER_HEARTBEAT_FILE = Path("/var/run") / _HEARTBEAT_FILENAME
_HEARTBEAT_INTERVAL_SECONDS = 15

# How long the worker startup loops between boot-admission proof attempts while
# the DB is behind / unreachable. Matches the api gate's cadence.
_BOOT_GATE_RETRY_SECONDS = 2.0

# The harness ``@app.task`` handlers that MUST be registered before this worker
# consumes jobs (#1476). Registration is a side effect of ``import
# aios.harness.tasks`` at the top of this module; the worker-boot backstop below
# names each one so a dropped import fails loud. A bare truthiness check on
# ``app.tasks`` is VACUOUS: procrastinate's ``App.__init__`` unconditionally
# registers its own builtin ``remove_old_jobs`` task, so ``app.tasks`` is never
# empty even when the harness import is missing — the exact degraded state this
# guard exists to catch. Keep in sync with the ``@app.task(name=...)`` decls in
# ``aios.harness.tasks``.
_REQUIRED_HARNESS_TASKS = {
    "harness.wake_session",
    "harness.wake_workflow",
    "harness.run_trigger",
}


@dataclass(frozen=True, slots=True)
class StartupRecoveryResult:
    """Return value of :func:`run_startup_recovery` — the composed counts
    from every startup-recovery step, so the boot log line and tests can
    assert on the whole pass without unrolling the composition.
    """

    reaped_jobs: int
    repaired_ghosts: int
    woken_sessions: int
    woken_runs: int


async def run_startup_recovery(
    pool: asyncpg.Pool[Any],
    inflight_tool_registry: InflightToolRegistry,
) -> StartupRecoveryResult:
    """The worker's startup-recovery sequence (issue #1757).

    Extracted from :func:`worker_main` so it is a unit an integration test can
    call directly against a seeded Postgres, rather than only being
    exercisable by booting a full worker process. Order is load-bearing —
    each step is a precondition for the next:

    1. :func:`reap_stalled_jobs` — fail every ``doing`` procrastinate job left
       by the dead predecessor (the singleton lock has handed off, so every
       ``doing`` row is orphaned by construction). MUST run before step 3: a
       stalled ``wake_session`` job holds ``lock=session_id`` /
       ``queueing_lock=session_id``, so until it is failed and its lock
       released, a fresh wake for that session raises procrastinate's
       ``AlreadyEnqueued`` and is silently swallowed by
       :func:`aios.jobs.app.defer_wake` (issue #147) — the session would stay
       wedged until the *next* sweep tick found it again.
    2. :func:`reset_inflight_harvests` — clear the in-process
       model-dispatch-harvest key set. Every harvest task from the
       predecessor's process is provably gone (same singleton-lock handoff),
       so a stale key here would make step 3's crash-recovery re-park (#1635)
       treat a genuinely lost park as still-serviced and skip it. MUST run
       before step 3 for the same reason.
    3. :func:`wake_sessions_needing_inference` — repair ghost tool calls
       (including re-parking the model-dispatch park via
       :func:`repark_stranded_model_dispatch`) and defer a wake for every
       session now needing inference. Freshly-unblocked sessions (step 1) and
       freshly-reparked model-dispatch parks (step 2) are picked up in THIS
       pass because it runs after both.
    4. :func:`aios.workflows.sweep.wake_runs_needing_step` — the workflow-run
       analog of step 3; independent of the session-side steps.

    Called once at worker boot (before this process consumes any job) and
    exposed here so the integration test can seed the full post-kill residue
    (a ghost tool call, a wedged ``doing`` job, a stranded model-dispatch
    park) and assert the combined convergence in one pass.
    """
    from aios.harness.model_workflow import reset_inflight_harvests
    from aios.jobs.app import app as procrastinate_app
    from aios.workflows.sweep import wake_runs_needing_step

    reaped_jobs = await reap_stalled_jobs(procrastinate_app.job_manager)
    reset_inflight_harvests()
    sweep = await wake_sessions_needing_inference(pool, inflight_tool_registry)
    woken_runs = await wake_runs_needing_step(pool)
    return StartupRecoveryResult(
        reaped_jobs=reaped_jobs,
        repaired_ghosts=sweep.repaired_ghosts,
        woken_sessions=sweep.woken_sessions,
        woken_runs=woken_runs,
    )


async def _await_retirements_admissible(
    pool: asyncpg.Pool[Any],
    log: Any,
    latch: asyncio.Event,
) -> None:
    """Loop the fail-closed boot-admission gate until the DB is provably safe.

    Refuses to return — keeping the worker pre-heartbeat, so the container
    HEALTHCHECK reports unhealthy — until
    :func:`assert_retirements_admissible` passes (#1575). The ``latch`` is the
    worker's supervision/shutdown event: if it fires while we're waiting
    (SIGTERM, a supervised task died), abandon the wait so shutdown isn't
    blocked behind an unreachable DB.
    """
    logged_wait = False
    while not latch.is_set():
        try:
            await assert_retirements_admissible(pool)
            return
        except RetirementsNotAdmissible as exc:
            if not logged_wait:
                log.warning("worker.boot_gate.refusing_readiness", reason=str(exc))
                logged_wait = True
            with contextlib.suppress(asyncio.TimeoutError):
                # Wake early if shutdown fires; otherwise retry after the delay.
                await asyncio.wait_for(latch.wait(), timeout=_BOOT_GATE_RETRY_SECONDS)


def _resolve_heartbeat_path() -> Path:
    """Return the liveness file path the worker should touch.

    In a container, the Dockerfile HEALTHCHECK stats
    ``/var/run/aios-worker-alive``, so that exact path is required. On
    the host, ``/var/run`` is not writable by the running user, so fall
    back to a user-writable temp directory (honoring ``$TMPDIR``). This
    keeps the periodic touch from EACCES-spamming the log and ensures
    the file actually exists for anything consuming it.
    """
    if is_running_in_container():
        return _CONTAINER_HEARTBEAT_FILE
    # Resolve ``$TMPDIR`` here rather than via ``tempfile.gettempdir()``,
    # which caches its first result process-wide and so wouldn't reflect
    # the environment the worker was actually launched with.
    tmpdir = os.environ.get("TMPDIR") or tempfile.gettempdir()
    return Path(tmpdir) / _HEARTBEAT_FILENAME


_HEARTBEAT_FILE = _resolve_heartbeat_path()


class _SupervisedTaskFailure(TypedDict):
    exception: BaseException | None


def _supervise(
    task: asyncio.Task[Any],
    *,
    latch: asyncio.Event,
    fatal: _SupervisedTaskFailure,
) -> None:
    """Attach fail-stop supervision to one long-lived worker task."""

    def _done(done: asyncio.Task[Any]) -> None:
        if latch.is_set() or done.cancelled():
            return
        task_name = done.get_name()
        try:
            exc = done.exception()
        except asyncio.CancelledError:
            return
        if exc is None:
            exc = RuntimeError(f"supervised task {task_name!r} returned cleanly")
        log = get_logger("aios.worker")
        log.error(
            "worker.supervised_task_died",
            task_name=task_name,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        with contextlib.suppress(OSError):
            _HEARTBEAT_FILE.unlink(missing_ok=True)
        fatal["exception"] = exc
        latch.set()

    task.add_done_callback(_done)


def _make_advisory_lock_termination_listener(
    *,
    latch: asyncio.Event,
    fatal: _SupervisedTaskFailure,
) -> Any:
    """Return a fail-stop callback for singleton lock connection termination."""

    def _terminated(_conn: asyncpg.Connection[Any]) -> None:
        if latch.is_set():
            return
        exc = RuntimeError("singleton advisory lock connection lost")
        log = get_logger("aios.worker")
        log.error(
            "worker.advisory_lock_lost",
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        with contextlib.suppress(OSError):
            _HEARTBEAT_FILE.unlink(missing_ok=True)
        fatal["exception"] = exc
        latch.set()

    return _terminated


async def _cancel_and_drain(task: asyncio.Task[Any]) -> None:
    """Cancel a background task and suppress prior task exceptions after logging."""
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception:
        get_logger("aios.worker").exception(
            "worker.teardown_task_error",
            task_name=task.get_name(),
        )


def _make_worker_id() -> str:
    from ulid import ULID

    return f"worker_{ULID()}"


async def worker_main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    log = get_logger("aios.worker")
    install_exit_diagnostics(log)

    # Single-instance guard.  Two `aios worker` processes against the same
    # database would compete over per-worker state that procrastinate's
    # session-scoped job lock doesn't cover — in-memory sandbox container
    # stewardship, attachment-staging atomicity, the startup orphan sweep.
    # So we refuse to boot a second worker by holding a session-scoped
    # advisory lock on a dedicated connection.  Pool-borrowed connections
    # release the lock on return, so the lock conn is intentionally NOT
    # in the pool.
    lock_conn = await _acquire_worker_lock(settings.db_url, log)
    if lock_conn is None:
        sys.exit(1)

    # Everything below holds resources that need ordered teardown; the
    # try/finally wraps the entire construction so a partial-startup
    # failure (e.g. ``create_pool`` racing a temporarily unreachable DB)
    # still releases the advisory lock and any already-built resource.
    pool: asyncpg.Pool[Any] | None = None
    sandbox_registry: SandboxRegistry | None = None
    inflight_tool_registry: InflightToolRegistry | None = None
    mcp_session_pool: McpSessionPool | None = None
    github_clone_breaker: GithubCloneBreaker | None = None
    tool_broker: ToolBroker | None = None
    procrastinate_opened = False
    sweep_task: asyncio.Task[None] | None = None
    invariant_sweep_task: asyncio.Task[None] | None = None
    interrupt_task: asyncio.Task[None] | None = None
    mcp_evict_task: asyncio.Task[None] | None = None
    github_clone_breaker_clear_task: asyncio.Task[None] | None = None
    heartbeat_task: asyncio.Task[None] | None = None
    scheduler_task: asyncio.Task[None] | None = None
    watchdog_task: asyncio.Task[None] | None = None
    supervised_latch = asyncio.Event()
    supervised_failure: _SupervisedTaskFailure = {"exception": None}
    lock_conn.add_termination_listener(
        _make_advisory_lock_termination_listener(
            latch=supervised_latch,
            fatal=supervised_failure,
        )
    )

    try:
        pool = await create_pool(settings.db_url, max_size=settings.db_pool_max_size)
        crypto_box = CryptoBox.from_base64(settings.vault_key.get_secret_value())
        async with pool.acquire() as conn:
            await queries.audit_credentialless_root(conn)
        sandbox_registry = SandboxRegistry(backend=select_sandbox_backend(settings))
        inflight_tool_registry = InflightToolRegistry()
        mcp_session_pool = McpSessionPool()
        github_clone_breaker = GithubCloneBreaker()
        if settings.sandbox_backend != "disabled":
            await ensure_sandbox_network()
        tool_broker = ToolBroker(socket_path=settings.tool_broker_socket_path)
        await tool_broker.start()
        for broker_task in tool_broker.serve_tasks():
            _supervise(broker_task, latch=supervised_latch, fatal=supervised_failure)

        # Register the connector subsystem's ToolProvider impl against the
        # Protocol slot from PR 3 (#328). The harness reaches the
        # subsystem via this registration plus a function-scoped import in
        # ``services/inbound.handle_inbound`` for the resolver; see
        # ``aios_connectors/__init__.py`` for the full module-boundary
        # contract.
        from aios_connectors.providers import SubsystemToolProvider

        runtime.pool = pool
        runtime.crypto_box = crypto_box
        runtime.worker_id = _make_worker_id()
        runtime.sandbox_registry = sandbox_registry
        runtime.inflight_tool_registry = inflight_tool_registry
        runtime.mcp_session_pool = mcp_session_pool
        runtime.github_clone_breaker = github_clone_breaker
        runtime.tool_broker = tool_broker
        runtime.tool_provider = SubsystemToolProvider()

        # Repair pre-existing root-owned shared-workspace dirs (#959) before
        # any provisioning job can run. No-op unless the worker is root.
        # to_thread: synchronous lstat/chown walk must not block the event loop.
        repaired = await asyncio.to_thread(repair_workspace_ownership)
        if repaired:
            log.info("worker.workspace_ownership_repaired", count=repaired)

        # Probe the actual memory-store mount's ctime granularity (#1748 §6):
        # the bash memory-reconcile stat-prefilter's soundness rests on
        # sub-hot-window ctime granularity. That FS is slated to potentially
        # move off the shared postgres volume onto something coarser
        # (overlayfs/tmpfs/NFS), so this is a runtime probe, not a build-time
        # human check. Cached for the process lifetime; fails closed (forces
        # every reconcile candidate to be hashed) if the mount reports no
        # ctime or a granule at/above the hot window. to_thread: two
        # synchronous writes + stats must not block the event loop.
        from aios.sandbox.volumes import memory_stores_root
        from aios.tools.bash_memory_reconcile import (
            probe_mount_ctime_granularity,
            set_prefilter_state,
        )

        prefilter_state = await asyncio.to_thread(
            probe_mount_ctime_granularity, memory_stores_root()
        )
        set_prefilter_state(prefilter_state)
        if not prefilter_state.enabled:
            log.warning(
                "worker.memory_reconcile_prefilter_disabled",
                observed_granule_ns=prefilter_state.observed_granule_ns,
            )

        # Fail-closed boot-admission gate (#1575): before this worker does ANY
        # work — before it opens procrastinate, sweeps, or touches the liveness
        # heartbeat that the container HEALTHCHECK watches — prove the live DB
        # is safe for this code (alembic at/past every contract_rev AND zero
        # live residue on every registered surface). A behind / residue-bearing
        # / unreachable DB loops here, keeping the heartbeat file absent, so the
        # new container reports unhealthy and the old one keeps serving under a
        # rolling deploy. The proof re-runs on EVERY boot, including raw
        # restores, and is never cached.
        await _await_retirements_admissible(pool, log, supervised_latch)
        log.info("worker.boot_gate.admitted")

        # Registration backstop (#1476): tasks are registered as a side effect
        # of ``import aios.harness.tasks`` at the top of this module. If that
        # import is ever dropped, the worker would boot with an empty task
        # registry and every job would fail silently at job-lookup time. The
        # test-substrate assertion (test_wf_sweep.py) imports the tasks module
        # itself, so it can't catch a worker entrypoint missing the import —
        # this boot assert is the only check on the production path, converting
        # that silent failure into a loud one at startup.
        missing = _REQUIRED_HARNESS_TASKS - set(procrastinate_app.tasks)
        assert not missing, (
            f"harness tasks not registered: {missing} — the worker must "
            "`import aios.harness.tasks` before consuming jobs"
        )

        await procrastinate_app.open_async()
        procrastinate_opened = True

        # Sweep orphan attachments at startup before any inbound POST can
        # land.  An attachment-staging write that's in-flight elsewhere
        # is invisible to the events table until its dedup transaction
        # commits, so commit-happens-before-sweep ordering is governed
        # by Postgres snapshot isolation rather than wall-clock.
        deleted_attachments = await sweep_orphan_attachments(pool)
        if deleted_attachments:
            log.info("worker.reaped_orphan_attachments", count=deleted_attachments)

        log.info(
            "worker.startup",
            worker_id=runtime.worker_id,
            concurrency=settings.worker_concurrency,
        )

        # Startup sweep — see :func:`run_startup_recovery` for the composed
        # sequence and why its ordering (reap → reset in-flight harvests →
        # wake sessions → wake runs) is load-bearing.
        recovery = await run_startup_recovery(pool, inflight_tool_registry)
        if recovery.woken_sessions or recovery.repaired_ghosts:
            log.info(
                "worker.startup_sweep",
                woken=recovery.woken_sessions,
                repaired_ghosts=recovery.repaired_ghosts,
            )
        if recovery.woken_runs:
            log.info("worker.startup_sweep.workflows", woken_runs=recovery.woken_runs)

        # Start the snapshot GC reconciler (durable session sandboxes). Its
        # immediate first tick replaces the old boot-time orphan reap: rather
        # than removing every managed container at boot (which lost their
        # filesystems), it salvages crash corpses and reconciles images +
        # pointers against store truth, then repeats hourly. Boot is not
        # blocked — a session waking mid-reconcile salvages its own corpse
        # inline under its own lock.
        sandbox_gc_task = sandbox_registry.start_gc(
            pool,
            pressure_callback=lambda pressure: _consume_snapshot_pressure(
                sandbox_registry, pressure
            ),
        )
        _supervise(sandbox_gc_task, latch=supervised_latch, fatal=supervised_failure)

        # Start container + MCP-pool idle-TTL reapers.
        sandbox_reaper_task = sandbox_registry.start_reaper(
            idle_timeout=settings.container_idle_timeout_seconds
        )
        _supervise(sandbox_reaper_task, latch=supervised_latch, fatal=supervised_failure)
        egress_refresh_task = sandbox_registry.start_egress_refresh()
        _supervise(egress_refresh_task, latch=supervised_latch, fatal=supervised_failure)
        mcp_reaper_task = mcp_session_pool.start_reaper(
            idle_timeout=settings.mcp_pool_idle_timeout_seconds
        )
        _supervise(mcp_reaper_task, latch=supervised_latch, fatal=supervised_failure)

        # Start the idle host scratch-dir reaper (#1192): GC the
        # ``_session_repos`` working-tree clones and ``_runs`` per-run scratch
        # that otherwise grow monotonically with session/run count (the
        # 2026-06-16 floodgates disk-fill). Immediate first sweep at boot, then
        # periodic; honours the ``host_dir_reaper_enabled`` kill-switch.
        host_dir_reaper_task = asyncio.create_task(
            _host_dir_reaper_loop(pool),
            name="host_dir_reaper",
        )
        _supervise(host_dir_reaper_task, latch=supervised_latch, fatal=supervised_failure)

        # Start the uncorrelated memory-reconcile audit (#1748 §Uncorrelated
        # detector): a low-rate, out-of-band full-hash disk-vs-DB divergence
        # check that is the falsifying backstop for the bash memory-reconcile
        # stat-prefilter. Immediate first sweep at boot so the detector is
        # proven ringing early; honours ``memory_reconcile_audit_enabled``.
        memory_reconcile_audit_task = asyncio.create_task(
            _memory_reconcile_audit_loop(pool),
            name="memory_reconcile_audit",
        )
        _supervise(memory_reconcile_audit_task, latch=supervised_latch, fatal=supervised_failure)

        # Start the archived-session workspace reaper (#40 — the 45G hole): GC
        # the per-session ``/workspace`` host dir of archived sessions, which
        # the #1192 reaper does NOT cover. Ships DARK
        # (``workspace_reaper_enabled`` defaults OFF) — it deletes real working
        # files, not reconstructible clones, so it is enabled only after review.
        workspace_reaper_task = asyncio.create_task(
            _workspace_reaper_loop(pool),
            name="workspace_reaper",
        )
        _supervise(workspace_reaper_task, latch=supervised_latch, fatal=supervised_failure)

        # Start the reclaimable-ephemera prune sweep (T6 #1461): age-based GC
        # of terminal+archived runs (and their WfRunEvent journals) plus
        # archived agent/skill/workflow definitions no live session/run pins —
        # the unbounded DB-row growth the never-prune-anything regime left
        # unaddressed. Immediate first sweep at boot, then periodic; honours
        # the ``reclaimable_prune_enabled`` kill-switch. NEVER touches memory,
        # referenced history, live-pinned versions, or accounts (the sacred set).
        reclaimable_prune_task = asyncio.create_task(
            _reclaimable_prune_loop(pool),
            name="reclaimable_prune",
        )
        _supervise(reclaimable_prune_task, latch=supervised_latch, fatal=supervised_failure)

        # Start periodic sweep (every 30s).
        sweep_task = asyncio.create_task(
            _periodic_sweep(pool, inflight_tool_registry, interval=30),
            name="periodic_sweep",
        )
        _supervise(sweep_task, latch=supervised_latch, fatal=supervised_failure)

        invariant_sweep_task = asyncio.create_task(
            _periodic_invariant_sweep(pool, interval=settings.invariant_sweep_interval_seconds),
            name="periodic_invariant_sweep",
        )
        _supervise(invariant_sweep_task, latch=supervised_latch, fatal=supervised_failure)

        interrupt_task = asyncio.create_task(
            _run_interrupt_listener(settings.db_url, inflight_tool_registry, pool),
            name="interrupt_listener",
        )
        _supervise(interrupt_task, latch=supervised_latch, fatal=supervised_failure)

        # Listen for operator credential-rotation NOTIFYs and evict the
        # rotated vault's pooled MCP sessions so the new secret propagates
        # immediately, instead of waiting out the idle TTL (#1030).
        mcp_evict_task = asyncio.create_task(
            _run_mcp_evict_listener(settings.db_url),
            name="mcp_evict_listener",
        )
        _supervise(mcp_evict_task, latch=supervised_latch, fatal=supervised_failure)

        # Listen for github-repo token-rotation NOTIFYs and clear the
        # rotated attachment's clone-breaker state so the fixed credential
        # re-probes on the very next provision (#1720).
        github_clone_breaker_clear_task = asyncio.create_task(
            _run_github_clone_breaker_clear_listener(settings.db_url),
            name="github_clone_breaker_clear_listener",
        )
        _supervise(
            github_clone_breaker_clear_task, latch=supervised_latch, fatal=supervised_failure
        )

        # Start event-driven scheduler. Sleeps until the next due
        # ``next_fire``, woken early by NOTIFY on
        # ``aios_scheduled_tasks_due`` (insert/delete or
        # scheduling-relevant UPDATE on ``triggers`` via the trigger from
        # migration 0080; channel name kept byte-identical). On wake, claims
        # due rows and defers ``run_trigger`` jobs.
        scheduler_task = asyncio.create_task(
            event_driven_scheduler(pool, settings.db_url),
            name="scheduler",
        )
        _supervise(scheduler_task, latch=supervised_latch, fatal=supervised_failure)

        watchdog_task = asyncio.create_task(
            run_production_watchdogs(
                pool,
                settings.db_url,
                held_threshold_seconds=settings.held_connection_watchdog_threshold_seconds,
                dead_man_threshold_seconds=settings.worker_dead_man_threshold_seconds,
                interval_seconds=settings.worker_watchdog_interval_seconds,
                rate_limit_seconds=settings.worker_watchdog_rate_limit_seconds,
                specimen_dir=settings.worker_watchdog_specimen_dir,
                journal_limit=settings.worker_watchdog_journal_events,
                operation_timeout_seconds=settings.worker_watchdog_operation_timeout_seconds,
                activity_limit=settings.worker_watchdog_activity_rows,
                max_specimens=settings.worker_watchdog_max_specimens,
            ),
            name="production_watchdogs",
        )

        # Start liveness heartbeat AFTER all critical resources are up,
        # so the healthcheck can't go green until the worker is fully
        # operational. Touch once now for an immediate green signal,
        # then the task takes over the periodic refresh.
        with contextlib.suppress(OSError):
            _HEARTBEAT_FILE.touch()
        heartbeat_task = asyncio.create_task(
            _periodic_heartbeat(interval=_HEARTBEAT_INTERVAL_SECONDS),
            name="heartbeat",
        )
        _supervise(heartbeat_task, latch=supervised_latch, fatal=supervised_failure)

        worker_task = asyncio.create_task(
            procrastinate_app.run_worker_async(
                # Sorted for a deterministic startup log; the set is the single
                # source of truth shared with the defer-time routing guard so a
                # queue can never be polled here yet unroutable there, or vice
                # versa (issue #1699).
                queues=sorted(WORKER_POLLED_QUEUES),
                concurrency=settings.worker_concurrency,
                wait=True,
                install_signal_handlers=True,
                # procrastinate defaults to NEVER deleting finished jobs, so
                # ``procrastinate_jobs`` would grow one row per wake forever.
                # Reap successful jobs; keep failures around for triage.
                delete_jobs="successful",
            ),
            name="procrastinate-worker",
        )
        latch_task = asyncio.create_task(supervised_latch.wait(), name="worker-supervision-latch")
        try:
            done, _pending = await asyncio.wait(
                {worker_task, latch_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if latch_task in done:
                worker_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await worker_task
            else:
                latch_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await latch_task
                await worker_task
        finally:
            latch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await latch_task
    finally:
        supervised_latch.set()
        log.info("worker.shutdown")
        # Drop the heartbeat file BEFORE other teardown so the healthcheck
        # flips to unhealthy as soon as shutdown begins — orchestrators
        # (Coolify, k8s) get the right liveness signal during the
        # potentially-slow drain that follows.
        with contextlib.suppress(OSError):
            _HEARTBEAT_FILE.unlink(missing_ok=True)
        if heartbeat_task is not None:
            await _cancel_and_drain(heartbeat_task)
        if sweep_task is not None:
            await _cancel_and_drain(sweep_task)
        if invariant_sweep_task is not None:
            await _cancel_and_drain(invariant_sweep_task)
        if interrupt_task is not None:
            await _cancel_and_drain(interrupt_task)
        if mcp_evict_task is not None:
            await _cancel_and_drain(mcp_evict_task)
        if github_clone_breaker_clear_task is not None:
            await _cancel_and_drain(github_clone_breaker_clear_task)
        if scheduler_task is not None:
            await _cancel_and_drain(scheduler_task)
        if watchdog_task is not None:
            await _cancel_and_drain(watchdog_task)
        if sandbox_registry is not None:
            sandbox_registry.stop_reaper()
            sandbox_registry.stop_gc()
            sandbox_registry.stop_egress_refresh()
        if inflight_tool_registry is not None:
            await inflight_tool_registry.shutdown()
        if sandbox_registry is not None:
            # Durable session sandboxes: STOP (don't destroy) every container
            # so their filesystems survive; the next worker's boot GC tick
            # salvages the stopped corpses.
            await sandbox_registry.stop_all()
        if mcp_session_pool is not None:
            mcp_session_pool.stop_reaper()
            await mcp_session_pool.close_all()
        if tool_broker is not None:
            await tool_broker.stop()
        if procrastinate_opened:
            await procrastinate_app.close_async()
        if pool is not None:
            await pool.close()
        # Lock conn drops last so single-instance enforcement holds for
        # the entire shutdown sequence (a parallel `aios worker` mid-startup
        # would still get refused while we tear down).
        with contextlib.suppress(asyncpg.PostgresError, OSError):
            await lock_conn.close()
        if supervised_failure["exception"] is not None:
            raise supervised_failure["exception"]


async def _acquire_worker_lock(
    db_url: str,
    log: Any,
    *,
    timeout_seconds: float = 30.0,
    poll_interval_seconds: float = 0.5,
) -> asyncpg.Connection[Any] | None:
    """Try to grab the single-worker advisory lock, waiting up to
    ``timeout_seconds`` for an existing holder to release.

    Returns the held connection on success, or ``None`` after timeout.
    The caller exits the process on ``None`` per the single-instance
    invariant. Postgres releases session-scoped advisory locks on
    connection close, so the caller's only obligation is to close the
    connection on shutdown.

    The wait handles rolling deploys: the previous worker is still
    shutting down when the new one starts, and a few seconds of
    polling is plenty of time for the old container to release. The
    timeout cap distinguishes "normal handoff" (resolves in seconds)
    from "stuck holder" (real bug, fail fast so it surfaces).

    The connection is dedicated — never returned to the pool — because
    pool reset would issue ``DISCARD ALL``, which releases advisory
    locks and silently drops the guarantee.
    """
    dsn = normalize_dsn(db_url)
    conn = await asyncpg.connect(dsn, server_settings=LISTENER_TCP_KEEPALIVE_SETTINGS)
    deadline = time.monotonic() + timeout_seconds
    waiting_logged = False
    try:
        while True:
            held: bool = await conn.fetchval(
                "SELECT pg_try_advisory_lock(hashtextextended($1, 0))",
                _WORKER_SINGLETON_LOCK_KEY_TEXT,
            )
            if held:
                return conn
            if time.monotonic() >= deadline:
                log.error(
                    "worker.duplicate_instance_refused",
                    lock_key=_WORKER_SINGLETON_LOCK_KEY_TEXT,
                    waited_seconds=timeout_seconds,
                )
                await conn.close()
                return None
            if not waiting_logged:
                log.info(
                    "worker.lock_busy.waiting",
                    lock_key=_WORKER_SINGLETON_LOCK_KEY_TEXT,
                    timeout_seconds=timeout_seconds,
                )
                waiting_logged = True
            await asyncio.sleep(poll_interval_seconds)
    except Exception:
        await conn.close()
        raise


async def _periodic_heartbeat(*, interval: int = _HEARTBEAT_INTERVAL_SECONDS) -> None:
    """Background task: touch the heartbeat file so the container's
    HEALTHCHECK can detect a hung or crashed worker.

    Started inside the worker's main try block, so it only runs after
    every other resource is up — the lock is held, the pool is open,
    procrastinate is consuming jobs. A worker that's stuck in startup
    (lock contention, pool DNS failure, etc.) does NOT touch the file
    and thus reports unhealthy after the threshold elapses, which is
    the behavior we want.
    """
    log = get_logger("aios.worker.heartbeat")
    while True:
        try:
            _HEARTBEAT_FILE.touch()
        except OSError as e:
            # tmpfs unavailable / permission denied — surface but don't
            # crash the worker; a missing heartbeat file simply means
            # the healthcheck reports unhealthy, which an operator
            # can investigate.
            log.warning("heartbeat.touch_failed", path=str(_HEARTBEAT_FILE), error=str(e))
        await asyncio.sleep(interval)


async def _periodic_invariant_sweep(pool: asyncpg.Pool[Any], *, interval: int) -> None:
    """Run C5 independently at its deliberately low cadence."""
    from aios.harness.invariant_sweep import sweep_session_invariants

    invariant_log = get_logger("aios.worker.invariant_sweep")
    while True:
        await asyncio.sleep(interval)
        try:
            result = await sweep_session_invariants(pool)
            invariant_log.info(
                "invariant_sweep.complete", selected=result.selected, archived=result.archived
            )
        except Exception:
            invariant_log.exception("invariant_sweep.failed")


async def _periodic_sweep(
    pool: asyncpg.Pool[Any],
    inflight_tool_registry: InflightToolRegistry,
    *,
    interval: int = 30,
) -> None:
    """Background task: run the sweep periodically."""
    log = get_logger("aios.worker.sweep")
    while True:
        await asyncio.sleep(interval)
        try:
            sweep = await wake_sessions_needing_inference(pool, inflight_tool_registry)
            if sweep.woken_sessions or sweep.repaired_ghosts:
                log.info(
                    "periodic_sweep.woken",
                    count=sweep.woken_sessions,
                    repaired_ghosts=sweep.repaired_ghosts,
                )
            from aios.workflows.sweep import wake_runs_needing_step

            woken_runs = await wake_runs_needing_step(pool)
            if woken_runs:
                log.info("periodic_sweep.workflows", woken_runs=woken_runs)

            await sweep_trigger_fires(pool)
        except Exception:
            log.exception("periodic_sweep.failed")


async def _host_dir_reaper_loop(pool: asyncpg.Pool[Any]) -> None:
    """Background loop: immediate first sweep, then every configured interval.

    Mirrors the snapshot GC reconciler (``SandboxRegistry._gc_loop``): the
    try/except is INSIDE the loop so one DB/FS hiccup never silently disables
    the reaper for the worker's lifetime, and the first sweep runs at boot to
    reclaim the predecessor's idle scratch immediately. ``sweep_host_dirs``
    itself honours the ``host_dir_reaper_enabled`` kill-switch.
    """
    log = get_logger("aios.worker.host_dir_reaper")
    interval = get_settings().host_dir_reaper_interval_seconds
    first = True
    while True:
        try:
            if not first:
                await asyncio.sleep(interval)
            first = False
            removed = await sweep_host_dirs(pool)
            if removed:
                log.info("host_dir_reaper.swept", removed=removed)
        except Exception:
            log.exception("host_dir_reaper.tick_failed")


async def _workspace_reaper_loop(pool: asyncpg.Pool[Any]) -> None:
    """Background loop for the archived-session workspace reaper (#40).

    Same shape as ``_host_dir_reaper_loop``: try/except INSIDE the loop so one
    DB/FS hiccup never silently disables it, immediate first sweep at boot, then
    every configured interval. ``sweep_archived_workspaces`` itself honours the
    ``workspace_reaper_enabled`` kill-switch (default OFF — ships dark).
    """
    log = get_logger("aios.worker.workspace_reaper")
    interval = get_settings().workspace_reaper_interval_seconds
    first = True
    while True:
        try:
            if not first:
                await asyncio.sleep(interval)
            first = False
            result = await sweep_archived_workspaces(pool)
            if result.reaped or result.bytes_freed:
                log.info(
                    "workspace_reaper.swept",
                    reaped=result.reaped,
                    bytes_freed=result.bytes_freed,
                    dry_run=result.dry_run,
                )
        except Exception:
            log.exception("workspace_reaper.tick_failed")


async def _memory_reconcile_audit_loop(pool: asyncpg.Pool[Any]) -> None:
    """Background loop for the uncorrelated memory-reconcile audit (#1748).

    Same shape as the other reaper loops: try/except INSIDE the loop so one
    DB/FS hiccup never silently disables the audit for the worker's
    lifetime, immediate first sweep at boot (so the detector is proven
    ringing early), then every ``memory_reconcile_audit_interval_seconds``.
    Honours ``memory_reconcile_audit_enabled`` (default ON — this is the
    uncorrelated backstop for the bash memory-reconcile stat-prefilter and
    must be live before that prefilter is trusted as the steady state).
    """
    log = get_logger("aios.worker.memory_reconcile_audit")
    settings = get_settings()
    from aios.harness.memory_reconcile_audit import run_memory_reconcile_audit

    interval = settings.memory_reconcile_audit_interval_seconds
    first = True
    while True:
        try:
            if not first:
                await asyncio.sleep(interval)
            first = False
            result = await run_memory_reconcile_audit(pool)
            log.info(
                "memory_reconcile_audit.tick",
                stores_checked=result.stores_checked,
                files_hashed=result.files_hashed,
                divergences=len(result.divergences),
            )
        except Exception:
            log.exception("memory_reconcile_audit.tick_failed")


async def _reclaimable_prune_loop(pool: asyncpg.Pool[Any]) -> None:
    """Background loop for the reclaimable-ephemera prune sweep (T6 #1461).

    Same shape as the reaper loops: try/except INSIDE the loop so one DB hiccup
    never silently disables the prune for the worker's lifetime, immediate first
    sweep at boot, then every ``reclaimable_prune_interval_seconds``.
    ``sweep_reclaimable_ephemera`` itself honours the ``reclaimable_prune_enabled``
    kill-switch and logs what it reclaimed.
    """
    log = get_logger("aios.worker.reclaimable_prune")
    interval = get_settings().reclaimable_prune_interval_seconds
    first = True
    while True:
        try:
            if not first:
                await asyncio.sleep(interval)
            first = False
            await sweep_reclaimable_ephemera(pool)
        except Exception:
            log.exception("reclaimable_prune.tick_failed")


async def _redrive_interrupts_for_tracked_sessions(
    pool: asyncpg.Pool[Any],
    inflight_tool_registry: InflightToolRegistry,
    *,
    log: Any,
) -> None:
    """Re-check every session this worker still has step/tool work in memory
    for against the durable interrupt marker, and cancel any that predates it.

    Deliverable invariant for #1756 hole 1 (lost NOTIFY): an interrupt that
    landed while the listener was disconnected/reconnecting writes its durable
    ``interrupt`` event (``routers/sessions.py``) regardless — that write
    always succeeds, only the fire-and-forget NOTIFY can be lost. This re-read
    of the durable state, run on every LISTEN (re)connect, re-drives
    cancellation for exactly the sessions this worker could still be running
    stale work for.

    Scope is ``inflight_tool_registry.tracked_session_ids()`` — the
    in-process step/tool tasks THIS worker owns — not a cross-worker DB scan:
    cheap (typically a handful of sessions), and the only scope that matters
    since only THIS worker's own asyncio tasks are cancellable from here. A
    session with no locally-tracked work has nothing to cancel regardless of
    interrupt state; the ghost-repair sweep is the backstop for a worker that
    crashed outright.

    Seq-bounded, not time-bounded (the constraint from #1756): each
    session's latest ``interrupt`` seq is compared against the ``start_seq``
    already threaded through the registry (:meth:`InflightToolRegistry.register_step`
    / the per-task ``dispatch_seq`` captured in :meth:`InflightToolRegistry.add`),
    so a step or tool dispatch that began AT OR AFTER the interrupt — a
    legitimate post-interrupt follow-up step (documented re-wake semantics,
    ``routers/sessions.py:553-556``) — is never cancelled. Cancelling an
    already-finished task is a no-op (idempotent), so running this on every
    reconnect — not just once — is safe and self-healing.
    """
    from aios.services.sessions import find_latest_interrupt_seq, load_session_account_id

    session_ids = inflight_tool_registry.tracked_session_ids()
    for session_id in session_ids:
        try:
            account_id = await load_session_account_id(pool, session_id)
            latest_interrupt_seq = await find_latest_interrupt_seq(
                pool, session_id, account_id=account_id
            )
            if latest_interrupt_seq is None:
                continue
            step_cancelled = inflight_tool_registry.cancel_step(
                session_id, min_start_seq=latest_interrupt_seq
            )
            tools_cancelled = inflight_tool_registry.cancel_session(
                session_id, min_start_seq=latest_interrupt_seq
            )
            if step_cancelled or tools_cancelled:
                log.info(
                    "interrupt_listener.redrive_dispatch",
                    session_id=session_id,
                    latest_interrupt_seq=latest_interrupt_seq,
                    step_cancelled=step_cancelled,
                    tools_cancelled=tools_cancelled,
                )
        except Exception:
            # Per-session isolation, matching the sweep's cross-session loops: one
            # gone session (archived/deleted mid-scan) or a transient DB error must
            # not abort the redrive for the rest of this worker's tracked sessions.
            log.exception("interrupt_listener.redrive_failed", session_id=session_id)


async def _run_interrupt_listener(
    db_url: str,
    inflight_tool_registry: InflightToolRegistry,
    pool: asyncpg.Pool[Any],
) -> None:
    """Drain pg_notify on the session-interrupt channel and cancel matching steps.

    Dispatch exceptions are isolated per payload. LISTEN connection failures
    or termination escape to the outer reconnect loop, which mirrors the
    scheduler: log, wait one backoff, and re-enter LISTEN indefinitely.
    ``CancelledError`` propagates so worker shutdown remains clean.

    #1756 (lost-NOTIFY hole): on every successful (re)connect — including the
    very first — :func:`_redrive_interrupts_for_tracked_sessions` re-reads the
    durable interrupt marker for every session this worker still has
    step/tool work tracked for, so an interrupt that landed during the
    reconnect backoff window (dropped NOTIFY, no live listener to receive it)
    is enforced at most one backoff late instead of never. ``pool`` is passed
    explicitly (not read off :mod:`aios.harness.runtime`) so this function
    stays independently testable without a full worker boot.
    """
    log = get_logger("aios.worker.interrupt_listener")
    while True:
        try:
            async with listen_for_session_interrupts(db_url) as queue:
                await _redrive_interrupts_for_tracked_sessions(
                    pool, inflight_tool_registry, log=log
                )
                while True:
                    try:
                        session_id = await queue.get()
                        if session_id == "":
                            raise ConnectionError("session interrupt LISTEN connection terminated")
                        step_cancelled = inflight_tool_registry.cancel_step(session_id)
                        tools_cancelled = inflight_tool_registry.cancel_session(session_id)
                        log.info(
                            "interrupt_listener.dispatch",
                            session_id=session_id,
                            step_cancelled=step_cancelled,
                            tools_cancelled=tools_cancelled,
                        )
                    except ConnectionError:
                        raise
                    except Exception:
                        log.exception("interrupt_listener.dispatch_failed")
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("interrupt_listener.listen_failed_will_retry")
            await asyncio.sleep(_LISTEN_RECONNECT_BACKOFF_SECONDS)


async def _run_mcp_evict_listener(db_url: str) -> None:
    """Drain pg_notify on the MCP-pool eviction channel and evict pooled
    sessions for the rotated vault.

    An operator credential mutation (PUT/archive/delete on a vault
    credential, plus vault archive/delete) runs in the API process and
    NOTIFYs ``aios_mcp_evict_vault`` with the ``vault_id``. The worker owns
    the MCP session pool, so it LISTENs here and calls
    :meth:`McpSessionPool.evict_by_vault` — discarding idle sessions and
    flagging in-use ones for discard-on-release so the rotated secret
    propagates immediately, instead of waiting out the 900s idle TTL that
    active polling defeats (#1030).

    Mirrors :func:`_run_interrupt_listener`'s survivability contract: the
    dispatch try/except is nested INSIDE ``while True`` so a transient
    eviction failure doesn't disable the listener for the worker's lifetime;
    LISTEN-connection failures escape to the outer reconnect loop.
    ``CancelledError`` propagates so worker shutdown stays clean.
    """
    log = get_logger("aios.worker.mcp_evict_listener")
    while True:
        try:
            async with listen_for_mcp_evict_vault(db_url) as queue:
                while True:
                    try:
                        vault_id = await queue.get()
                        if vault_id == "":
                            raise ConnectionError("mcp evict LISTEN connection terminated")
                        pool = runtime.mcp_session_pool
                        if pool is None:
                            # Pool not yet initialized (startup race) or already
                            # torn down — nothing to evict; the idle reaper and
                            # a fresh pool cover the gap.
                            continue
                        await pool.evict_by_vault(vault_id)
                        log.info("mcp_evict_listener.dispatch", vault_id=vault_id)
                    except ConnectionError:
                        raise
                    except Exception:
                        log.exception("mcp_evict_listener.dispatch_failed")
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("mcp_evict_listener.listen_failed_will_retry")
            await asyncio.sleep(_LISTEN_RECONNECT_BACKOFF_SECONDS)


async def _run_github_clone_breaker_clear_listener(db_url: str) -> None:
    """Drain pg_notify on the github-clone-breaker clear channel and clear
    the rotated resource's breaker state (#1720).

    :func:`aios.services.github_repositories.rotate_token` runs in the API
    process and NOTIFYs ``aios_github_clone_breaker_clear`` with the
    ``ghrepo_...`` resource id after a token/identity UPDATE commits. The
    worker owns ``runtime.github_clone_breaker``, so it LISTENs here and
    calls :meth:`GithubCloneBreaker.clear` — a fixed credential re-probes on
    the very next provision instead of serving out a cooldown opened under
    the old secret.

    Mirrors :func:`_run_mcp_evict_listener`'s survivability contract: the
    dispatch try/except is nested INSIDE ``while True`` so a transient clear
    failure doesn't disable the listener for the worker's lifetime;
    LISTEN-connection failures escape to the outer reconnect loop.
    ``CancelledError`` propagates so worker shutdown stays clean.
    """
    log = get_logger("aios.worker.github_clone_breaker_clear_listener")
    while True:
        try:
            async with listen_for_github_clone_breaker_clear(db_url) as queue:
                while True:
                    try:
                        resource_id = await queue.get()
                        if resource_id == "":
                            raise ConnectionError(
                                "github clone breaker clear LISTEN connection terminated"
                            )
                        breaker = runtime.github_clone_breaker
                        if breaker is None:
                            # Not yet initialized (startup race) or already
                            # torn down — nothing to clear; the breaker's own
                            # cooldown/half-open re-probe covers the gap.
                            continue
                        breaker.clear(resource_id)
                        log.info(
                            "github_clone_breaker_clear_listener.dispatch",
                            resource_id=resource_id,
                        )
                    except ConnectionError:
                        raise
                    except Exception:
                        log.exception("github_clone_breaker_clear_listener.dispatch_failed")
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("github_clone_breaker_clear_listener.listen_failed_will_retry")
            await asyncio.sleep(_LISTEN_RECONNECT_BACKOFF_SECONDS)
