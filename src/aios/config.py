"""Server configuration loaded from environment variables.

Single source of truth for every env-driven knob in aios. Imported once at
process start; downstream modules read attributes off the ``settings`` instance
rather than touching ``os.environ`` directly.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from aios.models.vaults import OAuthProviderApp

# Wall-clock cap on a single ``run_session_step`` call (the harness step
# budget). Imported by ``aios.harness.loop`` as the job-level asyncio.wait_for
# timeout; the validators below reject per-phase budgets that meet or exceed it.
HARNESS_STEP_TIMEOUT_S: float = 960.0

OutboundToolQuota = tuple[
    Annotated[int, Field(gt=0)],
    Annotated[int, Field(gt=0)],
]


class Settings(BaseSettings):
    """All env-driven aios configuration.

    Reads from process env first, then a ``.env`` file if present, then defaults.
    Required values without defaults will raise on instantiation if missing.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIOS_",
        # Layered: user-level shared secrets (seeded once) then per-worktree
        # overrides. Later file wins; process env still beats both. `~` is not
        # expanded by pydantic-settings, so we resolve Path.home() explicitly.
        # Missing files are silently skipped.
        env_file=(
            str(Path.home() / ".aios" / "secrets.env"),
            ".env",
        ),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── instance identity ──────────────────────────────────────────────────
    instance_id: str = Field(
        default="default",
        pattern=r"^[a-z_][a-z0-9_]*$",
        description="Distinguishes concurrent aios deployments on a shared host. "
        "Flows into Docker container labels so each worker's orphan-reaper only "
        "touches its own containers, and into dev-bootstrap DB naming. The "
        "`default` value preserves pre-existing single-instance behavior; `aios "
        "dev bootstrap` writes a unique value per git worktree. The pattern "
        "restricts the value to chars safe for Postgres identifiers and Docker "
        "labels.",
    )

    # ── crypto (required) ──────────────────────────────────────────────────
    vault_key: SecretStr = Field(
        ...,
        description="Base64-encoded 32-byte master key for libsodium secretbox.",
    )
    vault_key_previous: SecretStr | None = Field(
        default=None,
        description="Optional previous AIOS_VAULT_KEY used only by `aios rekey` to decrypt during rotation.",
    )
    egress_ca_key: SecretStr = Field(
        ...,
        description="Base64-encoded 32-byte key for the sandbox egress CA.",
    )
    bootstrap_token: SecretStr | None = Field(
        default=None,
        description="Unlocks ``POST /v1/accounts/bootstrap`` while the ``accounts`` table "
        "has no root row; the endpoint is 404 once a root exists. Generate with "
        "``openssl rand -base64 32``.",
    )

    inference_credential_policy: Literal["account_only", "observe_legacy_env", "legacy_env"] = (
        Field(
            default="account_only",
            description="Admission policy for paid inference credentials. account_only fails closed; "
            "legacy modes explicitly permit LiteLLM process-environment fallback during migration.",
        )
    )
    tenancy_posture: Literal["single_operator", "external_byok"] = Field(
        default="single_operator",
        description="Deployment tenancy posture. external_byok forbids every process-env inference fallback.",
    )

    default_spend_limit_usd: float | None = Field(
        default=None,
        ge=0,
        description=(
            "Default lifetime USD spend limit for accounts that do not set or "
            "inherit account config spend_limit_usd. Unset means ungated."
        ),
    )

    # ── interactive OAuth (vault credential "Connect") ─────────────────────
    oauth_provider_apps: list[OAuthProviderApp] = Field(
        default_factory=list,
        description="Operator-registered OAuth client apps for MCP providers that don't "
        "support Dynamic Client Registration (Google, Microsoft, Slack, …). When a "
        "server's discovered OAuth issuer/host matches an app's ``match``, the interactive "
        "Connect flow uses that app so end users sign in without supplying any credentials. "
        "Set ``AIOS_OAUTH_PROVIDER_APPS`` to a JSON list of "
        "{match, client_id, client_secret?, token_endpoint_auth_method?, "
        "token_endpoint_hosts?, scope?, authorize_params?}. Register "
        "each app with the provider using the console callback "
        "``<console_url>/api/auth/mcp-oauth/callback`` as the redirect URI.",
    )
    oauth_allow_insecure_hosts: str = Field(
        default="",
        description="DEV-ONLY comma-separated host[:port] allowlist that bypasses the "
        "interactive-Connect SSRF guard (https requirement + private-range block) for "
        "the listed hosts. Lets a self-hosted MCP fleet reachable only over plain http "
        "on an internal host (e.g. ``workspace-mcp:8000``) be connected from dev. Leave "
        "empty in production (prod MCP servers are https), where any value would re-open "
        "SSRF to internal hosts. Plain comma string (not JSON) so the natural ``.env`` "
        "format ``host-a,host-b:8000`` parses without brackets.",
    )

    # ── database (required) ────────────────────────────────────────────────
    db_url: str = Field(
        ...,
        description="Postgres connection URL (postgresql://user:pass@host:port/db).",
    )

    # ── sandbox ────────────────────────────────────────────────────────────
    docker_image: str = Field(
        default="ghcr.io/eumemic/aios-sandbox:latest",
        description="Container image used for all v1 sessions. Published from "
        "`docker/Dockerfile.sandbox` to GHCR by the build-sandbox workflow. "
        "Override to a local tag for development "
        "(`docker build -t aios-sandbox:latest -f docker/Dockerfile.sandbox .`).",
    )
    workspace_root: Path = Field(
        default=Path("/var/lib/aios/workspaces"),
        description="Host directory containing per-session workspace subdirectories. "
        "Each session gets <workspace_root>/<session_id> bind-mounted to /workspace "
        "inside its sandbox container.",
    )
    workspaces_owner_uid: int = Field(
        default=1000,
        ge=0,
        description="uid that should own every directory under workspace_root. The "
        "worker (root) chowns newly-created shared-tree components to this uid so the "
        "api container (running as this uid) can write into them. Set via "
        "AIOS_WORKSPACES_OWNER_UID.",
    )
    workspaces_owner_gid: int = Field(
        default=1000,
        ge=0,
        description="gid counterpart to workspaces_owner_uid. Set via AIOS_WORKSPACES_OWNER_GID.",
    )
    sandbox_cpu_quota: float | None = Field(
        default=None,
        ge=0.01,
        description="Maximum cumulative CPU cores a sandbox can use, in "
        "decimal core units (e.g. 2.0 = two cores at 100%, 0.5 = half "
        "of one core). Translates to ``docker run --cpus``. The lower "
        "bound is Docker's effective floor — below ~0.01 the runtime "
        "rounds the CFS quota down and the value becomes meaningless. "
        "``None`` leaves the host's default in place (unlimited).",
    )
    sandbox_memory_bytes: int | None = Field(
        default=None,
        ge=4 * 1024 * 1024,
        description="Maximum resident memory a sandbox can use, in bytes. "
        "Translates to ``docker run --memory`` and ``--memory-swap`` "
        "(swap pinned to the same value so the sandbox can't lean on "
        "swap to exceed the cap). The kernel OOM-kills the offending "
        "process when this is breached. ``None`` leaves the host's "
        "default (unlimited) in place. Minimum 4 MiB to satisfy Docker's "
        "own floor.",
    )
    sandbox_egress_max_request_bytes: int = Field(
        default=100 * 1024 * 1024,  # 100 MiB
        ge=0,
        description="Maximum request-body size the secret egress proxy will "
        "buffer before forwarding upstream. The body is buffered whole so the "
        "credential placeholder can be swapped across chunk boundaries, so this "
        "is the per-request worker-memory ceiling for a sandbox-originated "
        "upload. A request whose body exceeds this is rejected with 413 before "
        "any upstream connection — the sandbox is untrusted, so this bounds the "
        "OOM blast radius on the shared worker. Set via "
        "AIOS_SANDBOX_EGRESS_MAX_REQUEST_BYTES.",
    )
    sandbox_pids_limit: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of PIDs the sandbox cgroup can hold. "
        "Translates to ``docker run --pids-limit``. Mitigates fork-bomb "
        "denial-of-service; ``None`` leaves the host's default in place.",
    )
    sandbox_snapshot_budget_bytes: int | None = Field(
        default=4 * 1024 * 1024 * 1024,
        ge=10 * 1024 * 1024,
        description="Per-session snapshot budget in **unique** bytes (durable "
        "session sandboxes, §5.7). When a session's unique snapshot bytes "
        "would exceed this at idle/teardown, the snapshot verb FLATTENS "
        "(collapse + whiteout, the definitive secret scrub) instead of "
        "committing another layer; commit-and-flag, never refuse (refusal "
        "would destroy the agent's work as punishment for a state it wasn't "
        "awake to prevent). Replaces the dead ``sandbox_disk_bytes`` "
        "writable-layer cap, which required overlay2-on-xfs+pquota and never "
        "worked on prod ext4. ``None`` ⇒ unbounded (the depth backstop still "
        "applies). Per-environment overridable via "
        "``EnvironmentConfig.snapshot_budget_bytes``.",
    )
    sandbox_pipeline_stall_seconds: float = Field(
        default=120.0,
        gt=0,
        description="Maximum interval without bytes moving through a Docker flatten pipeline.",
    )
    sandbox_docker_cli_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        description="Timeout for individual Docker CLI management calls.",
    )
    sandbox_inspect_size_timeout_seconds: float = Field(
        default=300.0,
        gt=0,
        description="Timeout for the ``docker inspect --size`` writable-layer "
        "walk during salvage; it scales with the corpse, so it gets this "
        "generous bound rather than the blanket ``DOCKER_CLI_TIMEOUT_S``.",
    )
    sandbox_snapshot_timeout_floor_seconds: float = Field(
        default=60.0, gt=0, description="Minimum Docker snapshot operation timeout."
    )
    sandbox_snapshot_timeout_ns_per_byte: float = Field(
        default=20e-9,
        gt=0,
        description="Fallback snapshot time per byte until measured throughput is available.",
    )
    sandbox_snapshot_timeout_safety_margin: float = Field(
        default=2.0,
        ge=1.0,
        description="Multiplier applied to throughput-derived snapshot budgets.",
    )
    sandbox_snapshot_timeout_retry_multiplier: float = Field(
        default=2.0, ge=1.0, description="Budget multiplier for each prior timeout of a corpse."
    )
    sandbox_snapshot_timeout_retry_cap: float = Field(
        default=16.0, ge=1.0, description="Maximum cumulative timeout retry multiplier."
    )
    sandbox_snapshot_throughput_ewma_alpha: float = Field(
        default=0.25, gt=0, le=1, description="Weight of the latest successful snapshot throughput."
    )
    sandbox_snapshot_throughput_state_path: Path = Field(
        default=Path("/var/lib/aios/snapshot-throughput.json"),
        description="Host-local persisted snapshot throughput calibration.",
    )
    sandbox_salvage_breaker_threshold: int = Field(
        default=3,
        ge=1,
        description="Consecutive failures of one corpse before salvage retries are suppressed.",
    )
    sandbox_flatten_disk_floor_bytes: int = Field(
        default=15 * 1024 * 1024 * 1024,
        ge=0,
        description="Free disk retained in addition to the estimated transient flatten cost.",
    )
    sandbox_snapshot_pool_bytes: int | None = Field(
        default=None,
        ge=0,
        description="Per-host snapshot pressure threshold in bytes (durable session "
        "sandboxes, §5.7). Crossing it emits an operator diagnostic but never "
        "authorizes deletion of canonical session state. Required in "
        "production once the eumemic-ops prune-cron exemption lands (that "
        "exemption removes the only existing disk control). ``None`` ⇒ "
        "unbounded retention (dev default); operators MUST set real host "
        "headroom in production. v1 enforces per-host and reports global.",
    )
    sandbox_snapshot_empty_floor_bytes: int = Field(
        default=8192,
        ge=0,
        description="Writable-layer byte floor at/below which a snapshot is "
        "treated as a no-write identity (``skipped_empty`` — no new image, no "
        "chain growth). NOT zero: on the production **containerd image store** "
        "a no-write container reports ``SizeRw == 4096``, not 0, so an "
        "``== 0`` test would never fire in prod and chat-only / read-only "
        "sessions would grow a chain every idle. The floor (default 8 KiB) is "
        "what keeps them from ever snapshotting.",
    )
    sandbox_seccomp_profile: str = Field(
        default=str(Path(__file__).resolve().parents[2] / "docker" / "seccomp-sandbox.json"),
        description="Path to the authored seccomp profile applied to every sandbox "
        "container via `docker run --security-opt seccomp=<path>`. The docker CLI "
        "reads this file from the WORKER container's filesystem and ships the JSON "
        "to the daemon. Defaults to the baked-in profile shipped in the worker image. "
        "Emergency rollback ONLY: set AIOS_SANDBOX_SECCOMP_PROFILE=unconfined to "
        "disable seccomp filtering. Never defaults to unconfined; the flag is always emitted.",
    )
    sandbox_backend: str = Field(
        default="docker",
        description="Sandbox backend implementation. 'docker' (default) runs "
        "DockerBackend. Override only where an alternate SandboxBackend impl is installed.",
    )
    sandbox_runtime: str | None = Field(
        default=None,
        description="Optional Docker container runtime for sandboxes and their "
        "network-lockdown sidecars. Unset (or Docker's configured default, normally "
        "runc) preserves local/CI behavior; set AIOS_SANDBOX_RUNTIME=runsc to run "
        "containers under gVisor where the host has runsc installed.",
    )
    bash_default_timeout_seconds: int = Field(
        default=120,
        ge=1,
        description="Default ceiling for a single bash tool call, in seconds. "
        "The agent can override per-call up to this maximum. A session bound "
        "to an environment with ``EnvironmentConfig.bash_timeout_seconds`` set "
        "uses that environment's ceiling instead — letting heavy dev workloads "
        "run >120s commands without raising the global default for every "
        "session on the worker (issue #725).",
    )
    bash_max_output_bytes: int = Field(
        default=100_000,
        ge=1_000,
        description="Maximum bytes of stdout+stderr returned from a bash tool call. "
        "Output beyond this is truncated with a [truncated] marker.",
    )
    tool_result_max_chars: int = Field(
        default=200_000,
        ge=1_000,
        description="Maximum characters of a tool result stored inline in the "
        "event log. A larger result is spilled into a session-scoped context "
        "variable and replaced inline with a stub carrying the variable handle "
        "plus a preview (docs/rlm.md), so a single oversized result can't "
        "exceed the context window.",
    )
    ctx_peek_max_bytes: int = Field(
        default=16_384,
        ge=256,
        description="Hard per-call cap on the content slice a ``ctx_peek`` tool "
        "call may return. Larger reads take multiple calls or a ``ctx_grep``/"
        "``ctx_eval`` — the cap is what keeps out-of-context variables from "
        "re-entering the prompt wholesale (docs/rlm.md).",
    )
    upload_max_size_bytes: int = Field(
        default=50 * 1024 * 1024,
        ge=1,
        description="Maximum size of a single file uploaded via "
        "``POST /v1/sessions/<id>/files``. Overflow is rejected with 413 "
        "after the multipart body is drained so the client sees a clean "
        "response rather than a transport reset.",
    )
    model_call_deadline_s: float = Field(
        default=900.0,
        ge=1.0,
        description="Total-duration deadline for a single model call, in seconds. "
        "Must stay strictly below the harness step timeout so job-level "
        "cancellation remains a final safety net instead of the normal model "
        "budget.",
    )
    github_clone_session_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        description="Wall-clock bound on each per-session "
        "``git clone --reference --dissociate`` (and the short admin ops "
        "around it: ``remote set-url``, ``config user.*``). Must stay "
        "well below the 960s harness step timeout — otherwise a hung "
        "clone burns a whole user turn before the step-level cap fires. "
        "See issue #697.",
    )
    github_clone_cache_timeout_seconds: float = Field(
        default=300.0,
        ge=1.0,
        description="Wall-clock bound on the bare-cache clone/fetch "
        "(``<workspace_root>/_github_repos/<sha256(url)>``). Cold-case "
        "clones of large repos can legitimately take minutes; failures "
        "here only delay the cache, they don't block the per-session "
        "working tree past the per-session budget. See issue #697.",
    )

    # ── worker ─────────────────────────────────────────────────────────────
    worker_concurrency: int = Field(
        default=4,
        ge=1,
        description="Concurrent session steps per worker process.",
    )
    held_connection_watchdog_threshold_seconds: float = Field(
        default=60.0,
        gt=0,
        description="Seconds a worker asyncpg checkout may remain held before specimen capture.",
    )
    worker_dead_man_threshold_seconds: float = Field(
        default=600.0,
        gt=0,
        description="Seconds claimed jobs may produce zero completed steps before an alarm.",
    )
    worker_watchdog_interval_seconds: float = Field(default=10.0, gt=0)
    worker_watchdog_rate_limit_seconds: float = Field(default=300.0, gt=0)
    worker_watchdog_journal_events: int = Field(default=100, ge=1, le=1000)
    worker_watchdog_operation_timeout_seconds: float = Field(default=5.0, gt=0, le=60)
    worker_watchdog_activity_rows: int = Field(default=100, ge=1, le=1000)
    worker_watchdog_max_specimens: int = Field(default=20, ge=1, le=1000)
    worker_watchdog_specimen_dir: Path = Field(default=Path("/tmp/aios-freeze-specimens"))

    # ── container lifecycle ────────────────────────────────────────────────
    container_idle_timeout_seconds: int = Field(
        default=1800,
        ge=30,
        description="Seconds of inactivity before a session's sandbox container "
        "is released. The idle reaper checks every 60s. Raised from 300 to 1800 "
        "(30 min) for durable session sandboxes (§5.10): teardown now costs a "
        "commit + a layer + eventual flatten, so keeping an idle ``tail -f`` "
        "container alive (~1 MB RSS) is the cheap option. At 300s a daily-driver "
        "session would commit 20-80x/day and a 15-min cron-trigger session "
        "~96x/day, hitting the flatten wall in days; at 1800s a 15-min cron "
        "session never idles out (zero commits until it stops) and a "
        "conversational session commits a handful of times a day.",
    )
    mcp_pool_idle_timeout_seconds: int = Field(
        default=900,
        ge=30,
        description="Seconds of inactivity before a pooled MCP session is "
        "closed. An OAuth token refresh rotates the bearer, orphaning the "
        "old keyed entry; the idle reaper (checks every 60s) reclaims it.",
    )

    # ── host scratch-dir reaper (#1192) ────────────────────────────────────
    host_dir_reaper_enabled: bool = Field(
        default=True,
        description="Kill-switch for the idle host scratch-dir reaper "
        "(``_session_repos`` working-tree clones and ``_runs`` per-run "
        "scratch). When False the reaper is never started and deletes "
        "nothing — so the worker that just had a P1 disk-fill can disable "
        "irreversible host deletion without a redeploy (set "
        "``AIOS_HOST_DIR_REAPER_ENABLED=false``). Default-on: the missing GC "
        "is itself the floodgates incident (#1192).",
    )
    host_dir_reaper_min_age_seconds: int = Field(
        default=3600,
        ge=0,
        description="Minimum age (by directory mtime) before an idle "
        "``_session_repos``/``_runs`` host dir is eligible for reaping. A "
        "race floor: a dir freshly created by an in-flight provision whose "
        "DB row has not yet committed (or a run whose terminal status is "
        "about to flip back) is left alone until it ages past this floor. "
        "The dominant safety check is DB liveness; this is the belt-and-"
        "suspenders for the provision/commit gap.",
    )
    host_dir_reaper_interval_seconds: float = Field(
        default=3600.0,
        gt=0.0,
        description="Interval for the periodic host scratch-dir reaper sweep "
        "(an immediate first sweep runs at worker startup, then repeats at "
        "this cadence — mirrors the snapshot GC reconciler).",
    )

    # ── archived-session workspace reaper (#40, the 45G hole) ──────────────────
    workspace_reaper_enabled: bool = Field(
        default=False,
        description="Kill-switch for the archived-session workspace reaper "
        "(reclaims the per-session ``/workspace`` host dir of "
        "``archived_at IS NOT NULL`` sessions — the 45G hole #1192 did not "
        "cover). Ships DARK (default OFF) and is enabled after review (set "
        "``AIOS_WORKSPACE_REAPER_ENABLED=true``): unlike the #1192 reaper, this "
        "deletes a session's actual working files, not reconstructible clones, "
        "so it does not ship enabled. When False the reaper is never started "
        "and deletes nothing.",
    )
    workspace_reaper_dry_run: bool = Field(
        default=False,
        description="When True the archived-session workspace reaper logs what "
        "it WOULD reap (paths + bytes reclaimable) and the structured per-sweep "
        "count, but deletes nothing. Lets a deploy verify the keep-set/predicate "
        "on real data before the destructive sweep is armed "
        "(``AIOS_WORKSPACE_REAPER_DRY_RUN=true``).",
    )
    workspace_reaper_min_archived_age_seconds: int = Field(
        default=86_400,
        ge=0,
        description="Minimum age, by ``sessions.archived_at`` (DB archive time, "
        "NOT file mtime), before an archived session's ``/workspace`` dir is "
        "eligible for reaping. Conservative default 24h: guards a just-archived "
        "session whose workspace something might still be reading in the window "
        "right after archive.",
    )
    workspace_reaper_min_mtime_age_seconds: int = Field(
        default=3600,
        ge=0,
        description="Belt-and-suspenders floor on the workspace dir's own mtime "
        "for the archived-session workspace reaper: a dir touched more recently "
        "than this (e.g. a step that was mid-write at archive time) is left "
        "alone regardless of archive age. The dominant gate is the DB "
        "archived-and-aged check; this is the provision/write race floor.",
    )
    workspace_reaper_interval_seconds: float = Field(
        default=3600.0,
        gt=0.0,
        description="Interval for the periodic archived-session workspace "
        "reaper sweep (an immediate first sweep runs at worker startup, then "
        "repeats at this cadence).",
    )
    # ── uncorrelated memory-reconcile audit (#1748 §Uncorrelated detector) ────
    memory_reconcile_audit_enabled: bool = Field(
        default=True,
        description="Kill-switch for the low-rate out-of-band full-hash "
        "memory-store audit (disk-vs-DB content_sha256 divergence check). "
        "Ships armed: this is the uncorrelated backstop for the bash "
        "memory-reconcile stat-prefilter (#1748) and MUST be live before the "
        "prefilter is trusted as the steady state — disabling it removes the "
        "only detector for a prefilter false-negative silently dropping "
        "durable memory.",
    )
    memory_reconcile_audit_interval_seconds: float = Field(
        default=21_600.0,
        gt=0.0,
        description="Interval for the periodic uncorrelated memory-reconcile "
        "audit sweep (default 6h — deliberately low-rate: a full content hash "
        "of every memory-store file is exactly the on-loop-hash cost #1733 "
        "forbids per bash call; this runs occasionally, off the hot path, as "
        "a correctness backstop, not a real-time gate).",
    )

    tool_broker_socket_path: Path | None = Field(
        default=None,
        description="Host path for the tool broker's Unix-domain socket. "
        "When set, the broker dual-listens on TCP (ephemeral port, used by "
        "any out-of-container tooling) AND on this UDS path; sandbox "
        "containers receive ``TOOL_BROKER_URL=unix:///var/run/aios/tool-broker.sock`` "
        "and a bind mount of the socket file. Use this in deployments where "
        "the worker's TCP port isn't reachable from the sandbox network — "
        "e.g. Coolify production where TCP port publishing isn't done. When "
        "unset (the default), sandboxes reach the broker over TCP via "
        "``http://aios-worker:<port>``.",
    )

    # ── database pool ──────────────────────────────────────────────────────
    db_pool_max_size: int = Field(
        default=8,
        ge=1,
        description="Maximum asyncpg pool size. Should comfortably exceed "
        "worker_concurrency * ~3 connections per turn.",
    )

    # ── api server ─────────────────────────────────────────────────────────
    api_host: str = Field(default="127.0.0.1")
    api_port: int = Field(default=8080, ge=1, le=65535)

    # ── web tools ──────────────────────────────────────────────────────────
    tavily_api_key: str | None = Field(
        default=None,
        description="Tavily API key for web_fetch and web_search tools.",
    )

    default_mcp_permission_policy: Literal["always_allow", "always_ask"] | None = Field(
        default=None,
        description="Fallback permission policy for MCP tools whose server "
        "has no matching ``mcp_toolset`` entry on the calling agent. "
        "When unset (the default), unmounted MCP toolsets gate on "
        "``always_ask`` confirmation — the safe default that protects "
        "against connector-mounted tools the agent never opted into. "
        "Operators in trusted environments can set this to "
        "``always_allow`` so connector tools dispatch immediately on any "
        "session that has the connection attached, without per-agent "
        "``mcp_toolset`` declarations.  Explicit per-toolset policies on "
        "``agent.tools`` always win — this only changes the fallback "
        "for unmounted servers.",
    )

    trusted_inference_api_bases: list[str] = Field(
        default_factory=list,
        description="Operator allowlist of trusted inference endpoints for the "
        "model-identity clamp at the workflow ``agent()`` spawn edge (#823). A "
        "named-agent child's effective ``api_base`` (from its ``litellm_extra``) "
        "must equal the launcher's (a workflow run has none — i.e. the default "
        "operator endpoint, so ``api_base`` must be unset) OR appear in this "
        "allowlist; otherwise the spawn fails closed with an "
        "``untrusted_api_base`` rejection. Empty (the default) means *no* "
        "redirected endpoint is trusted: a child may only run on the default "
        "endpoint. The clamp makes this boundary real and frozen independent of "
        "native self-management: the api_base axis is enforced at the spawn edge "
        "regardless of who authored the agent (the ``workflow:`` model-binding "
        "privilege, #1636, is the companion runtime guard on the orthogonal "
        "model/binding axis). Values are matched verbatim against "
        "the agent's ``litellm_extra['api_base']`` string.",
    )

    # ── triggers ───────────────────────────────────────────────────────────
    triggers_per_account_max: int = Field(
        default=100,
        ge=1,
        description="Per-account ceiling on enabled trigger rows "
        "across all sessions. Enforced at ``add_trigger`` time; on exceed, "
        "the call raises ``RateLimitedError``. Bounds the worst-case work "
        "the event-driven scheduler has to enumerate on each tick, and "
        "the fire-storm a misbehaving agent could deflect at the broker. "
        "Includes one-shot ``schedule_wake`` rows; bump if your workload "
        "legitimately needs more standing timers per tenant.",
    )
    trigger_runs_retention_days: int = Field(
        default=30,
        ge=1,
        description="Days of per-fire trigger audit rows (``trigger_runs``) to "
        "retain; older rows are pruned by the worker's periodic sweep. "
        "Time-based by design — see ``prune_trigger_runs`` for why a "
        "count-cap would be unsafe.",
    )
    # ── reclaimable-ephemera prune (T6, #1461) ────────────────────────────────
    reclaimable_prune_enabled: bool = Field(
        default=True,
        description="Kill-switch for the age-based prune of RECLAIMABLE instance "
        "ephemera (T6): terminal+archived ``wf_runs`` (and their ``WfRunEvent`` "
        "journals, which cascade) plus archived agent/skill/workflow definitions "
        "no live session/run still pins. When False the maintenance sweep prunes "
        "nothing (set ``AIOS_RECLAIMABLE_PRUNE_ENABLED=false``). NEVER touches the "
        "sacred set: memory content, referenced session history, any version a "
        "live session pins, or accounts. Time-based per the ``trigger_runs`` "
        "doctrine — no count-cap (which could evict a young claim row).",
    )
    wf_runs_retention_days: int = Field(
        default=30,
        ge=1,
        description="Days a TERMINAL+ARCHIVED workflow run (``wf_runs`` with "
        "``archived_at`` set by ``archive_run``) is retained before the prune "
        "sweep deletes it — dropping its ``wf_run_events`` journal (the unbounded-"
        "growth driver) via ``ON DELETE CASCADE`` in the same statement. Modeled "
        "on ``trigger_runs_retention_days``; age-keyed on ``archived_at``. "
        "Time-based by design — a count-cap is explicitly rejected (#1461).",
    )
    archived_definition_retention_days: int = Field(
        default=30,
        ge=1,
        description="Days an ARCHIVED agent/skill/workflow definition with NO live "
        "session/run pinning it is retained before the prune sweep deletes it "
        "(with its version history). Replay-stability holds any definition a live "
        "session still pins — those versions are SACRED and survive regardless of "
        "age. Modeled on ``trigger_runs_retention_days``; time-based.",
    )
    reclaimable_prune_interval_seconds: float = Field(
        default=3600.0,
        gt=0.0,
        description="Interval for the periodic reclaimable-ephemera prune sweep "
        "(an immediate first sweep runs at worker startup, then repeats at this "
        "cadence — mirrors the host scratch-dir reaper). The sweep is idempotent "
        "and safe to run repeatedly; it logs what it reclaimed.",
    )

    schedule_wake_max_delay_seconds: int = Field(
        default=60 * 60 * 24 * 30,  # 30 days
        ge=60,
        description="Upper bound on the agent-requested wake delay (via "
        "``schedule_wake``'s ``delay_seconds`` or the resolved offset of "
        "its ``at`` argument). Bounds (a) unbounded scheduler MIN-query "
        "enumeration over agent-set-and-forget rows that may never fire "
        "within a worker's lifetime, and (b) catches obvious mistakes "
        "(``delay_seconds=86400000`` instead of ``86400``) at the tool "
        "boundary with a clear typed error instead of letting them sit "
        "in the DB for years. Set high for legitimate long-horizon "
        "reminders.",
    )

    # ── workflows ──────────────────────────────────────────────────────────
    workflow_max_agent_calls: int = Field(
        default=1000,
        ge=1,
        description="Per-run lifetime ceiling on the number of ``agent()`` "
        "children a workflow may spawn. Counted from the run's ``call_started`` "
        "journal (monotonic), checked before each step's spawns: a step that "
        "would push the total past the cap errors the run (``too_many_agents``) "
        "before spawning any of that step's new children. Bounds a runaway "
        "script that spawns children in an unbounded loop. The per-call "
        "single-``parallel()`` fan-out width is bounded separately, in the "
        "host (``wf_script_host.MAX_PARALLEL_FANOUT``).",
    )
    workflow_default_child_model: str | None = Field(
        default=None,
        description="Default model for generic workflow agent() children launched via operator/HTTP runs when no per-run or per-call model is supplied.",
    )
    model_tiers: dict[str, str] = Field(
        default_factory=dict,
        description="Named model tiers (docs/rlm.md): tier name → raw LiteLLM "
        "model string, resolved late at the surface chokepoint so a config "
        "remap retargets every future step. The ``rlm_*`` tools accept tier "
        "names only (e.g. ``sub``, ``verify``); ``tier:<name>`` is the "
        "model-string scheme. Values may not themselves be ``tier:`` or "
        "``workflow:`` strings (validated at startup) so the model-binding "
        "privilege guards cannot be bypassed by indirection. "
        'Env form: AIOS_MODEL_TIERS=\'{"sub": "openrouter/…", …}\'.',
    )
    rlm_max_depth: int = Field(
        default=2,
        ge=0,
        description="Default recursion depth budget for ``rlm_query``/"
        "``rlm_map`` children of a root session (down-counted on the request "
        "edge; a child at depth 0 cannot spawn). Enforced in the dispatch "
        "path, never the prompt (docs/rlm.md).",
    )
    rlm_max_children_per_step: int = Field(
        default=8,
        ge=1,
        description="Cap on rlm children spawned by the tool calls of a single "
        "assistant step of the root session (rlm_map counts each fan-out "
        "child). Exhaustion returns a structured tool error.",
    )
    rlm_max_total_child_tokens: int = Field(
        default=500_000,
        ge=1_000,
        description="Token budget for the rlm child subtree per root turn "
        "(harvested children's total tokens, accumulated on the root ledger; "
        "admission refuses new spawns once crossed). Token-based rather than "
        "USD so unpriced/self-hosted tier models still budget honestly.",
    )
    workflow_max_inflight_children_per_run: int = Field(
        default=8,
        ge=1,
        description="Per-run cap on concurrently in-flight ``agent()`` children. A "
        "single workflow wake opens its emitted frontier in waves: at most this many "
        "new agent() children are spawned per step while that many are already "
        "in-flight; the rest are journaled as ``frontier_deferred`` markers (not "
        "spawned) and admitted on later wakes as earlier children resolve and free "
        "slots. Orthogonal to ``workflow_max_agent_calls`` (the lifetime ceiling) and "
        "to ``worker_concurrency`` (cross-run worker parallelism): this bounds a "
        "single run's standing agent() fan-out. The re-drive uses the existing "
        "child-completion re-wake — no new machinery.",
    )
    cancel_cascade_enabled: bool = Field(
        default=True,
        description="Kill switch for seeding edge-owned lifetime cancellation carriers.",
    )
    api_lease_deadline_seconds: float = Field(
        default=6 * 60 * 60,
        gt=0,
        description="Dry-run API lease abandonment horizon (separate from workflow deadlines).",
    )
    lease_ceiling_seconds: float = Field(
        default=60 * 60,
        gt=0,
        description="Maximum zero-lease lifetime for ephemeral sessions.",
    )
    invariant_sweep_dry_run: bool = Field(
        default=True,
        description="Keep C5 invariant reconciliation measurement-only until its cohort is reviewed.",
    )
    invariant_sweep_interval_seconds: int = Field(
        default=10 * 60,
        ge=5 * 60,
        le=15 * 60,
        description="Low-cadence interval for the edge-owned-lifetime invariant sweep.",
    )
    workflow_agent_deadline_seconds: float = Field(
        default=60 * 60,  # 1 hour
        gt=0,
        description="Wall-clock budget for a single ``agent()`` call, measured "
        "from its ``call_started``. A child that never goes idle (e.g. loops on "
        "tools forever) never trips the quiescence nudge, so without this its "
        "caller would suspend forever; past the deadline the harvest writes a "
        "``timeout`` error response (exactly-once), resolving the call as a "
        "catchable ``AgentError`` and keeping the run total. Generous by default "
        "— a child doing real model+tool work can legitimately run for minutes; "
        "this is the never-resolves backstop, not a tight SLA.",
    )
    workflow_call_llm_stale_seconds: float = Field(
        default=1200.0,  # 20 minutes
        gt=0,
        description="Re-dispatch horizon for an inflight ``call_llm()`` capability "
        "(#1706). A parked ``call_llm`` frame is worker-task-backed like ``tool``/"
        "``agent``: the worker runs the inference and writes the result signal only "
        "on completion. A worker crash mid-inference (restart / redeploy / OOM / "
        "graceful-shutdown ``CancelledError``) leaves an open ``call_started`` with "
        "no signal and no external resume, so — like ``tool`` — it needs the stale "
        "backstop to re-wake and re-dispatch (at-least-once). ``gate`` stays NULL-"
        "mapped because it is external-resume-driven, but ``call_llm`` is not. The "
        "in-worker ``call_llm`` deadline lives inside the task, so a kill removes it "
        "too; this horizon is the only recovery. Default is long enough to exceed "
        "any healthy in-worker call, so the sweep never re-drives a live inference.",
    )
    trace_max_nodes: int = Field(
        default=2000,
        ge=1,
        description="Node-count ceiling for a single ``/trace`` walk (#1149). "
        "#1124's depth counter bounds path length (≤10 by construction) but NOT "
        "the node count — ``workflow_max_agent_calls`` defaults to 1000 lifetime "
        "``agent()`` children per run, and fan-out caps are concurrency, not "
        "totals. So the trace walk carries this explicit ceiling: once this many "
        "nodes have been emitted, the frontier is cut and the response carries a "
        "typed ``truncated: {at_nodes}`` marker rather than silently dropping "
        "tail subtrees or running unbounded. Tunable upward for deliberately "
        "deep-audit traces.",
    )
    workflow_runs_per_launcher_max: int = Field(
        default=20,
        ge=1,
        description="Per-launcher-session ceiling on OUTSTANDING (non-terminal) "
        "runs — the horizontal fan-out bound on the agent's ``call_workflow`` "
        "builtin (the operator/HTTP path has no launcher and is exempt). A "
        "concurrency cap, not a rate or lifetime budget: slots free as runs "
        "reach a terminal status, so a sequential launch loop is unbounded by "
        "design. Enforced with the per-account cap under one advisory lock; "
        "on exceed, the launch raises ``RateLimitedError``.",
    )
    session_open_goals_max: int = Field(
        default=10,
        ge=1,
        description="Per-session ceiling on OUTSTANDING self-goals — the open "
        "self-referential awaited obligations opened by the ``create_goal`` builtin "
        "(#1508). A self-goal pins a definition-of-done the session is held to (it "
        "cannot quiesce while one is open), so an unbounded count would let a "
        "session paint itself into a corner it can never satisfy AND inflate the "
        "reserved obligations-tail budget. A concurrency cap, not a lifetime budget: "
        "closing a goal with ``return``/``error`` (its goal_id as request_id) frees "
        "slots, so a sequential goal loop is unbounded by design. On exceed, "
        "``create_goal`` returns a clear tool error (no obligation opened). Matched "
        "to ``MAX_RENDERED_OBLIGATIONS`` so the open "
        "self-goals always render as full lines in the tail block.",
    )
    workflow_runs_per_account_max: int = Field(
        default=100,
        ge=1,
        description="Per-account ceiling on OUTSTANDING (non-terminal) runs "
        "across all launchers, operator launches included — the backstop the "
        "per-launcher cap can't provide (many sessions, or a run's agent() "
        "children, are each fresh launchers). Same concurrency-not-budget "
        "semantics; worst-case standing exposure is this many runs each "
        "entitled to ``workflow_max_agent_calls`` children.",
    )
    workflow_wake_batch_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Coalescing window for run wakes from child completions and "
        "tool results: completions landing within the window collapse into one "
        "re-drive (the scheduled wake occupies the dedup lock, so further defers "
        "are absorbed — including, for the window's duration, otherwise-immediate "
        "wakes like a gate resume; their signals are still harvested by the "
        "delayed step). 0 disables (every completion wakes immediately — today's "
        "behavior). Note the effective floor: a not-yet-due scheduled job is "
        "picked up by worker polling (~5s on an otherwise-idle worker), so "
        "sub-second windows buy nothing.",
    )

    # ── connectors ─────────────────────────────────────────────────────────
    inbound_debounce_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Debounce window (seconds) before the FIRST wake of an "
        "idle session on a connector inbound. aios already coalesces "
        "mid-turn arrivals, but the very first wake of an idle session "
        "fires immediately, so a bursty sender that jitters several "
        "messages a few seconds apart (e.g. Jarvis, 2-5s between sends) "
        "spawns one turn per message instead of one turn for the burst. "
        "When > 0, the inbound path schedules the wake this many seconds "
        "out via ``defer_wake``'s ``delay_seconds``; the existing "
        "``queueing_lock=session_id`` dedup then collapses any follow-on "
        "inbounds that land inside the window into the same single wake. "
        "0.0 (the default) disables debounce — the wake fires immediately, "
        "preserving pre-feature behavior. Fixed delay, not random jitter "
        "(v1). Only the connector-inbound first wake is affected; the "
        "harness retry-backoff path (cause='reschedule') and the async "
        "tool-completion path (cause='connector_tool_result') are "
        "untouched.",
    )
    outbound_tool_quotas: dict[str, OutboundToolQuota] = Field(
        default_factory=dict,
        description="Per-tool rolling outbound dispatch quotas as "
        "tool_name -> (window_seconds, max_per_window). Empty (default) disables "
        "quota checks and performs no dispatch-path query. Connector verbs are "
        "matched without their MCP server namespace.",
    )
    inbound_rate_window_seconds: int = Field(
        default=0,
        ge=0,
        description="Rolling-window length (seconds) for the per-counterparty "
        "inbound rate/cost budget (#1504) enforced at the connector-write "
        "boundary. The budget is a per-``(connection_id, chat_id)`` rolling "
        "COUNT of admitted inbound messages over this window, keyed on the "
        '``orig_channel`` (``f"{connector}/{external_account_id}/{chat_id}"``) '
        "stamped on each ``role=user`` event, ``account_id``-scoped (so two "
        "tenants holding the same external ``chat_id`` get independent "
        "budgets). When an admitted counterparty exceeds "
        "``inbound_rate_max_per_window`` messages inside this window the "
        "inbound (or wake-bearing lifecycle) call is dropped BEFORE any append "
        "+ ``defer_wake`` — zero session spawn, zero event row, zero wake — as "
        "a routine non-fatal 4xx (429) that never crash-restarts the "
        "connector container. 0 (the default) DISABLES the budget: the helper "
        "short-circuits to admit with no query, a path byte-identical to "
        "pre-feature behavior (mirrors ``inbound_debounce_seconds``). v1 "
        "enforces message-rate only; the token/cost variant is a "
        "strictly-additive extension of the same window query (see #1504 "
        "Out of scope).",
    )
    inbound_rate_agent_window_seconds: int = Field(
        default=0,
        ge=0,
        description="Rolling-window length for the per-agent inbound budget. 0 disables it.",
    )
    inbound_rate_agent_max_per_window: int = Field(
        default=0,
        ge=0,
        description="Maximum inference-bearing inbounds per agent window. 0 disables it.",
    )
    inbound_rate_max_per_window: int = Field(
        default=0,
        ge=0,
        description="Per-window admitted-inbound message threshold for the "
        "per-``(connection_id, chat_id)`` inbound rate/cost budget (#1504), "
        "counted over ``inbound_rate_window_seconds``. A counterparty whose "
        "rolling count reaches this threshold is throttled (the next inbound / "
        "wake-bearing lifecycle wake is dropped before any append + wake). "
        "0 (the default) DISABLES the budget — byte-identical to pre-feature "
        "behavior, no extra query (mirrors ``inbound_debounce_seconds``'s "
        "disabled-by-default convention). Both knobs must be > 0 for the "
        "budget to engage. v1 is message-rate; the token/cost variant reuses "
        "the same window with a different aggregate (see #1504 Out of scope).",
    )
    connector_backfill_max_age_seconds: int = Field(
        default=60 * 60,  # 1 hour
        ge=60,
        description="Age ceiling, in seconds, on connector sends surfaced by the "
        "SSE subscribe-time backfill (``list_pending_calls_for_connector``). A "
        "pending connector send whose parent assistant turn is older than this is "
        "SKIPPED — excluded from the backfill, not expired (the event log is left "
        "untouched). A legitimately in-flight send executes within seconds of the "
        "connector reconnecting, so the default 1h never touches a real one; the "
        "bound only suppresses dormant, weeks-stale sends from being re-transmitted "
        "on a worker restart (#744). Does NOT affect ``Session.awaiting`` / the "
        "unresolved-tool-call read-model, which surfaces all unresolved calls "
        "regardless of age (#741).",
    )
    confirmed_dispatch_max_age_seconds: int = Field(
        default=60 * 60,  # 1 hour
        ge=60,
        description="Age ceiling, in seconds, on the confirmation of an "
        "operator-confirmed-but-unresolved tool call surfaced for auto-dispatch "
        "on resume (``list_confirmed_unresolved_tool_calls`` / sweep "
        "``CONFIRMED_ROWS_SQL``). A confirmed call whose ``tool_confirmed`` event "
        "is older than this is SKIPPED — excluded from re-dispatch, not expired "
        "(the event log is left untouched, no synthetic result). The bound is on "
        "the CONFIRM event's ``created_at``, NOT the assistant turn: an operator "
        "can confirm an OLD proposal, which is a FRESH intent to dispatch, so the "
        "dispatchable-since timestamp is when confirmation was granted. A "
        "legitimately confirmed call dispatches within seconds, so the default 1h "
        "never touches a real one; the bound only suppresses weeks-stale "
        "confirmations from re-running a side-effecting tool on a worker restart "
        "(#746). Parallel to ``connector_backfill_max_age_seconds`` (#744); both "
        "detection (sweep) and dispatch (resolver) apply this same bound in sync "
        "so they resolve the identical condition (no wake-with-no-progress loop, "
        "the #155 symptom).",
    )

    client_tool_call_max_age_seconds: int = Field(
        default=24 * 60 * 60,  # 24 hours
        ge=60,
        description="Age ceiling, in seconds, after which an unresolved "
        "CLIENT-result-pending tool call (a tool the harness never dispatches "
        "because the CLIENT executes it and returns the result — the non-MCP, "
        "not-in-registry branch of ``sweep._was_dispatched``) is treated as "
        "ABANDONED by ghost repair: a synthetic timeout/abandoned error result is "
        "appended, resolving the call. Unlike "
        "``connector_backfill_max_age_seconds`` / "
        "``confirmed_dispatch_max_age_seconds`` (which only SKIP, leaving the log "
        "untouched), this bound EXPIRES the call — without it the call's "
        "``open_tool_call_count`` contribution keeps the session a permanent wake "
        "candidate (``CANDIDATE_ROWS_SQL`` / ``_SESSION_ACTIVE_EXPR``) with no "
        "progress to make (the #155 wake-no-progress loop; regression from the "
        "open-tool-call-count predicate added in #750). The bound is on the "
        "ASSISTANT turn's ``created_at`` (when the call was emitted), consistent "
        "with the dispatched-ghost age semantics. The default 24h is deliberately "
        "generous — a legitimate slow client (a human-driven connector, a "
        "long-running custom tool) returns its result well within a day, so the "
        "bound only fires for a client that disconnected and will never return. "
        "Confirmation-pending calls (``always_ask`` tools awaiting a "
        "``tool_confirmed`` event) are EXCLUDED: those wait on the USER, not a "
        "client, and erroring them would kill a slow human-in-the-loop "
        "confirmation.",
    )

    # ── observability ──────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")

    @property
    def oauth_allow_insecure_host_set(self) -> frozenset[str]:
        """Parsed ``oauth_allow_insecure_hosts`` as a set of host[:port] entries."""
        return frozenset(h.strip() for h in self.oauth_allow_insecure_hosts.split(",") if h.strip())

    @model_validator(mode="after")
    def _external_byok_requires_account_only(self) -> Settings:
        if (
            self.tenancy_posture == "external_byok"
            and self.inference_credential_policy != "account_only"
        ):
            raise ValueError("external_byok requires account_only inference credentials")
        return self

    @model_validator(mode="after")
    def _require_absolute_workspace_root(self) -> Settings:
        if not self.workspace_root.is_absolute():
            raise ValueError(
                f"AIOS_WORKSPACE_ROOT must be an absolute path; got '{self.workspace_root}'. "
                "Note: pathlib does not expand '~'; write the full path. "
                "API and worker processes resolve relative paths against their own CWD, "
                "producing diverging session workspace_path values and ForbiddenError "
                "on tool calls."
            )
        return self

    @model_validator(mode="after")
    def _model_call_deadline_under_step_budget(self) -> Settings:
        if self.model_call_deadline_s >= HARNESS_STEP_TIMEOUT_S:
            raise ValueError(
                f"AIOS_MODEL_CALL_DEADLINE_S={self.model_call_deadline_s} must be "
                f"strictly less than the harness step budget "
                f"({HARNESS_STEP_TIMEOUT_S}s)."
            )
        return self

    @model_validator(mode="after")
    def _model_tiers_values_are_raw(self) -> Settings:
        # A tier whose value is itself a scheme string would route inference
        # around the guards that key on the literal model string: a
        # ``workflow:`` value bypasses the model-binding privilege check and
        # the call_llm leaf-only rejection; a ``tier:`` value makes resolution
        # recursive. Both are rejected loudly at startup (docs/rlm.md).
        for tier, value in self.model_tiers.items():
            if value.startswith(("tier:", "workflow:")):
                raise ValueError(
                    f"AIOS_MODEL_TIERS[{tier!r}] = {value!r}: tier values must be "
                    "raw provider model strings, never tier:/workflow: schemes"
                )
        return self

    @model_validator(mode="after")
    def _github_clone_session_timeout_under_step_budget(self) -> Settings:
        # The whole point of #697: per-session clone budget must fit
        # strictly inside the harness step budget so a hung clone can't
        # burn a whole user turn before the step-level cap fires. Reject
        # misconfiguration loudly at startup rather than letting an
        # operator silently defeat the fix.
        if self.github_clone_session_timeout_seconds >= HARNESS_STEP_TIMEOUT_S:
            raise ValueError(
                f"AIOS_GITHUB_CLONE_SESSION_TIMEOUT_SECONDS="
                f"{self.github_clone_session_timeout_seconds} must be strictly less than "
                f"the harness step budget ({HARNESS_STEP_TIMEOUT_S}s); "
                f"otherwise a hung clone "
                f"burns a whole user turn before the step-level cap fires. "
                f"See issue #697."
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide ``Settings`` singleton.

    Cached so every call site sees the same instance and env reads happen once.
    Tests that need different settings should use ``Settings(...)`` directly
    inside a fixture.
    """
    return Settings()
