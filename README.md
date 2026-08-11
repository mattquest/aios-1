# aios

**Interrupt and redirect an agent in the middle of a running tool — because it has no agentic loop.**

aios is an open-source, multi-tenant agent runtime for **assistants that live for months, not minutes**. A durable Postgres-backed job queue calls the model exactly once per step, fires every tool as a detached async task, and re-enters — so the model stays responsive to new messages even while tools run, and a user can redirect a busy agent mid-tool.

Its entire memory is **one append-only event log**. Status, spend, and "what is it waiting on" are *derived* from that log — never stored, never able to lie. Context is a strictly monotonic pure function of it (prompt-cache stable, no LLM compaction), and the agent recalls old turns with SQL.

**The model is just a [LiteLLM](https://github.com/BerriAI/litellm) model string** — `anthropic/claude-opus`, `ollama/llama3.3`, `openrouter/anything`. You self-host all of it; nothing load-bearing is hostage to a third party.

> No loop. No compaction. No lock-in. An append-only event log *is* the assistant. A call is a durable edge, not a function call.

**Project status:** alpha, actively developed, and deployed in production. ~255 unit-test files plus a Docker-backed e2e suite; `mypy --strict`, `ruff`, and OpenAPI/SDK/CLI drift-guards gate every PR and master push in CI.

---

## Contents

- [See it work](#see-it-work) · [Why aios is different](#why-aios-is-different) · [Who it's for](#who-its-for)
- [Architecture](#architecture) · [Quickstart](#quickstart) · [Resources at a glance](#resources-at-a-glance)
- [Harness & step model](#the-harness--step-model) · [Sessions, agents & events](#sessions-agents--events)
- [Workflows](#workflows--durable-replayable-orchestration) · [Invocation kernel](#the-unified-invocation-kernel) · [Triggers](#triggers--scheduling)
- [Connectors & multi-channel](#connectors--multi-channel) · [Sandboxes & environments](#durable-sandboxes--environments)
- [Tools, MCP & permissions](#tools-mcp--permissions) · [RLM (this fork)](#rlm--context-variables--recursive-sub-queries-this-fork) · [Memory & skills](#memory-stores--skills) · [Vaults](#vaults--credential-injection)
- [Security model](#security-model) · [Accounts & multi-tenancy](#accounts-multi-tenancy--spend) · [API, CLI & SDK](#api-cli--sdk--one-surface-three-faces)
- [Multimodal & files](#multimodal-files--attachments) · [vs. Anthropic Managed Agents](#vs-anthropic-managed-agents) · [Roadmap](#roadmap--where-this-is-heading) · [License](#license)

---

## See it work

Send a second message while a 90-second tool from the first is still running — the agent acknowledges and reprioritizes instead of blocking:

```
you ▸ scrape the last 200 issues from the repo and summarize the recurring themes
bot ▸ [calls bash → a long gh + jq pipeline starts running as a detached task]

you ▸ actually — just the ones labeled "bug", and tell me the top 3 first
bot ▸ Got it, narrowing to bug-labeled issues and prioritizing the top 3.
      (the original scrape is still running; this message was processed by the
       very next step, not after the tool returned)
```

Controller-loop frameworks (LangChain, LangGraph, CrewAI, AutoGPT) block the turn until each tool returns. aios calls the model **once** per step, launches each tool as a fire-and-forget asyncio task, sets `stop_reason=end_turn`, and returns — releasing its lock instantly. A durable [procrastinate](https://procrastinate.readthedocs.io/) job re-entering the step function is the only "loop."

---

## Why aios is different

Three properties no task-scoped framework has, each a direct consequence of the no-loop spine:

**Tools never block the model.** A user message landing mid-tool is processed by the very next step. You can redirect an agent in the middle of a 90-second fetch, fan out a dozen sub-agents without freezing the conversation, and cancel in-flight work cleanly — because a tool call is a detached task, not a blocking call inside a controller loop.

**The event log *is* the assistant — derived state can't lie.** One append-only, gapless Postgres journal per session is the single source of truth. There is no stored status column: `active/idle/archived/errored`, token spend, `awaiting` ("what is it blocked on"), and `obligations` ("what does it owe an answer to") are all SQL arithmetic over the log. The read path and the worker's wake sweep share the **identical** predicate, so they cannot drift — designing out a whole class of "worker wakes with no work / skips a session" bugs. Context is a strictly monotonic function of the log, so the prompt prefix cache stays hot, and scrolled-out history is recalled losslessly with the `search_events` SQL tool.

**Durable, replayable workflows whose orchestration spends zero model tokens.** A workflow is the dual of an agent: deterministic Python where the model would be. A run's entire state is an append-only journal, replayed from memo on each wake — crash-, deploy-, and month-long-suspension-durable — while the orchestration logic itself costs **nothing**. The replayed steps are real LLM-agent invocations; the glue between them is free.

The rest of the system follows from the same primitives:

| Pillar | What it buys you |
|---|---|
| **Unified invocation kernel** | Any caller (a model mid-conversation, an HTTP client, a workflow) invokes any servicer (a peer session, a fresh agent, a durable run) through **one** durable request edge. Recursive cancel, depth-budgeting, and a one-call causal trace fall out for free. |
| **Capability attenuation** | A child's authority is the lattice *meet* of its declared surface with its launcher's already-frozen one — frozen at the spawn edge, non-widening by construction. Even *which inference endpoint its mind talks to* is frozen at the spawn edge. |
| **Durable sandboxes** | Each session's container persists its **entire** filesystem across months via stop → commit → resume. The agent runs as root inside, yet can never read the credentials it authenticates with. |
| **One mind, every channel** | A single session reachable across Signal, Telegram, Slack, WhatsApp, and HTTP at once, with a `switch_channel` focal-attention primitive. |
| **Self-scheduled triggers** | cron / one-shot deadline / workflow-completion / authenticated webhook × bash / workflow / wake — a 4×4 product space the agent provisions itself. |
| **Hierarchical multi-tenancy** | Account tree with per-account HKDF crypto isolation. Spend limits inherit *down*; dollar spend rolls *up*; every step admits pre-flight against the summed subtree spend. |
| **Provider-agnostic** | The model is any LiteLLM model string, with correct dual-channel prompt caching and thinking-block preservation driven by family/substring rules — no per-model shims. |
| **Self-improving agents** | An agent authors and version-bumps its own skills and durable workflows from inside the session — bounded by capability, never by trusting model input. |
| **Security model** | Per-account HKDF crypto, a separately-keyed egress CA, two credential paths the model can never read, an attenuation lattice, and fail-closed sandbox sidecars. See [Security model](#security-model). |
| **One API, three faces** | Operator REST, an auto-reflected MCP server, and a code-generated SDK from one source of truth, with CI drift-guards. |

---

## Who it's for

aios is for **sovereign, self-hosted, long-lived assistant entities** — agents meant to run for months across an owner's channels and credentials, with real multi-tenancy and durable execution.

It's a good fit if you want Anthropic Managed Agents' architecture as open, auditable, self-hostable code; you're building a multi-tenant hosted agent product that needs cryptographic tenant isolation and subtree spend ceilings; you've outgrown in-process agent loops and need crash-recoverable, SQL-queryable execution; or you want the model to be just a LiteLLM model string.

**When *not* to use it:** if you want a stateless task-runner for one-shot jobs, or a lightweight in-process library to embed in an existing app, aios is overkill — it's infrastructure for persistent entities, not a function call.

---

## Architecture

aios is an event-driven runtime split across two processes sharing one Postgres (same DB, same `LISTEN/NOTIFY`, same job queue — no Redis, no broker).

```
                            ┌─────────────────────────────────────────────┐
   Signal / Telegram        │                  Postgres                    │
   Slack / WhatsApp ──┐     │   ┌───────────┐  ┌──────────┐  ┌──────────┐  │
   HTTP / SSE         │     │   │ event log │  │ job queue│  │LISTEN/   │  │
                      │     │   │(append-   │  │(procras- │  │NOTIFY    │  │
   ┌──────────────┐   │     │   │ only,     │  │ tinate)  │  │          │  │
   │  connectors  │   │     │   │ gapless)  │  │          │  │          │  │
   │ (out-of-proc,│   │     │   └───────────┘  └──────────┘  └──────────┘  │
   │  cred-isolated)──┐│     └───────▲──────────────▲────────────▲─────────┘
   └──────────────┘  ││             │              │            │
                     ││     ┌───────┴──────┐  ┌────┴───────────┴─────────┐
   ┌──────────────┐  └┼────▶│  API process │  │      Worker process      │
   │ operator /   │   │     │  (aios api)  │  │      (aios worker)       │
   │ SDK / MCP /  │───┘     │              │  │                          │
   │ CLI          │         │ • append     │  │ • run_session_step       │
   └──────────────┘         │   events     │  │   (model ONCE per step)  │
                            │ • defer wakes│  │ • launch_tool_calls      │
                            │ • serve SSE  │  │   (detached asyncio)     │
                            │ • NO model   │  │ • run_workflow_step      │
                            │ • NO tools   │  │   (deterministic replay) │
                            └──────────────┘  │ • sandbox mgmt           │
                                              │ • 30s sweep + scheduler  │
                                              └────────────┬─────────────┘
                                                           │
                                              ┌────────────▼─────────────┐
                                              │  per-session Docker       │
                                              │  sandbox (durable rootfs, │
                                              │  egress-locked, secrets   │
                                              │  swapped at TLS boundary) │
                                              └───────────────────────────┘
```

- **API process** (`aios api`): FastAPI server. Appends user messages, defers wake jobs, serves SSE streams. Does **not** call the model or run tools.
- **Worker process** (`aios worker`): Runs procrastinate jobs — model calls, tool dispatch, sandbox management, the 30s periodic sweep, the trigger scheduler, the interrupt listener.

**The step flow**: a wake job runs the step function, which calls the model once, fires tools as detached tasks, sets `end_turn`, and returns; each tool task appends its result and defers the next wake; the job queue re-enters. That re-entry is the loop.

### Key invariants

1. **Gapless seq per session** — every append locks the session row, increments `last_event_seq`, inserts. No gaps.
2. **Monotonic context** — appending events only appends to the context, never rewrites earlier messages. Prompt cache stays stable.
3. **`reacting_to` watermark** — every assistant message records the max seq of user/tool events it saw, so the sweep knows what's "new" without locks or polling.
4. **Tool-always-appends-result** — every async tool task appends exactly one result event and defers a wake (enforced structurally by a partial unique index).
5. **Procrastinate lock** — `lock=session_id` for mutual exclusion, `queueing_lock=session_id` for wake dedup.
6. **NOTIFY after commit** — `pg_notify` fires outside the transaction so subscribers never see uncommitted rows.

### Design philosophy

- **No agent loop.** The model is called once per step; the durable job queue re-enters. Tools never block the model.
- **Derived state can't lie.** Status, spend, and obligations are SQL over the log. The read path and the worker sweep share the *identical* predicate so they cannot drift.
- **Correct-by-construction over corrective.** Gapless seq, monotonic context, tool-always-appends-result, the attenuation lattice — illegal states are unrepresentable, not runtime-guarded.
- **Compose, don't accrete.** Variation is encoded as a *kind* (a discriminated arm), never a boolean flag — the trigger `source × action` matrix and the invocation `Ask|Tell × New|Existing` union are the models.
- **Fail hard, no fallbacks.** The model sees raw errors and retries through the session log; that IS the recovery design.
- **Extreme simplicity.** No defensive guards for model mistakes, no fuzzy matching, no per-model shims.

---

## Quickstart

aios needs Python 3.13+, [uv](https://docs.astral.sh/uv/), and Postgres. Docker is needed for the sandbox and for E2E tests.

```bash
# Install dependencies
uv sync --dev

# Configure (see Environment variables below) — minimally:
#   AIOS_DB_URL, AIOS_VAULT_KEY, AIOS_EGRESS_CA_KEY, AIOS_BOOTSTRAP_TOKEN
# Provider credentials are encrypted model-provider rows on non-root accounts.
# Use `aios model-providers create`; the platform root is credentialless.
# Env keys are migration-only: AIOS_INFERENCE_CREDENTIAL_POLICY=legacy_env.
set -a && source .env && set +a

# Run migrations (also applies the procrastinate schema + lock-release trigger;
# bare `alembic upgrade head` does NOT)
uv run aios migrate

# Start the two processes
uv run python -m aios api      # FastAPI server on :8080 (AIOS_API_PORT)
uv run python -m aios worker   # procrastinate worker
```

On a fresh DB, **mint the root account key** (auth hashes-to-a-row; there is no env-var compare):

```bash
# Sends AIOS_BOOTSTRAP_TOKEN as bearer; prints the once-only plaintext key.
uv run aios accounts bootstrap --display-name root
# Store the returned plaintext_key as AIOS_API_KEY for the API service and clients.
```

Then create an environment + agent and chat:

```bash
uv run aios envs create --file env.json
uv run aios agents create --file agent.json

# Interactive REPL (creates a session, streams the reply)
uv run aios chat --agent <agent_id> --environment-id <env_id>

# One-shot: send a message and stream until the turn ends
uv run aios chat --agent <agent_id> --environment-id <env_id> -m "list /workspace"
```

### Deploy with Docker Compose

`compose.yml` brings up the full stack in one command — `postgres`, a migrate step, `api`, `worker`, and an `echo-http` reference connector — with optional profiles for the platform connectors. The sandbox base image is published at `ghcr.io/eumemic/aios-sandbox` (built from `docker/Dockerfile.sandbox` with an authored seccomp profile).

```bash
docker compose up                          # postgres + migrate + api + worker + echo-http
docker compose --profile telegram up       # add the Telegram connector
```

### Worktree dev isolation

Every git worktree gets its own isolated DB on the shared local Postgres:

```bash
uv run aios dev bootstrap   # provisions aios_dev_<id> + writes a worktree-local .env
set -a && source .env && set +a
uv run aios dev status      # expect: mode: isolated
```

`aios dev status` prints a `mode:` line — `isolated`, `shared` (a linked worktree pointed at the shared DB — fix it), or `unbootstrapped`. As a backstop, `aios api`/`aios worker` hard-fail when started from a linked worktree against the shared DB unless `AIOS_ALLOW_SHARED_DB=1`.

### Checks

```bash
uv run mypy src tests
uv run ruff check src tests && uv run ruff format --check src tests
uv run pytest tests/unit -q                                   # ~255 files, fast, no Docker
DOCKER_HOST=unix://... uv run pytest tests/e2e -q             # needs Docker
```

---

## Resources at a glance

aios exposes its entire runtime through one versioned REST API (`/v1/...`, FastAPI, bearer auth) over these account-scoped resource families:

| Resource | What it is |
|---|---|
| **accounts** | Hierarchical multi-tenant tree; bearer keys, spend rollup, delegated minting. |
| **agents** | Immutable-versioned config: model + system prompt + tools + skills + MCP/HTTP servers. |
| **sessions** | A running agent instance and its append-only event log; the durable identity. |
| **events** | The append-only journal per session (message / lifecycle / span / interrupt). |
| **environments** | Reusable sandbox template: base image, packages, network policy, env, budgets. |
| **skills** | Versioned, progressively-disclosed knowledge bundles (`SKILL.md` + files). |
| **vaults** | Encrypted credential collections; two injection paths the model can't read. |
| **memory-stores** | Versioned, audit-trailed, path-addressed text mounted at `/mnt/memory/`. |
| **connections** | One platform account → a routing target (single session or per-chat template). |
| **connectors** | Root-owned per-type catalog (tools schema + typed capability descriptor). |
| **session-templates** | Frozen recipe for per-chat session spawn. |
| **triggers** | Per-session `source × action` scheduled/reactive edges. |
| **runtime-tokens** | Per-connector-type bearers for connector containers (optionally allowlisted). |
| **tasks** | The kind-agnostic request edge (caller → servicer). |
| **workflows / runs** | Durable, replayable deterministic-Python orchestration definitions + instances. |
| **files / github-repos** | Session-scoped uploads and git-repo mounts (encrypted clone token). |

---

## The harness & step model

**What it buys you:** an agent that stays live while tools run, recovers itself from any crash short of SIGKILL, and never falls into a runaway loop.

The harness turns the event log into a running agent. There is no controller loop.

- **No-loop step model** — the step function calls the model once, launches tool calls as detached asyncio tasks, sets `stop_reason=end_turn` unconditionally, and returns. A hard wall-clock cap (960s) is the final zero-hang safety net.
- **Implicitly-async tools with mid-turn injection** — tasks outlive the job body. End-of-step flips status to idle regardless of pending tool calls, so a user message arriving during inference or tool execution is just another event the next step's gate picks up.
- **Monotonic context builder** — the message list is a pure function of the windowed log. In-flight calls get synthetic `pending` results; results that completed during inference are re-injected as synthetic user messages at the tail (blind-spot injection), preserving prefix monotonicity.
- **Deterministic chunked stable-prefix windowing** — the context cutoff snaps forward in discrete chunks (defaults 50k/150k tokens). No per-turn slide (cache-busting), no LLM summarization (lossy). A head omission marker tells the model how much scrolled out and that `search_events` recalls it.
- **Tail-injected obligations block** — every open awaited request is rendered each step as an ephemeral, cache-safe, last-user-role block (capped at 10 lines), sourced from a full-log query so it survives windowing erasure of the original ask.
- **Consecutive-inaction request nudge** — the retry budget counts nudges only *since the latest tool-call turn* — a stuck-detector (budget 3), not a lifetime loop-limiter. An agent doing real work never trips it; one stuck doing nothing N turns running gets a `no_return`. This is the keystone that makes always-on agents safe.
- **Self-healing sweep** — the inference-need check runs at step entry, in every tool task's `finally`, and on a 30s periodic loop; ghost-repair synthesizes results for tool calls whose worker vanished. A SIGKILL'd worker or dropped NOTIFY can't permanently wedge a session.
- **Poison-event quarantine** — a single event whose render raises is replaced by a deterministic placeholder (a function of its seq only), degrading exactly one position instead of permanently bricking a months-long session.
- **Per-token SSE streaming over `pg_notify`** — content deltas stream with zero extra storage, and the worker skips the slower streaming path when no subscriber is attached (advisory-lock probe).
- **Provider-quirk normalization** — one cached model descriptor drives dual-channel prompt caching (Anthropic content-block markers vs OpenAI cache keys, mutually exclusive by construction), thinking-block preservation across replay, and refusal handling — via family/substring rules, no per-model lists.

<details>
<summary><b>Harness configuration</b></summary>

| Var | Purpose |
|---|---|
| `AIOS_MODEL_CALL_DEADLINE_S` | Single model-call deadline (default 900s; below the 960s step timeout). |
| `AIOS_WORKER_CONCURRENCY` | Concurrent session steps per worker (default 4). |
| `AIOS_TOOL_RESULT_MAX_CHARS` | Inline tool-result cap (default 200k); larger results spill to a readable attachment file with an inline stub. |
| `AIOS_DEFAULT_SPEND_LIMIT_USD` | Default effective spend ceiling; the step's pre-flight admission latches errored on a subtree breach. |
| `AIOS_INBOUND_DEBOUNCE_SECONDS` | Debounce connector-inbound wakes so rapid messages collapse into one step. |
| `AIOS_OUTBOUND_TOOL_QUOTAS` | JSON map of connector verb to `[window_seconds, max_per_window]`; empty by default (for example `{"matrix_invite":[3600,20],"matrix_create_room":[3600,20],"matrix_join":[3600,20],"matrix_send":[3600,500]}`). Calls at the cap become model-visible `quota_exceeded` tool errors before connector publish. Keep homeserver/appservice rate limiting enabled as an independent backstop; upstream `M_LIMIT_EXCEEDED` errors are surfaced without retries. |
| `AIOS_DUMP_CONTEXT` / `AIOS_DUMP_CONTEXT_DIR` | Dump the exact chat-completions payload sent to LiteLLM per step. |

</details>

---

## Sessions, agents & events

**What it buys you:** a durable identity that outlives the agent config that drives it — rebind its model mid-life, fork it at head, and never lose a turn.

- **Append-only event log as durable truth** — the chat-completions message dict is stored opaquely so `reasoning_content` / `thinking_blocks` / provider extensions round-trip. Corrections are new events, never rewrites.
- **Derived status, no status column** — `active/idle/archived/errored` is column arithmetic over five maintained watermark scalars. The read path and the worker sweep share the identical predicate generators, so they cannot drift. An errored session auto-recovers the instant a user message lands.
- **Immutable agent versioning** — every update mints a full snapshot into `agent_versions` with optimistic concurrency (the loser of a race gets a clean 409). A session pins a version (`agent_version: int`) or floats on latest (`null`).
- **Sessions outlive agents** — rebind `agent_id` / `agent_version` / `model` without losing a single turn of history. The identity is the *session*, not the agent config.
- **Clone/fork at head** — copy the full event prefix (plus vaults, resources, triggers with counters reset) into a fresh session yielding a byte-identical next-step context. A/B a different model from an identical history.
- **`archive_when_idle`** — self-reclaiming one-shot sessions that soft-archive the first time they go idle owing nothing (workflow children launch with this set).
- **Outbound-suppression mode** — reads pass through to real credentials while writes return a *synthesized* success the agent can't distinguish from production, with an audit span. The atomic flip-v1→v2 lever for parallel-run cutovers.
- **Two derived views** — `awaiting` (tool calls the session is blocked on) and `obligations` (requests it must answer), both computed from the log and surviving context-windowing erasure.

<details>
<summary><b>Session & agent endpoints</b></summary>

| Method | Path | Description |
|---|---|---|
| POST | `/v1/sessions` | Create (agent + env; optional version pin, vaults, resources, triggers, initial message). |
| GET | `/v1/sessions` | Keyset-paginated list with derived-status filters. |
| GET | `/v1/sessions/{id}` | Read view: derived status, `stop_reason`, `awaiting`, `obligations`, usage. |
| PUT | `/v1/sessions/{id}` | Rebind agent/version/model; flip outbound suppression. |
| POST | `/v1/sessions/{id}/clone` | Fork at head (idle parents only). |
| POST | `/v1/sessions/{id}/archive` | Soft-archive (terminal). |
| POST | `/v1/sessions/{id}/messages` | Append a user message and defer a wake. |
| POST | `/v1/sessions/{id}/interrupt` | Cancel in-flight work; status re-derives honestly. |
| GET | `/v1/sessions/{id}/context` | Byte-identical dry-run of the next chat-completions payload. |
| GET | `/v1/sessions/{id}/trace` | One-call linear trace of the session + nested runs/sessions. |
| POST | `/v1/agents` … `/v1/agents/{id}/versions/{version}` | Full agent CRUD + immutable version history. |

</details>

<details>
<summary><b>Session & agent CLI</b></summary>

```
aios sessions list | get | create | update | archive | clone | delete | send | interrupt
aios sessions events | profile | stream | tail | tool-result | tool-confirm
aios agents list | get | create | update | archive | versions | version
```

</details>

`aios sessions profile <id> [--turns N]` reconstructs per-phase latency (sweep / context-build / model-request vs tool-execute / queue-wait gaps) purely from span events.

---

## Workflows — durable, replayable orchestration

**What it buys you:** crash-, deploy-, and month-suspension-durable orchestration whose glue logic spends zero model tokens.

A workflow is the literal **dual of an agent**: deterministic Python where the model would be. A *run* is a durable execution instance whose entire state lives in an append-only journal. Each wake re-executes the author script from the top, replaying memoized capability results until it reaches the next unresolved one — **replay-from-memo**. No model is called inside a step; the actual LLM work happens only inside the `agent()` children the script spawns. **The orchestration logic itself spends zero model tokens.**

- **Replay-from-memo durable step** — a single journal writer allocates a gapless seq serialized by the per-run procrastinate lock (`lock=run_id`), idempotent on `(run_id, call_key, type)` via a `UNIQUE NULLS NOT DISTINCT` constraint, so a replayed append or procrastinate dual-execution collides and no-ops. A crash anywhere re-wakes to a valid state.
- **Credential-free out-of-process host** — the script runs in a fresh `python -m aios.workflows.wf_script_host` subprocess under a deny-by-default env allowlist that never inherits the master `CryptoBox`, the all-accounts pool, or any `*_API_KEY`. **The subprocess boundary — not a builtins allowlist — is the security perimeter:** even a full Python sandbox escape in author code (which may be agent-written) reaches zero tenant secrets. The parent enforces a wall-clock `SIGKILL` deadline plus memory/CPU rlimits.
- **Content-addressed determinism** — canonical-JSON encoding rejects NaN/Inf/sets/bytes at the call site (loud author error, never silent hash desync); per-content-hash call keys make divergence content-local; `PYTHONHASHSEED=0` and a pinned host-semantics epoch make a months-suspended run safe to resume on any worker.
- **Capability API** — a small orthogonal set composes the full space:

  | Capability | Meaning |
  |---|---|
  | `agent(input, agent_id=, output_schema=, model=)` | Invoke a child LLM **session**, await its `return`/`error`. |
  | `invoke_workflow(workflow_id, input)` | Invoke another workflow as a sub-**run**. |
  | `tool(name, input)` | Invoke a declared tool; tool errors are *values*, never raises. |
  | `gate()` | Suspend until an external resume delivers a value (human-in-the-loop). |
  | `budget()` | Read the run's shared child-spend budget. |
  | `parallel(thunks)` / `pipeline(items, *stages)` | Fan out concurrent branches with deterministic branch-local keys. |
  | `log()` / `phase()` | Journaled progress annotations (emit-once across replays). |

- **Surface attenuation clamp** — a run snapshots its surface at launch; an `agent()` child wields `agent-surface ∩ run-surface`, frozen at spawn. By associativity, a single meet against the launcher equals the whole-chain fold. A down-counting depth budget (`INVOKE_MAX_DEPTH=10`) bounds recursion with one integer.
- **At-least-once tool execution with idempotency** — a deterministic per-call token `sha256(run_id‖call_key)` is exported as `$AIOS_IDEMPOTENCY_KEY` in bash and substituted for a sentinel `Idempotency-Key` header in `http_request`, so non-idempotent POSTs survive crash re-drives.
- **Park-and-harvest concurrency** — a run never holds its lock/slot while a capability is outstanding; thousands of children/gates can be in flight without pinning workers.
- **Wave-admitted fan-out** — bounded along every axis: per-run concurrency (`max_inflight_children_per_run=8`), lifetime (`max_agent_calls=1000`), fan-out width (`MAX_PARALLEL_FANOUT=1000`), per-account/per-launcher outstanding-run caps, and a fail-closed replay-prefix divergence check.
- **Create-time AST validation** — the declared tool/agent surface must be a *superset* of the script's literal `tool()`/`agent()` calls, caught at authoring time as a clean 4xx instead of a silently-clamped runtime route-mismatch.

> **Note on examples:** `src/aios/workflows/deep_research.py` is a CI test-fixture builder, not a runnable example. Treat it as a reference for the shape (scouts → readers → synthesis → critic via `parallel`/`pipeline`/`gate`), not a deployable script.

<details>
<summary><b>Workflow endpoints, CLI & config</b></summary>

| Method | Path | Description |
|---|---|---|
| POST/PUT/GET | `/v1/workflows[/{id}]` | Create / version-update / read definitions. |
| GET | `/v1/workflows/{id}/versions[/{version}]` | Immutable version history. |
| POST | `/v1/runs` | Launch a run (binds env, vaults, budget, default child model). |
| GET | `/v1/runs/{id}` | Run in full: pinned script, status, input/output, per-run usage. |
| GET | `/v1/runs/{id}/events` | Page the run's journal. |
| GET | `/v1/runs/{id}/trace` | One-call DFS trace of the run + all nested sessions/sub-runs. |
| POST | `/v1/runs/{id}/resume` | Resume a suspended gate by `gate_nonce`. |
| GET | `/v1/runs/{id}/stream` | SSE journal stream, ending on `run_completed`. |

```
aios workflows list | get | create | update | archive | unarchive | versions
aios runs create | list | get | wait | events | stream | cancel | resume
```

| Var | Purpose |
|---|---|
| `AIOS_WORKFLOW_MAX_AGENT_CALLS` | Per-run lifetime ceiling on `agent()` children (default 1000). |
| `AIOS_WORKFLOW_MAX_INFLIGHT_CHILDREN_PER_RUN` | Per-run concurrency cap (default 8). |
| `AIOS_WORKFLOW_AGENT_DEADLINE_SECONDS` | Wall-clock budget per `agent()` call (default 1h). |
| `AIOS_WORKFLOW_RUNS_PER_LAUNCHER_MAX` / `_PER_ACCOUNT_MAX` | Outstanding-run caps (20 / 100). |
| `AIOS_WORKFLOW_WAKE_BATCH_SECONDS` | Coalescing window for run wakes (0 = immediate). |

</details>

---

## The unified invocation kernel

> A call is a durable edge, not a function call.

**What it buys you:** any caller invokes any servicer through one durable request edge — so cancellation, depth-limiting, and a complete causal trace of who-called-whom come for free, and the calling agent stays responsive to its human while a sub-agent works.

Any caller — a model inside a session, an external HTTP/operator client, or a workflow run — delivers a request to a servicer (an existing session, a freshly-spawned agent, or a durable run) and awaits its single answer through **one** resolver, **one** awaiter, **one** completion envelope, and **one** private `stimulate` spine.

- **One `stimulate` spine** over a 4-arm frozen-dataclass union — `AskNewSession` / `TellNewSession` / `AskExistingSession` / `TellExistingSession` — so illegal combinations (e.g. `output_schema` on a Tell) are unrepresentable. `Ask ⇒ awaited`; `Tell ⇒ fire-and-forget`.
- **Trusted request edge** — every invocation materializes a lifecycle event carrying `caller={kind,id}`, depth, the frozen capability surface, vault_ids, `awaited`, `output_schema`, and a summary. All enforcement reads come off this trusted frame, never a forgeable blob; the model cannot inject a `caller` (every arg model is `extra=forbid`).
- **Caller invocations park as implicit-async tasks** — model-callable `call_session` / `call_agent` / `call_workflow`, the HTTP `POST /v1/tasks`, and the workflow run-caller all converge on the same edge; the caller stays responsive to its human while the servicer works.
- **Functional recursive cancel** — cancel seeds only the root; each marked **session** node cancels itself under its own single-writer lock, then re-seeds markers on its awaited children (a run servicer finalizes as a single node — no down-cascade, by construction). No global supervisor, no lock-the-world; first-writer-wins makes a late `return` a harmless no-op.
- **Down-counting depth budget** — every trusted edge carries `parent.depth - 1`; the spawn edge refuses *before* writing any child row when depth hits 0. The decrement IS the cycle bound.
- **Background-priority demotion** — a session is demoted to background priority (`-10`) when its latest open request edge is background-rooted, so a workflow's fan-out can't starve a human's interactive message.
- **`trace`** — a zero-instrumentation read-projection: one `REPEATABLE READ` snapshot, flat DFS pre-order, each node normalized to `ok/errored/cancelled/suspended/running`, journals interleaved, typed truncation at `AIOS_TRACE_MAX_NODES` (2000).

> **Self-goals (roadmap):** a self-goal is a reflexive deliver-kernel request where caller == servicer == self. The enabling keystone (the consecutive-inaction nudge) and the `[self]` origin label are built; the thin `set_goal` tool that writes the reflexive edge is **not yet shipped**.

### Invocation tools & endpoints

| Tool | Description |
|---|---|
| `call_session` | Call an existing same-account session; park for `{ok\|error}`. (renamed from `invoke`) |
| `call_agent` | Spawn a fresh session from one of your agents and call it. (renamed from `invoke_agent`) |
| `call_workflow` | Launch a run as an awaited single-shot servicer. (renamed from `invoke_workflow`/`create_run`+`await_run`) |
| `stop_task` | Durably cancel one of your awaited `call_*` tasks (and its subtree) by `tool_call_id`. |
| `wake_session` | Wake another same-account session (depth cap 10, per-pair rate cap 10/hr). |
| `wake_self` | Append a user-role message to your own session (model tool AND sandbox `tool wake_self`). |
| `return` / `error` | Answer an open awaited obligation exactly-once. |

<details>
<summary><b>Invocation endpoints & CLI</b></summary>

| Method | Path | Description |
|---|---|---|
| POST | `/v1/tasks` | Kind-agnostic request-writer; returns `TaskHandle{servicer_kind, servicer_id, request_id}`. |
| GET | `/v1/tasks/{task_id}/await` | The one awaiter over both servicer kinds (≤60s long-poll, MCP-usable). |
| POST | `/v1/tasks/{task_id}/cancel` | Seed a recursive cancel (202 Accepted, idempotent). |

```
aios trace <id>                  # DFS pre-order tree of a run (wfr_…) or session (sess_…)
aios tasks create --target-kind --target --input
```

</details>

---

## Triggers & scheduling

**What it buys you:** an agent that schedules its own work — cron jobs, deadlines, webhooks, and run-to-run DAGs — from inside its own session, with authority re-clamped on every fire.

A trigger is a per-session row pairing a **source** (what fires) with an **action** (what runs) — two fully orthogonal discriminated unions, a **4×4 product space**, not four bolted-on features.

| | `sandbox_command` (bash, no model wake) | `wake_owner` (deliver to this session) | `wake_session` (deliver to another) | `workflow` (launch a run) |
|---|---|---|---|---|
| **`cron`** | scheduled poller | morning briefing | orchestrator nudge | scheduled pipeline |
| **`one_shot`** | deferred task | deadline reminder | timed hand-off | one-time launch |
| **`run_completion`** | post-run script | notify on completion | wake worker on done | run-to-run DAG |
| **`external_event`** | webhook → bash | webhook → wake | webhook → peer | webhook → workflow |

- **Reactive `run_completion`** fires are matched and INSERTed as carrier rows **inside the watched run's own completion transaction** — "the run completed" and "these fires are owed" commit as one atomic fact. Exactly-once via the journal, not a polling reconciler.
- **`external_event`** is the **one account-key-free route** in the system: a per-trigger ingest secret (`aios_evt_<32-byte>`, stored only as a SHA-256 hash, surfaced plaintext once) is the tenant proof at `POST /v1/triggers/ingest/{ingest_token}`. Cheapest-first checks (64 KiB cap → JSON-object → token lookup → INSERT); unknown/disabled/revoked/archived all collapse to a uniform 404 (no oracle).
- **Owner-authority re-clamped at every fire** — a `workflow` action re-clamps the run's surface to the owner's *current* agent, re-checks vaults, asserts the version pin, and counts against the run cap. A trigger written months ago can never escalate.
- **Pin-and-freeze** — `workflow_version` as a fire-time *drift assertion*: a mismatch refuses to run an edited-but-unreviewed script (a tripwire, not a time machine).
- **Event-driven scheduler** — a single async task sleeps until the next due fire or a `LISTEN` notify wakes it; not fixed-interval polling. Adding a trigger to a live session honors it within the NOTIFY round-trip.
- **`trigger_runs` audit + delivery semantics** — at-most-once for one-shots (DELETE before action), exactly-once for run-completion (carrier row), recoverable for both; stuck-running rows are counted+warned but never retried.
- **Auto-disable circuit breaker** — a standing trigger auto-disables after 5 consecutive failures, surfacing a user-visible message; a re-enable resets the counter and self-heals a `NULL next_fire`.

### Trigger tools, endpoints & CLI

| Tool | Description |
|---|---|
| `trigger_create` / `trigger_list` / `trigger_update` / `trigger_remove` | Manage triggers; `trigger_list` surfaces `last_fire_status` + `consecutive_failures`. |
| `schedule_wake` | Sugar: a one-shot `wake_owner` from `delay_seconds` or absolute/natural-language `at` (via dateparser). |

<details>
<summary><b>Trigger endpoints & CLI</b></summary>

| Method | Path | Description |
|---|---|---|
| GET/POST | `/v1/sessions/{id}/triggers` | List / add (external_event returns the once-only `ingest_token`). |
| PATCH/DELETE | `/v1/sessions/{id}/triggers/{name}` | Update (source/action replaced wholesale) / remove. |
| GET | `/v1/sessions/{id}/triggers/{name}/runs` | Per-fire audit, newest first (survives trigger deletion). |
| POST | `/v1/triggers/ingest/{ingest_token}` | The account-key-free webhook ingress. |

```
aios sessions triggers list | add | update | remove | runs <session_id> ...
```

</details>

---

## Connectors & multi-channel

> Your assistant lives on Signal, Telegram, Slack, and WhatsApp at the same time — one continuous mind, not a bot-per-app.

**What it buys you:** a single agent identity omnipresent across every consumer messaging app at once, holding one channel in focus at a time — with connectors that can crash or be compromised without reaching the master key, another tenant's secrets, or the worker's database pool.

A connection binds one platform account to a routing target — a single long-lived session, or a `session_template` that spawns a fresh session per unseen chat partner — and one session can be bound across many channels at once.

- **Connectors are out-of-process HTTP clients**, not in-tree plugins. Each is a standalone container that talks to aios purely over the management API (POST inbound, tail SSE for outbound calls, POST results). It never shares the worker's process, Postgres pool, or `CryptoBox` — so a crashed or compromised connector can't reach the master key, the database, or another tenant's secrets (only the platform credentials for the connection it serves).
- **Multi-channel focal attention** — a session holds exactly one focal channel at a time (or none — "phone down"); non-focal channels render as truncated unread markers. `switch_channel` is the only way focal attention changes after spawn, returning a re-orient recap.
- **Bare assistant text is internal monologue** — channels are reachable *only* via outbound tools, so "what the user saw" is exactly the set of outbound tool calls — a clean audit boundary.
- **Three routing modes from one bindings table** — `detached` / `single_session` / `per_chat`, with a three-tier resolver (chat-sessions ledger → routing-rule prefix demux → bindings.mode fallback). At-most-one-active-binding is a schema invariant.
- **Idempotent both ways** — inbound dedups on a client `event_id` inside the append transaction; outbound persists each result between the side-effecting send and the result POST (an answered-spool), so a send-succeeded/POST-failed window re-POSTs rather than re-sends.
- **Delivery results are stimuli** — send acknowledgements and failures both append to the log and wake the session, just like every other tool result. Delivery tools differ only in typing body exceptions as `delivery_failed`.
- **Encrypted per-connection secrets** — platform credentials are encrypted under a per-account subkey, write-only on the operator surface (`secrets_set: bool`), decryptable only by a connector holding a runtime token for that type.

### Connector roster

| Connector | Notes |
|---|---|
| **Signal** | Wraps `signal-cli` in single-account daemon mode; operator register/verify/profile. |
| **Telegram** | One bot per token, PTB long-polling; rich inbound (photos, stickers, edits, reactions). |
| **Slack** | Socket-Mode (no public ingress), markdown→mrkdwn, ack-first 3s window, self/loop/cross-app/mention gates. |
| **WhatsApp** | Python connector spawns a Go `whatsmeow` daemon over loopback JSON-RPC; rotating-QR pairing. |
| **echo-http** | The SDK reference example + e2e fixture (ping / echo / trigger_inbound). |

| Outbound tools (per platform) |
|---|
| `signal_send`, `signal_react`, `signal_delete`, `signal_create_group`, `signal_rename_group` |
| `telegram_send`, `telegram_typing`, `telegram_edit_message`, `telegram_delete_message`, `telegram_react` |
| `slack_send`, `slack_react`, `slack_edit_message`, `slack_delete_message` |
| `whatsapp_send`, `whatsapp_react`, `whatsapp_edit_message`, `whatsapp_delete_message`, `whatsapp_list_groups`, `whatsapp_create_group`, `whatsapp_rename_group` |

<details>
<summary><b>Connector endpoints & CLI</b></summary>

| Method | Path | Description |
|---|---|---|
| POST | `/v1/connections` | Create a connection (detached). |
| POST | `/v1/connections/{id}/attach` \| `/configure-per-chat` \| `/bind-chat` | Bind single-session / per-chat / pin a specific chat. |
| POST | `/v1/connections/{id}/reparent` | Atomically move a connection (id-preserving) to another account. |
| POST | `/v1/connectors/runtime/inbound` | Connector posts an inbound user message (multipart; idempotent on `event_id`). |
| GET | `/v1/connectors/runtime/calls` | SSE of pending custom tool calls for the connector type. |
| POST | `/v1/connectors/runtime/tool-results` | Submit an outbound tool result (always a stimulus — the session wakes to react). |
| GET | `/v1/connectors/runtime/secrets` | The only decryption path for a connection's secrets. |
| PUT | `/v1/connectors/{connector}/tools_schema` \| `/capabilities` | Connector publishes its derived tool catalog / typed capability descriptor (root-only). |
| POST | `/v1/connectors/signal/register\|verify\|profile` | Operator-facing Signal provisioning. |
| POST | `/v1/connectors/whatsapp/start-pairing\|pairing-code\|confirm-pairing\|unpair` | Operator-facing WhatsApp QR pairing. |

</details>

> The **typed capability descriptor** (`draft_streaming`, `native_buttons`) and its publication route + per-session read seam exist, but the intended consumer (an outbound delta renderer) is **not yet in-tree** — a deliberate seam awaiting its consumer.

<details>
<summary><b>Connector CLI</b></summary>

```
aios connections create | list | get | set-secrets | attach | detach
aios connections configure-per-chat | bind-chat | bound-chats | recent-chats | reparent | archive
aios signal register|verify|profile ...   /   aios whatsapp start-pairing|pairing-code|confirm-pairing|unpair ...
```

</details>

---

## Durable sandboxes & environments

> A sovereign per-session computer that remembers everything for months but holds none of your secrets.

**What it buys you:** the agent runs as root in its own container, installs whatever it wants, and that whole filesystem survives crashes, OOM, daemon restarts, and months of idle — yet it can never read, exfiltrate, or flush the credentials it authenticates with.

Every session gets its own Docker container, lazily provisioned on the first tool call (chat-only sessions never create one), bind-mounting a per-session host workspace at `/workspace`. Unlike task-scoped agent sandboxes, aios sandboxes are **durable**.

- **Full-filesystem persistence** — on teardown the whole writable rootfs is committed (`stop → docker commit → rm`) to a per-session image; the next wake resumes from it. Installed packages, dotfiles, and scratch state survive crashes, OOM, daemon restart, and months of idle. Containers run *without* `--rm` so an unplanned death leaves a salvageable corpse.
- **Commit-time secret scrub + flatten** — `docker commit` empties exactly the run-injected env keys; crossing a chain-depth or per-session byte budget flattens via `export | import` (the definitive scrub, also reclaiming deleted-file space). Budget-driven flatten is commit-and-flag, never refuse — the agent's work is never destroyed as punishment.
- **Snapshot GC reconciler** — hourly: salvage crash corpses, retain-vs-evict by a single rule, evict most-dormant-first over `sandbox_snapshot_pool_bytes`, enforce per-account caps, reconcile DB pointers. Eviction appends model-visible (non-waking) `sandbox_fs_reset/_expired/_over_limit` notices.
- **Security applied from *outside* the tenant-writable filesystem:**
  - **Network lockdown** (`iptables -P OUTPUT DROP` + allowlist) is applied AND read-back-verified from an ephemeral **operator-image sidecar** in the sandbox's netns. The sandbox holds **zero `NET_ADMIN`**, so a persisted poisoned `iptables`/`getent` binary can't subvert the firewall or flush its own lockdown. Fail-closed: an unverified Limited sandbox is torn down, never handed back open.
  - **IPv4-only egress** with a per-session `ip6tables` DROP (not relying on `--ipv6=false` alone) and a read-back verify that asserts the DROP actually landed — closing the "green verify while open" class.
- **Placeholder-only credentials** — vaulted env-var secrets surface only as opaque placeholders; a per-session TLS-terminating egress proxy swaps the real secret in headers/body (never the URL) as traffic leaves the box. The agent runs as root inside, trusts an aios CA, and still can't read, exfiltrate, or flush — **the credential never enters the container, log, or spec.**
- **Hardening** — `docker --init` (tini) zombie reaping, `--cpus`/`--memory`/`--pids-limit` caps, an always-emitted authored seccomp deny-list, `no-new-privileges`, `--ipc private`, and an in-container `timeout -s KILL` wrapper.
- **Workspace jail** — `<workspace_root>/<account_id>/<session_id>` validated at create AND re-validated at the bind-mount boundary (`resolve()` defeats symlink-swap TOCTOU); out-of-jail paths 403. Attachments (`/mnt/attachments`) and uploads (`/mnt/uploads`) mount read-only; memory stores at `/mnt/memory/<name>`.
- **Backend & runtime** — `AIOS_SANDBOX_BACKEND` selects the backend via discriminated dispatch (only `docker` ships; unknown values fail hard). `AIOS_SANDBOX_RUNTIME=runsc` opts into gVisor for sandboxes and lockdown sidecars. (There is no separate "runc backend" — runc is Docker's default runtime; runsc is the gVisor opt-in.)
- **Environments** — a reusable template: packages across 6 managers (apt/pip/npm/cargo/gem/go, best-effort + logged), network policy (`unrestricted` | `limited` with `allowed_hosts`), per-env base image override, injected env vars, and per-env snapshot/timeout budgets. The reserved `aios-sbx-` image prefix is rejected so a tenant can't mount another session's snapshot.

> **Honest status:** a `SnapshotStore` Protocol abstracts snapshot transport with host-independent refs, but only `LocalDaemonStore` (identity over the local Docker daemon) ships today. Multi-host is the named, additive, deferred hinge.

<details>
<summary><b>Environment endpoint, CLI & sandbox config</b></summary>

| Method | Path |
|---|---|
| POST/GET/PUT/DELETE | `/v1/environments[/{env_id}]` |

```
aios envs list | get | create | update | archive
```

| Var | Purpose |
|---|---|
| `AIOS_SANDBOX_BACKEND` / `AIOS_SANDBOX_RUNTIME` | Backend (`docker`) / runtime (`runsc` for gVisor). |
| `AIOS_DOCKER_IMAGE` | Default sandbox + lockdown-sidecar image (`ghcr.io/eumemic/aios-sandbox:latest`). |
| `AIOS_EGRESS_CA_KEY` | **Required.** HKDF-derives the deterministic egress CA, separate from the vault key. |
| `AIOS_SANDBOX_{CPU_QUOTA,MEMORY_BYTES,PIDS_LIMIT,SECCOMP_PROFILE}` | Per-sandbox resource + syscall caps. |
| `AIOS_SANDBOX_SNAPSHOT_{BUDGET_BYTES,POOL_BYTES,TTL_SECONDS}` | Per-session byte budget / per-host pool / dormancy TTL (30 days). |
| `AIOS_CONTAINER_IDLE_TIMEOUT_SECONDS` | Inactivity before a sandbox is released (default 1800s; release snapshots first). |

</details>

---

## Tools, MCP & permissions

> One tool spine, three transports, structural permissions.

Every tool is registered once against a module-level `ToolRegistry`; the same pure core is driven by three callers — the model's tool surface, a sandbox-side `tool` CLI inside bash, and the workflow run frontier — through one transport+permission resolution chain the model cannot bypass.

- **Five-way authority disposition** — the permission ladder is walked once and returns `IMMEDIATE / MCP_IMMEDIATE / NEEDS_CONFIRM / CUSTOM / UNKNOWN_MCP`. The dispatch loop, the `awaiting` view, and the crash-recovery sweep are all projections of this one result, so a route-aware refinement cannot be present in two paths and absent in the third.
- **`transport` as a security frontier** — `cli` / `agent_tool` / `both`. Outbound-side-effect tools default to `agent_tool` so the model stays the bottleneck for irreversible effects; overridable per-tool, per-MCP-server, or per-MCP-tool.
- **Credential isolation by construction** — `http_request` and MCP auth headers are authored by the worker from vault credentials and **never enter the sandbox**. Route allowlists carry glob path patterns, optional method scoping, default-deny query strings, and dot-segment rejection so the gate's check equals httpx's wire effect.
- **MCP toolsets** — an agent declares `mcp_servers` (streamable-HTTP URL + static headers, optional `include_instructions`) and per-tool configs. Tools are auto-discovered, namespaced `mcp__<server>__<tool>`, schema-sanitized, vault-authed (bearer/basic/custom/oauth2-refresh with transparent refresh), and pooled — the pool key hashes only static headers so OAuth rotation doesn't churn connections; a circuit breaker stops one hung server starving the turn prelude.
- **`always_ask` human-in-the-loop** — a gated call is held unresolved; the client POSTs allow (next step dispatches) or deny (model-visible error) — down to a single HTTP route or MCP tool. Connector-mounted MCP tools with no explicit policy fall back to `AIOS_DEFAULT_MCP_PERMISSION_POLICY` (unset by default; unmounted MCP toolsets then gate on `always_ask`).
- **Custom (client-executed) tools** — `type=custom`; the harness never runs them. The call is held and surfaced on `awaiting`; the operator or connector runtime POSTs the result.

### Tool inventory

| Tool | Description |
|---|---|
| `bash` | Run bash in the session's durable sandbox (`agent_tool`, executes in sandbox). |
| `read` | Line-numbered windows or inline an image as multimodal parts (vision-gated). |
| `write` / `edit` | Base64-stdin write / strict find-and-replace with diff; memory-mount-aware (durable versioned). |
| `glob` / `grep` | ripgrep-backed, gitignore-respecting. |
| `web_fetch` / `web_search` | Tavily `/extract` & `/search` with SSRF guard (`transport=both`). |
| `search_events` | Read-only SELECT against your own session's `events_search` view. |
| `http_request` | Authenticated call to a declared `http_server`; secret never in sandbox; route/method/query allowlisted. |
| `call_session` / `call_agent` / `call_workflow` | The invocation kernel (park for `{ok\|error}`). |
| `stop_task` / `list_tasks` / `wake_self` / `wake_session` / `switch_channel` / `list_related_sessions` | Self-state & coordination. |
| `schedule_wake` / `trigger_create` / `trigger_list` / `trigger_update` / `trigger_remove` | Self-scheduling. |
| `create_workflow` / `update_workflow` / `archive_workflow` / `cancel_run` / `resume_gate` / `get_run` / `list_runs` / … | Strange-loop workflow authoring (surface must be ⊆ the agent's own). |
| `skill_upsert` / `skill_archive` | Author/version-bump the agent's own skills (account/session ids loaded server-side, `extra=forbid`). |
| `return` / `error` | Workflow-child response edge (exactly-once, output-schema-enforced). |
| `mcp__<server>__<tool>` | Auto-discovered MCP tools. |
| `<custom>` | Client-executed tools held until a result is POSTed. |

<details>
<summary><b>Tool configuration</b></summary>

| Var | Purpose |
|---|---|
| `AIOS_DEFAULT_MCP_PERMISSION_POLICY` | Fallback for un-opted-in MCP tools (unset by default → gates on `always_ask`). |
| `AIOS_TAVILY_API_KEY` | Powers `web_fetch` / `web_search`. |
| `AIOS_HTTP_RESPONSE_MAX_CHARS` | Cap on `http_request` bodies (~1M; cut bodies flagged `truncated:true`). |

</details>

---

## RLM — context variables & recursive sub-queries (this fork)

The `cos-runtime` branch adds a **Recursive Language Model substrate**
([docs/rlm.md](docs/rlm.md)): the agent never holds its working state in the
prompt window — state lives as addressable **context variables** (session- or
agent-scoped named handles; content out of context, metadata cheap to list)
that it inspects programmatically and recursively queries via cheap sub-model
calls, so effective context is unbounded and nothing decays.

- **`ctx_list` / `ctx_peek` / `ctx_grep` / `ctx_write` / `ctx_eval`** — list
  metadata, read hard-capped slices, regex-search server-side, persist state
  (agent scope = a world model that outlives every session), and run Python
  over staged variables in the session sandbox. Reusable scripts persist as
  `kind='helper'` variables and re-run by name, so competence compounds.
- **Spill inversion** — an oversized tool result now spills *into* a
  session-scoped variable; the event stores a handle + deterministic preview
  instead of a truncation stub.
- **`rlm_query` / `rlm_map` / `rlm_verify`** — spawn attenuated generic child
  sessions (ordinary async tool calls riding the invocation kernel: clamped
  surfaces, frozen at spawn, read-only variable grants in the spawn
  transaction, crash-re-parkable parks) with **dispatch-path budgets**: depth
  down-counted on the request edge, children-per-step and child-tokens-per-turn
  on a per-session ledger. `rlm_verify` returns a
  `{supported, evidence[{var, quote}]}` verdict from a child that can only
  read the cited variables — verification before assertion.
- **Model tiers** — `AIOS_MODEL_TIERS='{"root":…,"sub":…,"verify":…}'` plus a
  `tier:` model-string scheme resolved late at the surface chokepoint; rlm
  tools accept tier names only, so sub-call unit economics live in one knob.
- **Operator plane** — `/v1/context-variables` + `aios vars`.

The first consumer is a chief-of-staff agent for solopreneurs, shipped as a
versioned template from a separate repo (agent JSON + world-model variable
seeds + skills + cron routines) against this substrate.

## Memory stores & skills

Two complementary long-term-knowledge resources that let one entity accrue knowledge across months and rewrite its own playbooks.

**Memory stores** are session-attachable, path-addressed text mounted at `/mnt/memory/<store>/`.

- **Immutable, gapless per-store version log** — every create/modify/delete appends a `memory_versions` row with operation, actor, sha256, size, and a per-store seq. Deletes are soft; history retains the `memory_id`.
- **Three write paths, one durability guarantee** — file tools (memory-intercepted), raw bash (diffed post-exec against a pre-command sha snapshot), and the HTTP API all converge on the same versioned rows + live shared mount.
- **Optimistic concurrency** — the `read` tool stamps a sha; a later `write` gates on it, so concurrent sessions of the same entity surface an actionable "file changed since your last read" error instead of a silent clobber.
- **Lazy materialization + live shared mount** — one host dir per store, bind-mounted into every attached session; a write in session A is instantly visible in session B. Re-materialization from DB survives ephemeral sandboxes.
- **Per-version redaction that preserves the audit trail** — scrub a historical version's bytes while keeping who/when; the live head can't be redacted.
- **Snapshot-at-attach** — a running session's mount name/path is frozen at attach; operator-side renames never disturb live sessions.

> **Note:** memory content is **plaintext at rest** (versioned + audited, *not* encrypted). Only vault credentials and the GitHub clone token use the vault.

**Skills** are agent-attachable, versioned knowledge bundles using **progressive disclosure**: only name+description sit in the system prompt (~100 tokens each); the full `SKILL.md` + scripts are read on demand from `/workspace/skills/`. A skill ref is `version=None` (auto-latest) or a pinned int.

- **`skill_upsert` / `skill_archive`** — the agent authors/version-bumps its own skills in-loop (`transport=agent_tool`; the sandbox CLI broker refuses them). Trusted ids are loaded server-side, never tool args — autonomy bounded by capability, not by trusting model input.

<details>
<summary><b>Memory & skill endpoints & CLI</b></summary>

| Method | Path |
|---|---|
| `/v1/memory-stores[/{store_id}]` + `/memories` + `/memory-versions[/{id}/redact]` | Full CRUD + version history + redaction. |
| `/v1/skills[/{skill_id}]` + `/versions[/{version}]` | Full CRUD + immutable version history. |

```
aios skills list | get | create --dir <SKILL.md dir> --title <t> | archive | versions | version
```

</details>

---

## Vaults & credential injection

> Your agent can call authenticated APIs without ever being able to read the credential.

Vaults are named, tenant-scoped credential collections. Every secret is encrypted at rest with libsodium SecretBox (XChaCha20-Poly1305) under a **per-account HKDF subkey** (a leaked subkey reads nothing across tenants), is write-only (never returned by any API), and is consumed two architecturally distinct ways:

1. **Header credentials** (`bearer_header` / `oauth2_refresh` / `basic` / `custom_header`) are decrypted in the worker and rendered into outbound auth headers for MCP/HTTP calls, with automatic OAuth refresh.
2. **`environment_variable` credentials** never enter the sandbox: only a deterministic opaque placeholder is materialized, and the per-session TLS-MITM egress proxy substring-swaps it for the real value in headers+body (never the URL) as traffic leaves the box.

- **Separately-keyed deterministic egress CA** — `AIOS_EGRESS_CA_KEY` HKDF-derives the CA keypair (zero stored state; every worker derives the same key). It is **distinct from `AIOS_VAULT_KEY`**: vault-key holders can decrypt at-rest rows but cannot mint sandbox-trusted certs.
- **Fail-closed SNI gate** — host scoping is enforced solely at leaf-mint time; the proxy re-resolves the SNI host, blocks SSRF/internal ranges, and pins the upstream IP (defeating DNS rebinding). No leaf for an off-allowlist or absent SNI host.
- **Path-prefix-scoped egress** — `host/<path-prefix>` (e.g. `github.com/repos/eumemic`) via one grammar shared by create-time validation and the runtime swap matcher, with segment-boundary and percent-encoded/backslash dot-segment defenses.
- **Basic-auth-aware swap** — decodes `user:pass` (UTF-8), swaps, re-encodes, since base64 hides the literal placeholder.
- **Rotation-stable, zero-row placeholders** — a pure HKDF function of (account salt, owner, credential); `aios rekey` re-encrypts the salt but doesn't change the value, so a placeholder an agent persisted into `/workspace` keeps resolving across master-key rotation.
- **Interactive OAuth Connect** — `oauth/start` does RFC 9728/8414 discovery → RFC 7591 Dynamic Client Registration → PKCE + CSRF → `authorization_url`; `oauth/complete` stores an `oauth2_refresh` credential. Operator-registered apps (`AIOS_OAUTH_PROVIDER_APPS`) let DCR-less providers (Google/Microsoft/Slack) connect without users supplying client secrets. This is the console's one-click "Connect" backend.
- **Prompt revocation** — archiving zeroes ciphertext; a `pg_notify` on the MCP eviction channel evicts pooled sessions immediately.

> **Documented residuals:** responses are not scrubbed; HTTPS-only; request-signing APIs (SigV4/HMAC/OAuth1) unsupported; Unrestricted-with-credentials is permit-with-warning (the SNI gate still confines the value, but there's no exfil containment without an allowlist).

<details>
<summary><b>Vault endpoints, CLI & config</b></summary>

| Method | Path |
|---|---|
| `/v1/vaults[/{vault_id}]` + `/archive` | Vault CRUD + scrub-on-archive. |
| `/v1/vaults/{id}/credentials[/{id}]` + `/archive` | Credential CRUD (secrets never returned). |
| POST | `/v1/vaults/{id}/credentials/oauth/start` \| `/complete` | Interactive OAuth Connect. |

```
aios vaults list | get | create | update | archive | delete
aios vaults credentials create | list | get | update | archive | delete
aios rekey
```

| Var | Purpose |
|---|---|
| `AIOS_VAULT_KEY` | **Required.** Master libsodium key; per-account subkeys HKDF-derived from it. |
| `AIOS_VAULT_KEY_PREVIOUS` | Decrypt-only previous key used only during `aios rekey`. |
| `AIOS_EGRESS_CA_KEY` | **Required.** Separate key for the sandbox egress CA. |
| `AIOS_OAUTH_PROVIDER_APPS` | Operator OAuth client apps for DCR-less providers. |

</details>

---

## Security model

aios's strongest differentiator is that security is **structural** — illegal states are unrepresentable, credentials never reach the model, and authority can only narrow. The pieces, consolidated:

- **Cryptographic tenant isolation** — bearer tokens hash to a row (revocable, never an env compare); every account's secrets are encrypted under a per-account HKDF subkey, so a leaked subkey reads nothing across tenants.
- **Two credential paths the model can never read** — vaulted header credentials are authored worker-side into outbound headers; env-var secrets surface only as opaque placeholders swapped by a per-session TLS-MITM egress proxy. The git PAT for repo mounts is held by an in-worker **GitProxy** the same way. In all three, the secret never enters the container, log, or spec.
- **Separately-keyed egress CA** — `AIOS_EGRESS_CA_KEY` is distinct from `AIOS_VAULT_KEY` by design, so vault-key holders cannot mint sandbox-trusted certs. The fail-closed SNI gate re-resolves and pins the upstream IP (anti-SSRF, anti-DNS-rebinding).
- **Capability-attenuation lattice + api_base freeze** — a child's authority is the lattice meet of its declared surface with its launcher's, frozen at the spawn edge and non-widening by construction; the inference endpoint is itself a separately-frozen, fail-closed identity check (equality-or-allowlist, not a lattice meet).
- **Fail-closed sidecars** — network lockdown is applied and read-back-verified from an ephemeral operator-image sidecar in the sandbox's netns; the sandbox holds zero `NET_ADMIN`, so a poisoned persisted binary can't subvert its own firewall.
- **Human-in-the-loop gating** — `always_ask` down to a single HTTP route or MCP tool.

---

## Accounts, multi-tenancy & spend

> A child agent can never exceed its parent — by construction, not by check.

**What it buys you:** real cryptographic multi-tenancy with a single spend ceiling at the top that bounds an entire fleet of self-spawning descendant agents.

aios is genuinely multi-tenant on a hierarchical account tree. Every resource is account-scoped, and authentication resolves a `Bearer` token by SHA-256-**hashing it to a row** in `account_keys` — never an env-var string compare. The resolved tuple binds onto structlog contextvars, so every request emits a tenant-attributed structured log line.

The same tree carries three orthogonal control planes:

1. **Capability-attenuation lattice** — `attenuate(declared, launcher)` is a pure lattice *meet* over `Surface = (tools, mcp_servers, http_servers)`: `always_ask` beats `always_allow`, transport GLB over `{cli, agent_tool, both}`, HTTP method sets intersect. Computed once at the spawn edge and stored immutably (read back on every step), so a later `update_agent` can't widen an in-flight months-long session. By associativity, a single meet against the launcher equals the whole-chain fold. The same operator doubles as the author-edge admission predicate (a declaration is admitted iff it's a fixpoint of the meet).
2. **Model-identity (`api_base`) freeze** — the *second* authority axis: a child whose `api_base` points at a hostile endpoint would ship its entire prompt context to the attacker on the first call, no tool involved. The spawn edge fails closed with a journaled `untrusted_api_base` rejection unless the endpoint equals the launcher's or is in `AIOS_TRUSTED_INFERENCE_API_BASES` (empty default = trust nothing redirected).
3. **Subtree spend rollup** — two budgets flow opposite directions on the same tree: spend *limits* inherit **down** the parent chain; dollar *spend* rolls **up** from descendants. Each step admits **pre-flight** against the summed subtree spend, so one ceiling at the top bounds a whole fleet of self-spawning descendants even when no single account crossed alone. Usage is charged only after the assistant message is durably persisted (fail-safe against double-billing).

- **Delegated minting** — `can_mint_children` flows strictly down and cannot be self-escalated via PATCH.
- **No existence leak** — out-of-scope account targets raise the *same* 404 as nonexistent ones; auth failures share one 401 regardless of cause.
- **Gated one-shot bootstrap** — `POST /v1/accounts/bootstrap` mints the root + first key (gated by `AIOS_BOOTSTRAP_TOKEN`); the root-exists check fires before the token check, and the endpoint 404s permanently once used.
- **Archive → purge** — soft-archive then a compliance hard-delete (`/purge`), with `ON DELETE RESTRICT` on core resource FKs.

<details>
<summary><b>Account endpoints, CLI & config</b></summary>

| Method | Path |
|---|---|
| POST | `/v1/accounts/bootstrap` |
| GET/POST | `/v1/accounts/me` \| `/children` \| `/by-path` |
| GET/PATCH/DELETE | `/v1/accounts/{id}` + `/purge` + `/usage` |
| POST/GET/DELETE | `/v1/accounts/{id}/keys[/{key_id}]` |
| POST/GET | `/v1/runtime-tokens[/{id}/revoke]` |

```
aios accounts bootstrap | me | list | get | mint | update | archive | purge | by-path
aios accounts keys list | mint | revoke
```

| Var | Purpose |
|---|---|
| `AIOS_API_KEY` | Bearer key; must hash to an unrevoked `account_keys` row (placeholder values silently 401). |
| `AIOS_BOOTSTRAP_TOKEN` | Gates root bootstrap on a fresh DB. |
| `AIOS_DEFAULT_SPEND_LIMIT_USD` | Root/server fallback limit (unset = ungated). |
| `AIOS_TRUSTED_INFERENCE_API_BASES` | Allowlist for the `api_base` clamp (empty = fail closed). |

</details>

---

## API, CLI & SDK — one surface, three faces

The entire runtime is **one** versioned FastAPI app that is:

- **auto-reflected into an MCP server** at `/mcp` under the same bearer auth — so an agent operates accounts/sessions/vaults/workflows over the exact REST surface an operator scripts. There is no hand-maintained second API for agents.
- **introspected into a committed `openapi.json`**, and
- **code-generated into a typed Python SDK** (`packages/aios-sdk`),

all from one source of truth, with **CI drift-guards** (including a coverage test that fails if any OpenAPI operation lacks a covering CLI command unless explicitly allowlisted). An `x-codegen.targets` contract per route controls which faces (`sdk`/`mcp`/`cli`) each operation appears on; an MCP polish pass derives `readOnlyHint`/`destructiveHint`/`idempotentHint` from the HTTP verb and roughly halves the schema size the model sees.

**Three ways to observe a session over one `LISTEN/NOTIFY` spine:**

| Mode | Endpoint | For |
|---|---|---|
| SSE stream | `GET /v1/sessions/{id}/stream` | Live event + per-token delta tail (preflight-or-503, weakref-finalize leak protection, heartbeats). |
| Long-poll | `GET /v1/sessions/{id}/wait` | Stacks that can't consume SSE (notably Node `fetch`). |
| Quiescence join | `GET /v1/sessions/{id}/await` | MCP-usable drive-and-join: block until the session fully reacts to a watermark. |

The SDK re-exports a generated `AuthenticatedClient` plus hand-written SSE consumers for the streaming endpoints the generator can't model (read timeout sized as 3× the server's 15s heartbeat). A separate `packages/aios-connector-http` SDK (`HttpConnector` + `@tool`) builds connectors against the same API. Errors render a structured `{error:{type,message,detail}}` envelope; `/health` is DB-free liveness, `/ready` does `SELECT 1` under a 2s budget.

<details>
<summary><b>CLI cheatsheet</b></summary>

```
aios chat --agent <id> --environment-id <id>      # interactive REPL (-m for one-shot)
aios sessions stream|tail <id> --after-seq N       # live tail
aios trace <run_or_session_id> [--chronological] [--verbose]
aios status                                        # reachability + auth check
aios dev bootstrap | status | teardown             # per-worktree isolated DB
aios api | worker | migrate | rekey                # operator entrypoints
# global flags: --url --api-key --format {table,json} --verbose
```

</details>

> **Note:** there is **no audit-log table or endpoint** — "tenant-attributed logging" means one structured `api.request` log line per request with `account_id` bound to the logging context, not a queryable audit trail.

---

## Multimodal, files & attachments

- **Vision pipeline** — inbound images get magic-byte mime correction, downsample-to-fit (`INLINE_SIZE_CAP_BYTES=3.75 MiB`, `INLINE_MAX_DIMENSION=2000`, a Pillow JPEG/PNG ladder), and inline-vs-path rendering, gated on the bound model supporting vision. The `read` tool inlines images the same way.
- **Attachment staging** — connector inbound files are streamed (no shared FS) into `/mnt/attachments/<connector>/<event-ulid>-<filename>`, bind-mounted read-only, replay-safe on `event_id`, with stranded files reclaimed by an attachment GC at worker startup. A forged-attachment exfil vector is blocked by stripping reserved metadata keys.
- **Tool-result spill** — a result larger than `AIOS_TOOL_RESULT_MAX_CHARS` spills to the attachments mount with an inline stub so the model can `read()` it.
- **File upload** — `POST /v1/sessions/{id}/files` stages a file the model sees at `/mnt/uploads/<id>/<name>`.
- **GitHub repos** — a first-class per-session resource mounts a git repo at a user-specified `mount_path` with a **write-only clone token encrypted at rest** (same libsodium `CryptoBox` as vault credentials), optional `git_user_name`/`email` stamped post-clone, `MAX_REPOS_PER_SESSION=8`, and traversal/reserved-mount guards. A per-session **GitProxy** holds the PAT worker-side and injects `Authorization` into outbound smart-HTTP git traffic, so **the token never enters the container** — the second of two credential-never-enters-the-box paths, alongside the vault egress proxy. Manage via `GET/POST/PUT/DELETE /v1/sessions/{id}/resources[/{resource_id}]`; token rotation recycles the sandbox and re-clones.

### Operational robustness

Any worker exit (native crash, unretrieved task exception, ordinary exit) leaves an auditable log line via a `faulthandler` + asyncio-loop-exception-handler + `atexit` net. Alongside it: the hourly `trigger_runs` prune, attachment GC, host-dir reaper (idle session-repo/run scratch), and an opt-in archived-workspace reaper form a self-maintaining worker that keeps a months-running host from filling.

---

## vs. Anthropic Managed Agents

aios shares the **core architecture** from Anthropic's Managed Agents work — session = append-only log, harness = stateless step, sandbox = cattle-not-pets containers, and the placeholder + TLS-MITM credential-injection model — but is the **open, self-hostable, multi-tenant clean-room version built for long-lived entities** rather than task-scoped sessions. Managed Agents is a hosted, closed, operator-owned product; aios is MIT-licensed code you run where nothing load-bearing is hostage to a third party.

The AMA-specific deltas:

- Sessions are **mutable and outlive the agent** that drives them — rebind model/config without losing history (Managed Agents sessions are immutable after creation).
- The model is **any LiteLLM model string**, not a fixed provider.
- One session is **omnipresent across Signal/Telegram/Slack/WhatsApp** via a focal-channel primitive Managed Agents has no analog for.
- **Agent-scheduled triggers** (vs operator-only cron deployments), and per-route / per-MCP-tool permission gating (vs coarser per-agent tool toggles).
- It's **open and self-hostable**: durable workflows, the invocation kernel, the multi-tenant account tree, the separately-keyed egress CA, and per-session leaf minting are all yours to run and audit.

Everything else aios adds over AMA — no-loop async tools, no compaction, durable workflows, the invocation kernel, multi-tenancy — is detailed in the sections above.

### Not yet built (honest status)

- **Self-goals** — the reflexive `set_goal` tool. The enabling keystone (consecutive-inaction nudge) and the `[self]` origin label are built; the user-facing tool is not.
- **Multi-host snapshots** — the `SnapshotStore` seam exists; only `LocalDaemonStore` ships.
- **The typed connector capability descriptor's consumer** — the outbound delta renderer is a deliberate, unbuilt seam.
- **Workflow version-history read-through** — version pins can refuse a drifted script but cannot resolve an old one.

---

## Roadmap — where this is heading

The most telling thing about the backlog is what's *absent* from it: new core primitives. Nearly every planned capability is a **composition of mechanisms that already exist** — the event log, the invocation edge, the trigger union, the attenuation lattice — so the core stays small and new behavior falls out of it rather than accreting onto it. Direction, not dated commitments:

- **The invocation kernel, completed** — a model-facing open-invocation surface (`list_invocations`, model-driven cancel, `await_all` / `await_any`) and durable await-resume that survives a worker restart, making multi-agent orchestration a fully first-class, crash-durable primitive.
- **Workflows v2 — the embodied run** — runs and their children sharing one workspace, multi-worker execution, replay-versioning that lets in-flight runs survive a deploy, and a distributable workflow library with input/output schemas.
- **Memory intelligence** — making the versioned-FS memory substrate *intelligent*: automatic profile injection into context, full-text/trigram recall over memory and the event log, and cron-driven distillation of recent memory into new immutable cards.
- **Behavioral eval + LLM-judge** — asserting over the event log (structural checks plus judged quality) as a published library that consumes already-shipped surfaces and adds nothing to core.
- **Typed authoring** — an illegal-states-unrepresentable SDK, a project scaffold, an edit-run-observe loop, and a published TypeScript client twin of the Python SDK.
- **Pluggable sandbox backends** — the `SandboxBackend` Protocol gaining a microVM implementation and template pre-warming, plus the multi-host snapshot transport the `SnapshotStore` seam already anticipates.
- **More channels, hosted onboarding** — an SMS connector, a target-agnostic OAuth engine behind a one-click "Connect," and a default-deny inbound admission gate for new chat partners.

The throughline: the primitives are largely in place; the work ahead is composing and sharpening them.

---

## License

MIT.