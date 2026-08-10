# RLM substrate — context variables + recursive sub-queries

Status: design of record for the `cos-runtime` branch. Part 1 of the
chief-of-staff runtime: the substrate that lets a long-lived agent hold an
unbounded world model *outside* its prompt window and query it recursively
with cheap sub-model calls. The CoS product layer (agent template, world-model
schemas, sync prompts) ships from a separate repo and consumes this substrate
through the ordinary agent/tool/API surface.

## Design stance

Everything here is a composition of existing primitives — no new orchestrator,
no parallel spawn path, no second storage convention:

| Need | Existing primitive reused |
| --- | --- |
| Spawn an attenuated child as an async tool call | `services/sessions.create_child_session` + `attenuation_service.clamp` + the `invoke_session.py` park (`_park_and_resolve`, `resumable=True`) |
| Out-of-context storage with caps + audit | the `memories` content-column pattern (`text` + TOAST + sha256/size CHECKs) |
| Oversized-result handling | `sandbox/tool_result_spill.py`'s append-boundary seam, inverted |
| Cheap-model routing | the `workflow:` model-string scheme pattern (`harness/model_binding.py`), generalized to `tier:` |
| Budget refusal | the run `budget_usd` pre-spawn refusal shape (`workflows/step.py`) |
| Provenance | the trusted `request_opened` edge (caller / depth / frozen_surface) + `model_request_end` spans + `aios trace` |
| Periodic sync | triggers (`cron` × `wake_owner` / `workflow`) — no new scheduler |
| Prompt affordance | the ephemeral tail-block pattern (`harness/obligations.py`) + a cache-stable system-prompt augmenter |

## A. Context variables

### Storage: Postgres `text` + TOAST, not large objects, not sandbox files

A variable is a named handle whose content lives in a new `context_variables`
table. Content is a `text` column with the `memories` discipline —
`content_sha256`, `content_size_bytes`, `CHECK (content_size_bytes =
octet_length(content))`, and a byte cap (default 2 MiB,
`AIOS_CONTEXT_VARIABLE_MAX_BYTES`). Rationale:

- **TOAST already gives out-of-row storage transparently.** Postgres large
  objects (`lo_*`) are used nowhere in this codebase and add a second I/O API,
  a separate GC story (`vacuumlo`), and no SQL access to content. The
  in-house precedent for capped model-adjacent content is `memories`
  (migration 0025); we copy it.
- **`ctx_grep` wants server-side scans.** Content in a column supports `~` /
  `ILIKE` (and FTS later) under the existing 30s statement timeout. A file or
  large object would force content through the worker for every grep.
- **Sandbox files fail the scope test.** The attachments mount is read-only
  in-container and boot-GC'd by the referenced-set sweep; the memory-store
  host dir is *deliberately* shared across sessions. A session-private,
  agent-durable variable plane fits neither. Files are still used — but only
  as a *staging projection* for `ctx_eval` and child sandboxes (below), never
  as the source of truth.

### Scopes

`scope` is a discriminated column: `'session'` (keyed `(session_id, name)`)
or `'agent'` (keyed `(account_id, agent_id, name)`; `session_id IS NULL`).
Agent scope is what makes the world model outlive any session: a fresh
session of the same agent sees the same agent-scoped variables. Partial
unique indexes (`WHERE archived_at IS NULL`) enforce handle uniqueness per
scope, per the `triggers` naming pattern. Rows carry `kind`
(`data | helper | spill | digest`), `schema_tag`, `description`, `metadata`
jsonb (provenance: writer session/tool_call/trigger), `created_by_*` actor
columns, timestamps. Session-scoped rows `ON DELETE CASCADE` with the
session; agent-scoped rows live until archived.

Metadata is cheap to list (content never fetched — the `include_content`
pattern from `memory_stores` queries); content is never auto-inlined
anywhere.

### Tools

Builtin names are flat identifiers (`BuiltinToolType` forbids dots), so the
surface is `ctx_list`, `ctx_peek`, `ctx_grep`, `ctx_write`, `ctx_eval` —
worker-executing builtins registered in the ordinary registry, dispatched
through `_tool_lifecycle` like every other tool.

- `ctx_list(scope?, kind?)` — metadata only: name, scope, kind, size,
  schema_tag, updated_at. Bounded (200 rows).
- `ctx_peek(name, offset?, length?)` — a slice of content, hard-capped per
  call (default 16 KiB, `AIOS_CTX_PEEK_MAX_BYTES`); returns
  `{content, offset, length, total_size, truncated}`.
- `ctx_grep(name, pattern, max_matches?)` — regex over content server-side,
  returns matching line spans with byte offsets, capped (default 100
  matches).
- `ctx_write(name, content, scope?, kind?, description?, schema_tag?,
  append?)` — create/overwrite/append with the byte cap enforced; stamps
  provenance. Session scope by default; agent scope explicit.
- `ctx_eval(code? | helper?, vars?, timeout?)` — runs Python **in the
  session's own Docker sandbox** via the existing
  `SandboxRegistry.get_or_provision` + `exec` path (the `bash.py` shape).
  Referenced variables and the script are staged host-side under the
  session attachments dir (`_attachments/<sid>/ctx/…`, visible live at
  `/mnt/attachments/ctx/…` read-only — no new mounts, no container
  recycle); the script gets `CTX_VARS_DIR` pointing at the staged copies.
  stdout is the result (bounded by `bash_max_output_bytes`); non-zero exit
  is data, not an error (no sandbox eviction). `helper=<name>` runs a saved
  `kind=helper` variable instead of inline code — this is how the agent
  authors reusable procedures over its own state (e.g. a
  `stale_commitments` helper re-run daily) and how competence compounds.

Visibility rule for reads: a session sees its own session-scoped variables,
its agent's agent-scoped variables, and any variables **granted** to it at
spawn (below). Reads are account-scoped like everything else.

### Spill inversion

`cap_tool_result_content` keeps its seam (both sinks —
`tool_dispatch._append_tool_result_event` and
`services/sessions.append_tool_result` — already funnel through it) but the
destination inverts: an oversized `str` result is written to a
session-scoped variable `tool_result_<tool_call_id>` (`kind=spill`,
idempotent on the tool_call_id-derived name, tolerant of a write whose
referencing event later dedup-skips), and the event content becomes a stub
carrying the **handle plus a deterministic preview** (head of the content,
frozen at append time so `build_messages` replay stays pure) and the
recovery affordance (`ctx_peek` / `ctx_grep`). The stub is computed before
`precompute_event_append` so `cumulative_tokens` counts the stub — the
windowing invariant the file spill was built to protect. The
`metadata.attachments` record and the attachments-file write are gone; the
variable store is the one recovery convention.

This is strictly better than the file spill even for the blind-spot path:
the duplicated synthetic user message is now a short handle+preview instead
of a truncation notice pointing at a file.

### Prompt affordance

Two pieces, both existing patterns:

1. A cache-stable system-prompt augmenter (`augment_with_context_vars`,
   called from `compute_step_prelude` beside `augment_with_memory_stores`)
   describing the facility — present iff the surface holds any ctx tool.
2. An ephemeral tail block (the `obligations.py` trio: `build_*_tail_block`
   + `max_*_block_local` upper bound reserved in `prelude_overhead_local`,
   tagged `EPHEMERAL_TAIL_KEY`) listing up to 30 variables (name, scope,
   kind, size, age) most-recently-updated first, with a `+K more — ctx_list`
   overflow line. Per-step-mutating state never enters the system prompt.

The listing read happens in `compute_step_prelude` (async, DB access is
sanctioned there) and is stubbed in `tests/unit/conftest.py` like the other
pre-inference leaves.

## B. Recursive sub-queries

### rlm_query

`rlm_query(prompt, vars?, tier="sub", expect="text"|"json",
output_schema?)` is porcelain over the existing child-session machinery —
structurally an `AskNewSession`:

1. **Spawn**: deterministic child id derived from
   `(parent_session_id, tool_call_id)` (the `workflows/child_id.py`
   pattern), created via `create_child_session` — one transaction writing
   the row (`surface_frozen=TRUE`), the first user message (the prompt plus
   a manifest of granted variable handles), and the trusted
   `request_opened` edge (`caller={kind:"session", id:parent,
   tool_call_id}`, `depth`, `awaited=True`, `output_schema`).
   `archive_when_idle=True`: children self-reclaim after answering.
2. **Authority**: declared child surface = the ctx read tools
   (`ctx_list`, `ctx_peek`, `ctx_grep`) + `rlm_query`/`rlm_verify` (so
   recursion is *possible*; depth makes it *bounded*), empty
   `mcp_servers`, empty `http_servers`. Effective surface =
   `attenuation_service.clamp(declared, launcher_effective)` — the meet, so
   recursion can only ever narrow. No network tools, no connectors, no
   write/edit/bash means the child never touches a sandbox or the outside
   world. Model identity passes `model_identity_trusted` fail-closed before
   any row exists.
3. **Variables**: granted read-only via a `context_variable_grants`
   junction written in the spawn transaction (no content copies; the child
   cannot write parent state because `ctx_write` isn't in its surface, and
   grants are read-path-only by construction). The grant set is recorded on
   the request edge — the "vars read" provenance is exact.
4. **Await**: the handler parks with the shared `_park_and_resolve` on the
   durable edge, `resumable=True`, so a worker crash re-parks instead of
   erroring (#1431) and the parent stays responsive throughout — recursion
   is concurrent fanout for free because every sub-query is an ordinary
   fire-and-forget tool task. `expect="json"` rides the existing
   caller-side `output_schema` validation.

### rlm_map

`rlm_map(prompt, var, chunker, tier?)` fans out one child per chunk:
chunker `{kind: "lines"|"bytes"|"json_items", size}` splits the variable's
content; each chunk is written as a session-scoped variable on the child
(chunks are new content, so a copy is honest) and the children run
concurrently — one handler, N spawn edges, `asyncio.gather` over N parks
(never holding a pool connection across the park). Fan-out is admitted
against the ledger before any spawn.

### Budgets — enforced in the dispatch path

A small `rlm_ledgers` row per root session (root = the nearest ancestor
with no rlm request edge; the root id is carried on every edge so children
inherit it):

- `max_depth` (default 2): carried down-counting on the request edge
  exactly like the workflow depth budget; `rlm_query` refuses at `depth <=
  0` with a structured tool error (`rlm_depth_exhausted`).
- `max_children_per_step` (default 8): counted against the assistant step
  that issued the launching tool call; refusal is
  `rlm_children_exhausted`.
- `max_total_child_tokens` per root turn (default 500k): the harvest side
  adds each resolved child's session-row token counters to the ledger;
  admission refuses new spawns once the ledger crosses the cap
  (`rlm_tokens_exhausted`). In-flight children can overshoot by at most
  `max_children_per_step` children — accepted slack, same stance as the
  run budget's post-hoc accounting.

All three are `Settings` knobs (`AIOS_RLM_*`), never prompt text. Structured
errors mean the model can react (narrow the query, use a saved helper)
instead of silently degrading. Dollar spend needs nothing new: children
share the account and their usage rolls into the existing subtree spend
gate.

### Model tiers

`Settings.model_tiers: dict[str, str]` (`AIOS_MODEL_TIERS`, e.g.
`{"root": "...", "sub": "...", "verify": "..."}`) plus a `tier:` model-string
scheme mirroring `workflow:` (`is_tier_model` / `resolve_tier_model`,
fail-loud on unknown names; tier values may not themselves be `tier:` or
`workflow:` strings, so the binding-privilege guards cannot be bypassed by
indirection). Resolution happens late at the surface chokepoint
(`_load_for_session_conn`) so every consumer — capability gates, window
math, provider auth, dispatch, spans — sees a raw model with zero per-site
edits; the two seams that bypass it (`run_llm.py`, the workflow spawn edge)
resolve inline. rlm tools accept **tier names only** and stamp
`tier:<name>` into the child row; a raw model string is refused at the tool
boundary. Retargeting a tier in config retargets every future child step —
a solopreneur's unit economics live in one env var.

## C. Provenance + verification

Every `rlm_query`/`rlm_map`/`rlm_verify` result carries provenance:

- The durable skeleton is already there — the request edge records caller
  (with `tool_call_id`), depth, frozen surface, and grant set; the child's
  `model_request_end` spans record per-call model/usage/cost; the child's
  tool events record its `ctx_*` calls; `request_response` survives
  archive.
- The tool result content includes `{child_session_id, tier, vars_read,
  tokens, cost_usd}`; result `metadata` (event-persisted, wire-stripped)
  carries the fuller record including the child's ctx-call trace summary
  harvested from its event log.
- Surfacing: `aios sessions events --format json` shows the metadata;
  `aios trace <sess_…>` already renders session→session call trees via the
  request edges, so parent→child nesting comes free.

`rlm_verify(claim, vars, tier="verify")` is `rlm_query` with a stricter
child: surface = `ctx_peek` + `ctx_grep` only, grants = exactly the cited
variables, and a forced output schema
`{supported: bool, evidence: [{var, quote, offset?}], notes}`. The CoS
verifies before it acts or asserts — "you told Sarah March 12" needs a
receipt, and the receipt is an evidence span in a variable the verifier
could only read, not infer.

## What Part 2 consumes (separate repo)

The CoS product layer ships as data against this substrate: an
`AgentCreate` JSON (system prompt, ctx/rlm tools, Gmail/GCal MCP servers),
agent-scoped variable seeds (`people`, `projects`, `commitments`,
`pipeline`, `inbox_digest`, `calendar_digest`, `history`,
`session_digest` — schema tags versioned, evolution additive), skills for
procedures, and cron triggers (`wake_owner` for model-in-the-loop sync and
daily review; `workflow` actions for deterministic no-model syncs).
Operator surface: `/v1/context-variables` CRUD + `aios vars` for seeding
and inspection.

## Explicit non-goals (v1)

- No FTS index on variable content (regex/LIKE under the statement timeout
  is enough at solopreneur scale; the `memories_search` pattern is the
  upgrade path).
- No cross-agent or cross-account variable sharing.
- No variable version log (the `memory_versions` pattern is the upgrade
  path if world-model audit needs more than `history` + event provenance).
- No new surface dimensions: "read-only sandbox" is expressed by tool
  omission, "no network" by empty server lists + no network tools — the
  attenuation normal form stays untouched.
