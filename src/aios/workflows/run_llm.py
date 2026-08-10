"""Run-side raw inference — a workflow run's ``call_llm()`` capability (#1633).

Part of the **Workflows-as-Models** epic. Adds the worker-side resolver for the
author ``call_llm(request)`` primitive: it runs one inference turn against
``call_litellm`` (:mod:`aios.harness.completion`) and returns the **raw assistant
turn** (content + *unexecuted* tool_calls + usage + finish_reason + cost). The
script subprocess is credential-free (no provider keys), so — exactly like
``tool()`` — the script only emits the ``call_llm`` frontier and the inference
runs HERE, on the worker, against credentials resolved from the run's
account (per-account ``model_providers`` config, or the worker's env vars
when the account has none configured — see
``aios.services.model_providers.resolve_provider_auth``).

**Park-and-harvest.** Identical shape to :mod:`aios.workflows.run_tools`: the step
opens a ``call_llm`` frontier (journals ``call_started``), launches a
fire-and-forget worker task here, and parks the run. The task runs the inference
and, on completion, writes a ``tool_result`` ``wf_run_signals`` row + wakes the
run; the next step's pre-replay harvest folds the signal into a ``call_result``.

**The four runtime guards (this module owns them; they're per-call, not static):**

* **``workflow:`` rejection (leaf-only).** ``call_llm`` is the inference *leaf* — it
  must never recurse into a workflow binding. The model arg may be COMPUTED, so this
  can't be a static check; we reject a ``workflow:`` target here, at the call site.
* **model-identity clamp (#823).** A ``params.api_base`` redirects the inference to
  another endpoint, sending the whole prompt there. Admit it iff it equals the
  launcher's (a run has none → the operator default) OR sits in the operator
  ``trusted_inference_api_bases`` allowlist; else reject — same clamp the spawn edge
  applies to a child ``agent()``'s model identity. Orthogonal to the next guard: this
  one asks "may the prompt go there at all"; the next asks "whose key rides along."
* **provider-auth conflict.** ``auth.api_base`` (injected via the separate ``auth``
  parameter, never through ``params``/``litellm_extra``) is never evaluated by this
  clamp or its allowlist — trust for an account-resolved redirect is established by
  row ownership instead (see ``aios.services.model_providers``). Reject when
  ``params`` redirects the endpoint while the effective key would come from an
  account above the run's own (an ancestor's ``model_providers`` row, or the
  worker's env var on a non-root account).
* **cost meter (charge once at the inference site).** A successful call charges
  LiteLLM's per-request cost to the run's ``call_llm_cost_microusd`` meter, in the
  SAME transaction that writes the result signal, so the next step's ``budget_usd``
  gate sees the spend. A call that returns no cost (provider didn't report) charges 0.

**Errors are values.** Every failure — a ``workflow:`` target, an untrusted
``api_base``, a malformed request, a provider/deadline error — resolves as a
recoverable ``{"error": …}`` the script branches on, never a run-terminal raise
(matching ``tool()`` / ``agent()``'s "errors resolve" contract).

**At-least-once.** A hard worker crash mid-call leaves no signal; the periodic
sweep re-wakes the run and the step re-dispatches — a second, billable inference.
Inference is read-only (no external mutation), so a re-drive only duplicates spend.
"""

from __future__ import annotations

import asyncio
from typing import Any

import asyncpg

from aios.config import get_settings
from aios.db.queries import workflows as wf_queries
from aios.harness import runtime
from aios.harness.completion import (
    LlmRequest,
    ModelCallDeadlineError,
    call_litellm,
    estimate_cost_usd,
)
from aios.harness.model_tier import (
    UnknownModelTierError,
    is_tier_model,
    resolve_tier_model,
)
from aios.jobs.app import defer_run_wake
from aios.logging import get_logger
from aios.models.attenuation import api_base_of
from aios.models.workflows import WfRun
from aios.services import attenuation as attenuation_service
from aios.services import model_providers as model_providers_service

log = get_logger("aios.workflows.run_llm")

# A model target naming a workflow binding — ``call_llm`` is the inference LEAF and
# must reject it at runtime (the model arg may be computed, so it can't be a static
# check). Match is on the ``workflow:`` scheme prefix, the binding namespace.
_WORKFLOW_MODEL_PREFIX = "workflow:"

# Per-worker in-flight call_llm tasks, keyed (run_id, call_key) — shares the same
# registry semantics as run_tools._INFLIGHT (gate launching, never harvesting), but
# kept separate so a call_llm and a tool with the same call_key never collide.
_INFLIGHT: dict[tuple[str, str], asyncio.Task[None]] = {}


def has_inflight(run_id: str, call_key: str) -> bool:
    """True iff a live call_llm task for ``(run_id, call_key)`` runs on this worker."""
    task = _INFLIGHT.get((run_id, call_key))
    return task is not None and not task.done()


def launch_call_llm_task(
    pool: asyncpg.Pool[Any],
    run: WfRun,
    *,
    call_key: str,
    spec: dict[str, Any],
) -> None:
    """Launch the worker task for a freshly-opened call_llm frontier (no-op if live)."""
    key = (run.id, call_key)
    if has_inflight(*key):
        return
    _INFLIGHT[key] = asyncio.create_task(
        _run_call_llm_task(pool, run, call_key=call_key, spec=spec)
    )


async def _run_call_llm_task(
    pool: asyncpg.Pool[Any],
    run: WfRun,
    *,
    call_key: str,
    spec: dict[str, Any],
) -> None:
    """Run one inference, charge its cost, write its result signal, wake the run."""
    try:
        result, cost_microusd = await invoke_call_llm(run=run, spec=spec)
        try:
            # Charge the cost meter and write the signal in ONE transaction so a budget
            # read on the next step never sees the result without its spend (charge once,
            # at the inference site).
            async with pool.acquire() as conn, conn.transaction():
                await wf_queries.add_run_call_llm_cost_microusd(
                    conn, run.id, cost_microusd, account_id=run.account_id
                )
                await wf_queries.insert_run_signal(
                    conn,
                    run_id=run.id,
                    call_key=call_key,
                    kind="tool_result",
                    result=result,
                )
            await defer_run_wake(run.id, batch=True)
        except Exception:
            # The inference ran but persisting/waking failed (DB blip). If the signal
            # committed and only the wake failed, the sweep re-wakes within a tick; with
            # no signal at all the stale clause re-wakes and the harvest re-dispatches
            # (a second billable call — at-least-once). Log so the stall is diagnosable.
            log.exception("call_llm.signal_failed", run_id=run.id, call_key=call_key)
    finally:
        # CancelledError (worker shutdown) propagates here with no signal — the periodic
        # sweep re-wakes the run and the step re-dispatches.
        _INFLIGHT.pop((run.id, call_key), None)


async def invoke_call_llm(*, run: WfRun, spec: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Resolve one ``call_llm`` call. Returns ``(result, cost_microusd)``.

    ``result`` is the script-facing dict — the raw assistant turn on success, or a
    recoverable ``{"error": …}`` on any rejected/failed call (never raises).
    ``cost_microusd`` is what to charge the run's inference meter: the call's cost
    on success, ``0`` on any error (a rejected/failed call bought nothing).

    The four runtime guards are applied IN ORDER before any inference leaves the
    worker: resolve+validate the model, reject a ``workflow:`` target, clamp the
    effective ``api_base``, then check the account-resolved provider auth doesn't
    conflict with it. Only a call past all four runs.
    """
    messages = spec.get("messages")
    if not isinstance(messages, list):
        return {"error": "call_llm requires a 'messages' list"}, 0

    # Model resolution: the request's model, else the run's default child model.
    model = spec.get("model") or run.default_child_model
    if not isinstance(model, str) or not model:
        return (
            {
                "error": (
                    "call_llm has no model: pass model= in the request or set the "
                    "run's default child model"
                )
            },
            0,
        )

    # ``tier:`` resolution (docs/rlm.md) — before the guards, so guards 1-2
    # see the raw model. This is one of the two seams that bypass the surface
    # chokepoint in ``services.agents``. Tier values can never themselves be
    # ``workflow:``/``tier:`` strings (Settings validation), so guard 1 stays
    # sound post-resolution.
    if is_tier_model(model):
        try:
            model = resolve_tier_model(model, get_settings().model_tiers)
        except UnknownModelTierError as exc:
            return {"error": str(exc)}, 0

    # Guard 1 — ``workflow:`` rejection (leaf-only). The model arg may be computed,
    # so this is a runtime check at the call site, not a static one.
    if model.startswith(_WORKFLOW_MODEL_PREFIX):
        return (
            {
                "error": (
                    f"call_llm rejects a workflow: model target ({model!r}); call_llm "
                    "is leaf-only and cannot bind to a workflow"
                )
            },
            0,
        )

    params = spec.get("params") if isinstance(spec.get("params"), dict) else None

    # Guard 2 — model-identity clamp (#823). A redirected api_base in params would send
    # the whole prompt to another endpoint; admit it only if trusted. A run is the
    # launcher and carries no litellm_extra of its own (None) → the equality arm reduces
    # to "must not redirect"; the operator allowlist is the only way to admit a redirect.
    if not attenuation_service.model_identity_trusted(params, None):
        redirect = api_base_of(params)
        return (
            {
                "error": (
                    f"call_llm routes inference to an untrusted endpoint ({redirect!r}); "
                    "add it to the operator trusted_inference_api_bases allowlist to permit it"
                )
            },
            0,
        )

    # Guard 3 — provider-auth conflict. pool/crypto_box come off the worker-global
    # runtime module — this function has no pool param, matching the
    # connector-auth-resolution precedent in run_tools.py — not threaded through
    # launch_call_llm_task/_run_call_llm_task. resolve_provider_auth_or_conflict
    # fuses resolution with the conflict check on `params`'s redirect (if any) so
    # the two can never run out of order or independently at this call site.
    #
    # The resolve does I/O (DB) and crypto (decrypt), so unlike guards 1-2 it CAN
    # raise (a corrupt/key-mismatched ciphertext row → CryptoDecryptError). This
    # function's contract is "never raises — every failure is a recoverable value"
    # (see module docstring); an uncaught raise here would escape _run_call_llm_task
    # (which has no outer except) with no result signal, and the sweep would
    # re-dispatch forever — a silent wedge. So a persistent resolve failure resolves
    # as an error the script branches on, exactly like a provider error from the
    # inference call below. Transient DB errors self-heal via the sweep's re-wake.
    pool = runtime.require_pool()
    try:
        auth, conflict = await model_providers_service.resolve_provider_auth_or_conflict(
            pool,
            runtime.require_crypto_box(),
            account_id=run.account_id,
            model=model,
            litellm_extra=params,
        )
    except Exception as exc:
        log.warning("call_llm.provider_auth_error", run_id=run.id, model=model, error=str(exc))
        return {
            "error": f"call_llm provider-auth resolution failed: {type(exc).__name__}: {exc}"
        }, 0
    if conflict is not None:
        return {"error": f"call_llm refused: {conflict}"}, 0
    if auth is None and (
        get_settings().inference_credential_policy == "account_only"
        or get_settings().tenancy_posture == "external_byok"
    ):
        return {
            "error": model_providers_service.PROVIDER_NOT_CONFIGURED_MESSAGE,
            "error_kind": "model_provider_not_configured",
        }, 0

    tools = spec.get("tools") if isinstance(spec.get("tools"), list) else None
    session_id = spec.get("session_id") if isinstance(spec.get("session_id"), str) else None
    request = LlmRequest(messages=messages, tools=tools, params=params, session_id=session_id)

    try:
        response = await call_litellm(request, model=model, auth=auth)
    except ModelCallDeadlineError as exc:
        # A deadline still spent provider time — charge whatever the partial usage
        # estimates, so a run can't dodge its budget by timing out. Result is the
        # recoverable error value the script branches on.
        cost = estimate_cost_usd(model, exc.usage) if exc.usage else None
        return {"error": f"call_llm timed out: {exc}"}, _to_microusd(cost)
    except Exception as exc:  # provider error — surface as a recoverable value
        log.warning("call_llm.provider_error", run_id=run.id, model=model, error=str(exc))
        return {"error": f"call_llm failed: {type(exc).__name__}: {exc}"}, 0

    result = {
        "content": response.content,
        "tool_calls": response.tool_calls,
        "finish_reason": response.finish_reason,
        "usage": response.usage,
        "cost": response.cost,
        "message": response.message,
    }
    return result, _to_microusd(response.cost)


def _to_microusd(cost_usd: float | None) -> int:
    """Round a USD cost to a non-negative micro-USD integer; ``None``/negative → 0."""
    if cost_usd is None or cost_usd <= 0:
        return 0
    return round(cost_usd * 1_000_000)
