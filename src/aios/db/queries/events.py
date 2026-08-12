"""Event-log queries.

A subsystem module of the ``aios.db.queries`` package — see ``__init__`` for the
shared scoping helpers and the package-level re-export contract. Raw SQL against
asyncpg, same conventions as the rest of the package.
"""

from __future__ import annotations

import json
import math
import time
from datetime import UTC, datetime
from types import EllipsisType
from typing import Any, Final, NamedTuple, Protocol, runtime_checkable

import asyncpg

from aios.db import queries
from aios.db.queries.connections import _session_bound_to_connection_predicate
from aios.errors import (
    NotFoundError,
)
from aios.harness.chat_type import ChatType, chat_type_of
from aios.harness.window import WindowedEvents, WindowOmission
from aios.ids import (
    EVENT,
    make_id,
)
from aios.models.events import (
    MODEL_VISIBLE_LIFECYCLE_EVENTS,
    Event,
    EventKind,
    is_errored_lifecycle_event,
)


@runtime_checkable
class OverheadLocalSplit(Protocol):
    """Structural type for the overhead-local split the windower weights.

    :class:`aios.harness.step_context.PreludeOverheadSplit` satisfies this.
    Declared structurally here so ``events.py`` does not import the harness
    composer (avoids a layering cycle).  ``system`` / ``tools`` / ``reserves``
    are the per-class local token costs; ``total`` is their sum (the
    pre-#1609 single ``overhead_local`` scalar).
    """

    system: int
    tools: int
    reserves: int

    @property
    def total(self) -> int: ...


# ─── events ───────────────────────────────────────────────────────────────────


def _row_to_event(row: asyncpg.Record) -> Event:
    raw_data = row["data"]
    data = raw_data
    return Event(
        id=row["id"],
        session_id=row["session_id"],
        seq=row["seq"],
        kind=row["kind"],
        data=data,
        cumulative_tokens=row["cumulative_tokens"],
        created_at=row["created_at"],
        orig_channel=row["orig_channel"],
        focal_channel_at_arrival=row["focal_channel_at_arrival"],
        channel=row["channel"],
    )


async def _latest_cumulative_tokens(conn: asyncpg.Connection[Any], session_id: str) -> int | None:
    """Fetch the cumulative_tokens value of the most recent message event."""
    val: int | None = await conn.fetchval(
        "SELECT cumulative_tokens FROM events "
        "WHERE session_id = $1 AND kind = 'message' "
        "AND cumulative_tokens IS NOT NULL "
        "ORDER BY seq DESC LIMIT 1",
        session_id,
    )
    return val


# The four per-message content classes whose neutral token mass the windower
# blends R_eff over (issue #1657). This is deliberately the *message-level*
# subset of :data:`aios.harness.tokens.CONTENT_CLASSES` -- the ``system`` and
# ``tools`` classes come from the caller's overhead split, never from a stored
# message row, so they carry no cumulative column.
_MESSAGE_CONTENT_CLASSES = ("text", "tool_result", "thinking", "tool_use")

# Per-class cumulative-mass column names, keyed by content class. Kept next to
# the classifier so the append-time increment and the read-time seek share one
# source of truth for the column <-> class mapping.
_CLASS_MASS_COLUMN = {
    "text": "cumulative_text_mass",
    "tool_result": "cumulative_tool_result_mass",
    "thinking": "cumulative_thinking_mass",
    "tool_use": "cumulative_tool_use_mass",
}


def _message_content_class(role: str | None, data: dict[str, Any]) -> str:
    """Classify a message event into its dominant content class.

    Mirrors :func:`aios.harness.tokens.content_class` AND the ``CASE`` that
    :func:`_retained_class_mass` used to run in SQL -- the append-time
    per-class running sum must partition mass into exactly the buckets the
    (now-removed) full-slate WindowAgg produced, or the stored cumulative
    would diverge from the fallback estimator on the same slate.

    Message events never carry ``role == 'system'`` (the system prompt is
    composed at build time, not logged), so unlike ``content_class`` this
    returns only one of :data:`_MESSAGE_CONTENT_CLASSES`. A defensive
    ``system`` role folds into ``text`` (its neutral shape), keeping every
    message's mass attributable to a stored bucket.
    """
    if role == "tool":
        return "tool_result"
    if role == "assistant":
        tool_calls = data.get("tool_calls")
        if isinstance(tool_calls, list) and len(tool_calls) > 0:
            return "tool_use"
        if any(data.get(f) for f in ("reasoning_content", "thinking_blocks")):
            return "thinking"
    return "text"


class _CumulativeState(NamedTuple):
    """The running-sum state carried on the latest message row (issue #1657).

    ``tokens`` is the pre-existing ``cumulative_tokens`` total; ``messages`` is
    the running count of user+assistant messages (secondary term); ``mass`` is
    the per-content-class running neutral-token mass (primary term). ``None``
    fields mean "no prior cumulative data" -- the append path seeds from 0 and
    the read path falls back to the full-scan estimator.
    """

    tokens: int | None
    messages: int | None
    mass: dict[str, int | None]


async def _latest_cumulative_state(
    conn: asyncpg.Connection[Any], session_id: str
) -> _CumulativeState:
    """Fetch the running cumulative counters from the most recent message row.

    One index seek on ``events_session_cumtokens_idx`` (``ORDER BY seq DESC``,
    the latest message). Returns the ``cumulative_tokens`` total plus the
    ``cumulative_messages`` count and the four per-class mass running sums the
    append path increments (issue #1657). All-``None`` when the session has no
    message with cumulative data yet (pre-backfill / brand-new session).
    """
    row = await conn.fetchrow(
        "SELECT cumulative_tokens, cumulative_messages, "
        "       cumulative_text_mass, cumulative_tool_result_mass, "
        "       cumulative_thinking_mass, cumulative_tool_use_mass "
        "FROM events "
        "WHERE session_id = $1 AND kind = 'message' "
        "AND cumulative_tokens IS NOT NULL "
        "ORDER BY seq DESC LIMIT 1",
        session_id,
    )
    if row is None:
        return _CumulativeState(
            tokens=None,
            messages=None,
            mass={c: None for c in _MESSAGE_CONTENT_CLASSES},
        )
    return _CumulativeState(
        tokens=row["cumulative_tokens"],
        messages=row["cumulative_messages"],
        mass={
            "text": row["cumulative_text_mass"],
            "tool_result": row["cumulative_tool_result_mass"],
            "thinking": row["cumulative_thinking_mass"],
            "tool_use": row["cumulative_tool_use_mass"],
        },
    )


_MODEL_TOKEN_RATIO_MIN_SAMPLES = 5
# Hard cap on the number of calibration spans the per-model ridge fit pulls
# per query.  Without it the fetch had no ORDER BY / LIMIT / time bound and
# scanned *every* successful calibration span ever logged for the model — a
# row count that grows linearly and forever — on the step hot path
# (``read_windowed_events``), eventually tripping the pool's 30 s
# ``statement_timeout`` inside the session step and JSON-decoding all N rows
# on the event loop.  1000 is large enough for a stable ridge fit while
# bounding the fit to the newest rows (``ORDER BY created_at DESC LIMIT $2``).
# See issue #1711.  ``seq`` is session-local and cannot express global recency.
_MODEL_TOKEN_RATIO_SAMPLE_LIMIT = 1000
_MODEL_TOKEN_RATIO_MIN = 0.5
_MODEL_TOKEN_RATIO_BUCKET_FLOOR = 0.001
_MODEL_TOKEN_RATIO_CACHE_TTL_SECONDS = 60.0
# Shorter TTL for the "not enough samples yet" path: every step on a freshly
# deployed model fired this aggregate JSONB scan otherwise, because the
# below-threshold branch used to skip the cache write entirely.  10 s bounds
# the activation lag once the model crosses the sample threshold.
_MODEL_TOKEN_RATIO_BELOW_THRESHOLD_CACHE_TTL_SECONDS = 10.0
# Fixed per-sample stddev prior for the tokenizer ratio.  Empirically,
# observed per-span CV is ~0.5-1.5 % across the models we've measured
# (Opus 4.7: 0.75 %; Haiku 4.5: ~1 %), so 0.02 is a conservative upper
# bound.  Keeping this fixed (rather than using the observed sample
# stddev) makes the bucket width a deterministic function of ``n`` alone
# — the core property #170 / #171 require: quantization stability is a
# function of ``(n, mean)`` only, independent of the noisy observed-
# stddev estimate that wobbles at small n.
_MODEL_TOKEN_RATIO_SIGMA_PRIOR = 0.02

# ── per-class calibration (issue #1609) ───────────────────────────────────
# The single lifetime scalar R is generalized to a per-content-class
# coefficient vector, fit by collinearity-robust ridge least squares over
# logged ``model_request_end`` spans (response = provider input_tokens,
# regressors = the stamped ``local_tokens_by_class`` vector).  The windower
# consumes only the *blended* R_eff (a composition-weighted average of the
# coefficients) — the raw per-class coefficients are NOT individually
# load-bearing (they are collinear; see the issue's robustness note).
from aios.harness.tokens import CONTENT_CLASSES  # noqa: E402

# Ridge regularization strength.  Pulls each coefficient toward a neutral
# prior of 1.0 (the old scalar's degenerate "no correction" value).  The
# regressors are severely collinear (tool_result x thinking r~0.979), so a
# bare OLS fit lets mass swing wildly between two coefficients as spans
# arrive; ridge damps that while leaving the *blend* the windower sees
# stable.  Tuned so a single calibration class folds back toward neutral
# 1.0 when it is under-identified rather than swinging past the ±25%
# stability gate (issue #1609 acceptance #8).
_MODEL_TOKEN_CLASS_RIDGE_LAMBDA = 1e-2
# Per-coefficient prior the ridge penalty shrinks toward: neutral 1.0, so
# the byte-identical-fallback property holds class-by-class.
_MODEL_TOKEN_CLASS_PRIOR = 1.0
# Physical clamps on a fitted coefficient: a provider tokenizer never
# prices a class below 0 tok/local-tok, and the windower divides by the
# blend so a near-zero blend must be guarded.  Mirrors the scalar's
# ``_MODEL_TOKEN_RATIO_MIN`` lower bound and adds a generous upper bound.
_MODEL_TOKEN_CLASS_MIN = 0.05
_MODEL_TOKEN_CLASS_MAX = 8.0

# Safety-margin slack between the budgeted window and the provider hard cap
# (issue #1609 acceptance #4).  Sized to the per-class residual MAX, not the
# mean: honest out-of-sample LOO max APE = 29.2%, so ~30% of slack means a
# composition shift toward the heaviest class degrades to "drops one chunk
# early," never a cap breach.  The windower inflates the budgeted total by
# this factor before the snap math, so the *real* sent prompt sits at least
# this far under ``window_max``.
_WINDOW_SAFETY_MARGIN = 0.30

_model_token_ratio_cache: dict[tuple[str, float], tuple[float, dict[str, float]]] = {}


def _clear_model_token_ratio_cache() -> None:
    """Clear the process-local token-ratio cache for tests."""
    _model_token_ratio_cache.clear()


def _neutral_class_ratios() -> dict[str, float]:
    """The all-1.0 coefficient dict — reduces the windower to today's
    byte-identical model-neutral behavior (issue #1609 acceptance #5)."""
    return {c: 1.0 for c in CONTENT_CLASSES}


def _solve_ridge(
    rows: list[tuple[list[float], float]],
    *,
    n_features: int,
    lam: float,
    prior: float,
) -> list[float] | None:
    """Ridge-regularized linear least squares via the normal equations.

    Solves ``min over beta of sum_i (x_i . beta - y_i)^2 + lam . sum_j (beta_j - prior)^2`` for the small
    fixed-size system (``n_features`` ≈ 6 class regressors).  Pure Python
    (no numpy dependency — this is a fleet-wide harness module).  Returns
    ``None`` if the regularized normal matrix is singular (degenerate
    data), so the caller folds back to the neutral all-1.0 vector.

    The ridge shift toward ``prior`` (here 1.0) is what damps the
    collinear coefficient swings and gives the neutral-fallback its
    class-by-class meaning.
    """
    p = n_features
    # Normalize the whole least-squares system to token-scale units before
    # forming XᵀX.  Without this, ~1e10 normal-matrix entries make λ=1e-2
    # numerically equivalent to unregularized OLS.  A single global scale
    # preserves coefficient units and the neutral 1.0 prior.
    # Keep typical normalized normal-matrix entries around 1e2 rather than
    # 1e10.  This leaves data dominant while making λ representable and
    # consequential for weak/collinear directions.
    scale = math.sqrt(sum(sum(v * v for v in x) for x, _ in rows) / max(len(rows), 1)) / 10.0
    if not math.isfinite(scale) or scale <= 0:
        return None
    normalized_rows = [([v / scale for v in x], y / scale) for x, y in rows]

    # Normal equations: (XᵀX + λI) β = Xᵀy + λ·prior·1
    ata = [[0.0] * p for _ in range(p)]
    aty = [0.0] * p
    for x, y in normalized_rows:
        for j in range(p):
            aty[j] += x[j] * y
            xj = x[j]
            row_j = ata[j]
            for k in range(p):
                row_j[k] += xj * x[k]
    for j in range(p):
        ata[j][j] += lam
        aty[j] += lam * prior
    return _gaussian_solve(ata, aty)


def _gaussian_solve(a: list[list[float]], b: list[float]) -> list[float] | None:
    """Solve ``a·x = b`` by Gaussian elimination with partial pivoting.

    Returns ``None`` if ``a`` is (numerically) singular.
    """
    n = len(b)
    # Work on copies.
    m = [[*row[:], b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            return None
        m[col], m[pivot] = m[pivot], m[col]
        piv = m[col][col]
        for r in range(n):
            if r == col:
                continue
            factor = m[r][col] / piv
            if factor == 0.0:
                continue
            for c in range(col, n + 1):
                m[r][c] -= factor * m[col][c]
    return [m[i][n] / m[i][i] for i in range(n)]


async def model_token_class_ratios(
    conn: asyncpg.Connection[Any],
    model: str,
    *,
    account_id: str,
    k_bucket: float = 2.0,
) -> dict[str, float]:
    """Per-(model, content-class) actual/local token coefficients (#1609).

    Generalizes the old single lifetime scalar ``R`` into a coefficient
    dict over :data:`~aios.harness.tokens.CONTENT_CLASSES`, fit by
    **collinearity-robust ridge least squares** over logged
    ``model_request_end`` spans:

    * **response** = the provider's ``input_tokens`` usage value;
    * **regressors** = the stamped ``local_tokens_by_class`` vector
      (model-neutral per-class local counts).

    The ridge penalty shrinks each coefficient toward the neutral prior
    ``1.0``.  The regressors are severely collinear (``tool_result x
    thinking`` r≈0.979), so a bare OLS fit would let mass swing wildly
    between two coefficients as new spans arrive; the regularizer damps
    that while leaving the **blended R_eff** the windower actually
    consumes stable.  **Do NOT depend on the raw per-class coefficients
    as individually meaningful** — they are self-validated until ≥10
    substantial-thinking windows from ≥3 distinct sessions accumulate
    (issue #1609 acceptance #2).

    Below the per-model sample threshold (or when the regularized normal
    system is degenerate / singular), every coefficient is the neutral
    ``1.0`` — so a freshly deployed model preserves the old
    model-agnostic windowing behavior **byte-identically** until
    calibration is meaningful (acceptance #5).  Old spans lacking
    ``local_tokens_by_class`` are excluded by a ``data ?`` predicate
    (zero backfill — identical to the existing ``data ? 'local_tokens'``
    guard), so the regression trains only on NEW spans and self-heals as
    they accumulate (same contract as #160).

    Mature fits are cached 60 s; below-threshold neutral results are cached
    10 s to bound the activation lag.  ``k_bucket`` partitions the cache so
    callers using different calibration bucket widths do not share fitted
    results.  The per-class fit's stability comes from the ridge regularizer,
    not bucket quantization.

    ``model`` is the raw model string (``agent.model``) — NO
    NORMALIZATION; the same string must appear at stamp and query time.
    "actual" is the provider's ``input_tokens`` (the full prompt count,
    cached portions included — do NOT add ``cache_*`` breakdowns).  Uses
    the ``events_model_calibration_recency_idx`` partial index (migration
    0142), keyed ``((data->>'model'), created_at DESC, session_id DESC,
    seq DESC)`` so the ``ORDER BY … LIMIT`` recency fetch is a bounded
    index seek rather than a full-model scan + sort (issue #1711).

    **``account_id`` is intentionally not a query predicate — by design.**
    These coefficients are model-intrinsic: the provider ``input_tokens``
    is regressed on model-neutral, account-neutral local class counts, so
    the fit is a deliberate *global-per-model* aggregate, not a
    per-tenant one.  This is NOT a tenant leak.  Do **not** add an
    ``account_id`` predicate: it would fragment the sample (slowing
    calibration convergence) and require a new index without buying any
    correctness.  The parameter retains the caller's account-scoped query
    interface while the cache and fit intentionally remain global per model.

    The fetch is bounded to the most recent
    :data:`_MODEL_TOKEN_RATIO_SAMPLE_LIMIT` spans via ``ORDER BY created_at
    DESC LIMIT`` (issue #1711).  ``seq`` is only session-local, so it cannot
    rank spans pooled across sessions.  The limit caps both the SQL result
    and the per-row JSON decode done on the event loop.
    """
    if k_bucket <= 0:
        raise ValueError("k_bucket must be positive")

    cache_key = (model, k_bucket)
    now = time.monotonic()
    cached = _model_token_ratio_cache.get(cache_key)
    if cached is not None:
        expires_at, ratios = cached
        if expires_at > now:
            return dict(ratios)
        del _model_token_ratio_cache[cache_key]

    fitted, n_samples = await model_token_class_ratio_fit(conn, model, account_id=account_id)

    if n_samples < _MODEL_TOKEN_RATIO_MIN_SAMPLES or fitted is None:
        neutral = _neutral_class_ratios()
        _model_token_ratio_cache[cache_key] = (
            now + _MODEL_TOKEN_RATIO_BELOW_THRESHOLD_CACHE_TTL_SECONDS,
            neutral,
        )
        return dict(neutral)

    _model_token_ratio_cache[cache_key] = (
        now + _MODEL_TOKEN_RATIO_CACHE_TTL_SECONDS,
        fitted,
    )
    return dict(fitted)


async def model_token_class_ratio_fit(
    conn: asyncpg.Connection[Any], model: str, *, account_id: str
) -> tuple[dict[str, float] | None, int]:
    """Return a fresh bounded calibration fit and its usable sample count.

    This is the uncached calibration reader shared by the hot-path cached
    wrapper and the on-demand observability surface. ``account_id`` remains
    intentionally unused because calibration is global per model.
    """
    del account_id
    rows = await conn.fetch(
        """
        SELECT
            (data->'model_usage'->>'input_tokens')::float AS it,
            data->'local_tokens_by_class'                 AS by_class
        FROM events
        WHERE kind = 'span'
          AND data->>'event' = 'model_request_end'
          AND (data->>'is_error')::boolean = false
          AND data->>'model' = $1
          AND data ? 'local_tokens'
          AND data ? 'local_tokens_by_class'
          AND data ? 'model'
          AND (data->'model_usage') ? 'input_tokens'
          AND (data->'model_usage'->>'input_tokens') IS NOT NULL
          -- Regex gates make the casts defensive against future malformed or
          -- non-integral JSON values (e.g. "150.5").
          AND (data->'model_usage'->>'input_tokens') ~ '^[0-9]+$'
          AND (data->'model_usage'->>'input_tokens')::numeric > 0
          AND (data->>'local_tokens') ~ '^[0-9]+$'
          AND (data->>'local_tokens')::numeric > 0
        -- Bound the scan to the globally most recent N spans (issue #1711).
        -- seq is session-local, so created_at must lead across sessions.
        -- MIN_SAMPLES is still checked below the fetch.
        ORDER BY created_at DESC, session_id DESC, seq DESC
        LIMIT $2
        """,
        model,
        _MODEL_TOKEN_RATIO_SAMPLE_LIMIT,
    )
    return _fit_class_ratios(rows), len(rows)


async def calibration_telemetry(conn: asyncpg.Connection[Any]) -> dict[str, dict[str, Any]]:
    """Compute fit-vs-measured token ratios for models active in the last 24h.

    ``fitted_r_eff`` is the *composition-weighted* blended ratio the windower
    actually consumes (:func:`blended_r_eff`: ``Σ_c coef_c·local_c / Σ_c
    local_c``), evaluated over a representative composition — the summed
    ``local_tokens_by_class`` mass over the same 24h window as
    ``measured_ratio``.  It is NOT the unweighted arithmetic mean of the
    per-class coefficients: for a skewed class mix the two diverge
    substantially, so the mean would put ``fitted_r_eff`` on a different scale
    than ``measured_ratio`` and make the ops-agent's ``|fitted - measured|``
    divergence check miss or falsely fire — the exact failure this endpoint
    exists to detect.  ``fitted_coefficients`` still carries the raw per-class
    fit for finer diagnosis.
    """
    # Aggregate the recent class composition alongside the measured ratio.  The
    # per-class mass columns are built from the trusted ``CONTENT_CLASSES``
    # constant (never user input — no injection surface), and ``coalesce``
    # keeps a class absent from every span at 0.0 mass rather than NULL.
    class_mass_cols = ",\n".join(
        f"               coalesce(sum((data->'local_tokens_by_class'->>'{c}')::float), 0.0) "
        f"AS mass_{c}"
        for c in CONTENT_CLASSES
    )
    measured = await conn.fetch(
        f"""
        SELECT data->>'model' AS model,
               avg((data->'model_usage'->>'input_tokens')::float /
                   (data->>'local_tokens')::float) AS mean_ratio,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY
                   (data->'model_usage'->>'input_tokens')::float /
                   (data->>'local_tokens')::float) AS p50_ratio,
               count(*) AS n_samples,
{class_mass_cols}
        FROM events
        WHERE kind = 'span'
          AND data->>'event' = 'model_request_end'
          AND (data->>'is_error')::boolean = false
          AND created_at >= now() - interval '24 hours'
          AND data ? 'local_tokens'
          AND data ? 'local_tokens_by_class'
          AND data ? 'model'
          AND (data->'model_usage') ? 'input_tokens'
          AND (data->'model_usage'->>'input_tokens') IS NOT NULL
          AND (data->'model_usage'->>'input_tokens')::bigint > 0
          AND (data->>'local_tokens')::bigint > 0
        GROUP BY data->>'model'
        ORDER BY data->>'model'
        """
    )
    result: dict[str, dict[str, Any]] = {}
    for row in measured:
        model = str(row["model"])
        coefficients, fitted_n = await model_token_class_ratio_fit(
            conn, model, account_id="telemetry"
        )
        effective = coefficients or _neutral_class_ratios()
        composition = {c: float(row[f"mass_{c}"]) for c in CONTENT_CLASSES}
        result[model] = {
            "fitted_r_eff": blended_r_eff(effective, composition),
            "fitted_coefficients": effective,
            "measured_ratio": {
                "mean": float(row["mean_ratio"]),
                "p50": float(row["p50_ratio"]),
            },
            "n_samples": {"fitted": fitted_n, "measured": int(row["n_samples"])},
        }
    return result


def _fit_class_ratios(rows: list[Any]) -> dict[str, float] | None:
    """Ridge-fit the per-class coefficient dict from calibration rows.

    Each row carries provider ``it`` (input_tokens) and the stamped
    ``by_class`` local vector.  Builds the design matrix over
    :data:`~aios.harness.tokens.CONTENT_CLASSES` (skipping classes that
    are identically zero across all spans — they are unidentified, and
    fold to the neutral ``1.0``), fits ridge least squares, then clamps
    each coefficient.  Returns ``None`` if the system is singular.
    """
    classes = list(CONTENT_CLASSES)

    design: list[list[float]] = []
    responses: list[float] = []
    col_mass = [0.0] * len(classes)
    for r in rows:
        raw = r["by_class"]
        by_class = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(by_class, dict):
            continue
        x = [float(by_class.get(c, 0) or 0) for c in classes]
        design.append(x)
        responses.append(float(r["it"]))
        for j, v in enumerate(x):
            col_mass[j] += v

    if len(design) < _MODEL_TOKEN_RATIO_MIN_SAMPLES:
        return None

    # Classes with no mass across the whole sample are unidentified — drop
    # them from the fit (they keep the neutral 1.0).  This also lets the
    # ridge prior anchor the present classes without a spurious zero
    # regressor inflating the conditioning.
    active = [j for j in range(len(classes)) if col_mass[j] > 0.0]
    if not active:
        return None

    fit_rows = [([x[j] for j in active], y) for x, y in zip(design, responses, strict=True)]
    coefs = _solve_ridge(
        fit_rows,
        n_features=len(active),
        lam=_MODEL_TOKEN_CLASS_RIDGE_LAMBDA,
        prior=_MODEL_TOKEN_CLASS_PRIOR,
    )
    if coefs is None:
        return None

    out = _neutral_class_ratios()
    for idx, j in enumerate(active):
        c = coefs[idx]
        # Clamp to the physical range; the windower divides by the blend.
        c = max(_MODEL_TOKEN_CLASS_MIN, min(_MODEL_TOKEN_CLASS_MAX, c))
        out[classes[j]] = c
    return out


def blended_r_eff(ratios: dict[str, float], local_by_class: dict[str, float]) -> float:
    """Composition-weighted blended effective ratio ``R_eff`` (#1609).

    ``R_eff = Σ_c coef_c · local_c / Σ_c local_c`` — the only quantity the
    windower consumes from the per-class fit.  When the composition has no
    mass (all-zero), falls back to the unweighted mean of the coefficients
    (degenerate; the windower guards the all-zero case separately).
    """
    total = sum(local_by_class.get(c, 0.0) for c in ratios)
    if total <= 0:
        blend = sum(ratios.values()) / len(ratios)
    else:
        blend = sum(ratios[c] * float(local_by_class.get(c, 0.0)) for c in ratios) / total
    return max(blend, _MODEL_TOKEN_RATIO_MIN)


def _derive_tool_name(kind: str, data: dict[str, Any]) -> str | None:
    """Compute the stamped ``tool_name`` column for a new event.

    For tool-result events the name lives at ``data->>'name'``.  For
    assistant events that requested tools, the first tool_call's function
    name is promoted — multi-tool turns remain discoverable by that first
    name; the full list still lives in ``data->'tool_calls'``.  Pure
    function; paths mirror the backfill in migration 0022 so old and new
    rows stay byte-equivalent in this column.
    """
    if kind != "message":
        return None
    role = data.get("role")
    if role == "tool":
        name = data.get("name")
        return name if isinstance(name, str) else None
    if role == "assistant":
        tool_calls = data.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            return None
        first = tool_calls[0]
        if not isinstance(first, dict):
            return None
        function = first.get("function")
        if not isinstance(function, dict):
            return None
        name = function.get("name")
        return name if isinstance(name, str) else None
    return None


def _derive_sender_name(kind: str, data: dict[str, Any]) -> str | None:
    """Sender name for user events carrying connector metadata; else NULL."""
    if kind != "message" or data.get("role") != "user":
        return None
    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        return None
    name = metadata.get("sender_name")
    return name if isinstance(name, str) else None


def _derive_is_error(kind: str, data: dict[str, Any]) -> bool | None:
    """Error flag on events that carry ``is_error``; NULL when absent.

    Originally restricted to message-kind events (tool-result rows), but
    span events also carry ``is_error`` (e.g. ``model_request_end``,
    ``step_timeout``, ``harness_error``).  We now write the physical column
    for any kind that includes the field so that ``?error_only=true``
    filtering works across all event kinds.
    """
    flag = data.get("is_error")
    if flag is None:
        return None
    return bool(flag)


async def _lookup_tool_parent_channel(
    conn: asyncpg.Connection[Any],
    session_id: str,
    tool_call_id: Any,
    *,
    account_id: str,
) -> str | None:
    """Look up the ``focal_channel_at_arrival`` of the assistant event that
    requested ``tool_call_id`` — the channel a tool-role result belongs to.

    Matches ``tool_call_id`` against prior assistant rows'
    ``data->'tool_calls'``. Returns NULL if no parent is found (shouldn't
    happen in practice — tool results only arrive for assistant-requested
    tool calls — but the recap filter tolerates NULL). A non-str or empty
    ``tool_call_id`` also yields NULL.

    Pulled out of the old ``_derive_event_channel`` so ``append_event`` can
    run it BEFORE the row lock (issue #862), keeping the transaction free of
    this JSONB ``@>`` scan.
    """
    if not isinstance(tool_call_id, str) or not tool_call_id:
        return None
    # Predicates match ``events_assistant_tool_calls_idx`` (partial
    # index on (session_id, seq) for role=assistant rows that have
    # tool_calls — migration 0011) so the planner can walk it in
    # reverse-seq order and stop at the first matching parent.
    parent_focal: str | None = await conn.fetchval(
        "SELECT focal_channel_at_arrival FROM events "
        "WHERE session_id = $1 "
        "  AND account_id = $3 "
        "  AND kind = 'message' "
        "  AND data->>'role' = 'assistant' "
        "  AND data ? 'tool_calls' "
        "  AND data->'tool_calls' @> jsonb_build_array("
        "    jsonb_build_object('id', $2::text)) "
        "ORDER BY seq DESC LIMIT 1",
        session_id,
        tool_call_id,
        account_id,
    )
    return parent_focal


def _resolve_event_channel(
    kind: str,
    data: dict[str, Any],
    orig_channel: str | None,
    focal_at_arrival: str | None,
    tool_parent_channel: str | None,
) -> str | None:
    """Pure role dispatch for the derived ``channel`` column — no I/O.

    User events → ``orig_channel``.
    Assistant events → ``focal_at_arrival`` (the live focal at stamp time).
    Tool events → ``tool_parent_channel`` (the parent assistant's
    ``focal_channel_at_arrival``, resolved by the caller via
    :func:`_lookup_tool_parent_channel` or supplied by the live dispatch path).

    Non-message events and message events with no identifiable role
    return NULL.

    CROSS-REFERENCE (issue #1742): the ``channels`` SET clause in
    :func:`append_event`'s seq-allocating UPDATE re-derives this SAME role
    dispatch in SQL (user/tool arms via the ``chan_candidate`` param computed
    just above that UPDATE; the assistant arm via ``is_assistant_message`` +
    the row's own ``focal_channel``, which is exactly the ``focal_at_arrival``
    value RETURNING hands this function's caller, since the UPDATE never
    mutates ``focal_channel``). The two MUST stay identical — the integration
    test matrix in ``tests/integration/test_tool_channel_stamp.py`` and the
    sessions-channels invariant test enforce this by asserting
    ``list_session_channels`` == ``recompute_session_channels`` after every
    scripted append.
    """
    if kind != "message":
        return None
    role = data.get("role")
    if role == "user":
        return orig_channel
    if role == "assistant":
        return focal_at_arrival
    if role == "tool":
        return tool_parent_channel
    return None


def _event_token_delta(
    kind: str,
    data: dict[str, Any],
    orig_channel: str | None,
    focal_at_arrival: str | None,
) -> int:
    """Approximate per-event token contribution, computed pre-transaction.

    Mirrors the as-rendered form the windowing budget expects so the
    ``cumulative_tokens`` column stays honest for non-focal notification
    markers (which occupy far fewer tokens than their full-content
    counterparts):

    * non-message → 0 (only message events carry ``cumulative_tokens``);
    * user message → ``render_user_event(...)`` paired with an assistant
      separator (pre-paying for ``merge_adjacent_user_messages``), counted
      together;
    * any other message → ``approx_tokens([data])``.

    ``render_user_event``/``approx_tokens`` are imported lazily to preserve
    the litellm-bootstrap deferral of the original in-lock code.
    """
    if kind != "message":
        return 0
    from aios.harness.context import _USER_MESSAGE_SEPARATOR_CONTENT, render_user_event
    from aios.harness.tokens import approx_tokens

    if data.get("role") == "user":
        # ``created_at`` isn't assigned until the INSERT (DB DEFAULT now()),
        # so render with a now() stand-in in the default UTC zone — same
        # bounded drift the in-lock code accepted (see ``append_event``).
        rendered = render_user_event(data, orig_channel, focal_at_arrival, datetime.now(UTC))
        separator = {"role": "assistant", "content": _USER_MESSAGE_SEPARATOR_CONTENT}
        return approx_tokens([rendered, separator])
    return approx_tokens([data])


async def find_tool_result_event(
    conn: asyncpg.Connection[Any],
    session_id: str,
    tool_call_id: str,
    *,
    account_id: str,
) -> Event | None:
    """Return the existing tool-role event for ``tool_call_id``, or ``None``.

    Used by ``services.append_tool_result`` to make the intake idempotent
    on ``(session_id, tool_call_id)``: a retried POST returns the original
    event instead of appending a duplicate that would later violate the
    monotonic-context invariant (``harness/context.py:499-506`` keeps the
    latest tool_result per id by dict-overwrite — duplicates silently
    rewrite history).
    """
    row = await conn.fetchrow(
        """
        SELECT * FROM events
         WHERE session_id = $1
           AND account_id = $2
           AND kind = 'message'
           AND role = 'tool'
           AND data->>'tool_call_id' = $3
         LIMIT 1
        """,
        session_id,
        account_id,
        tool_call_id,
    )
    return _row_to_event(row) if row is not None else None


async def find_user_message_by_client_message_id(
    conn: asyncpg.Connection[Any],
    session_id: str,
    client_message_id: str,
    *,
    account_id: str,
) -> Event | None:
    """Return the user event carrying a canonical client message UUID.

    The predicate mirrors migration 0113's partial unique index so retries are
    an indexed lookup and the database remains the structural concurrency
    floor for ``(account_id, session_id, client_message_id)``.
    """
    row = await conn.fetchrow(
        """
        SELECT * FROM events
         WHERE account_id = $1
           AND session_id = $2
           AND kind = 'message'
           AND role = 'user'
           AND data->'metadata'->>'client_message_id' = $3
           AND data->'metadata'->>'client_message_id'
               ~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
         LIMIT 1
        """,
        account_id,
        session_id,
        client_message_id,
    )
    return _row_to_event(row) if row is not None else None


async def find_tool_confirmed_event(
    conn: asyncpg.Connection[Any],
    session_id: str,
    tool_call_id: str,
    *,
    account_id: str,
) -> Event | None:
    """Return the existing ``lifecycle/tool_confirmed`` event for
    ``tool_call_id``, or ``None``.

    Used by ``services.confirm_tool_allow`` to make the intake
    idempotent on ``(session_id, tool_call_id)``: a retried POST returns
    the original event instead of appending a duplicate. Mirrors the
    same-shape sibling :func:`find_tool_result_event` (used by the deny
    twin's idempotency).
    """
    row = await conn.fetchrow(
        """
        SELECT * FROM events
         WHERE session_id = $1
           AND account_id = $2
           AND kind = 'lifecycle'
           AND data->>'event' = 'tool_confirmed'
           AND data->>'tool_call_id' = $3
         LIMIT 1
        """,
        session_id,
        account_id,
        tool_call_id,
    )
    return _row_to_event(row) if row is not None else None


async def find_latest_interrupt_seq(
    conn: asyncpg.Connection[Any],
    session_id: str,
    *,
    account_id: str,
) -> int | None:
    """Return the ``seq`` of the most recent ``kind='interrupt'`` event for
    ``session_id``, or ``None`` if the session has never been interrupted.

    Consumed by the confirmed-tool cold-dispatch guard (#1756,
    :func:`aios.harness.loop._dispatch_confirmed_tools`): a
    ``tool_confirmed``/``allow`` whose event ``seq`` is LOWER than this value
    was confirmed BEFORE the latest interrupt and must not be cold-dispatched
    -- the durable interrupt marker the ``/interrupt`` endpoint writes
    (``routers/sessions.py``) is otherwise never consulted at this dispatch
    point. A FRESH confirmation issued AFTER the interrupt has a higher seq
    and is unaffected (mirrors the #746 "fresh confirm of an old proposal is
    fresh intent" rule, applied to the interrupt boundary instead of the age
    bound).
    """
    seq: int | None = await conn.fetchval(
        "SELECT seq FROM events "
        "WHERE session_id = $1 AND account_id = $2 AND kind = 'interrupt' "
        "ORDER BY seq DESC LIMIT 1",
        session_id,
        account_id,
    )
    return seq


async def lookup_tool_name_by_call_id(
    conn: asyncpg.Connection[Any],
    session_id: str,
    tool_call_id: str,
    *,
    account_id: str,
) -> tuple[str | None, str | None]:
    """Return ``(name, focal_channel_at_arrival)`` for the parent assistant
    event that requested ``tool_call_id``, or ``(None, None)`` if no parent.

    ``name`` is the function name of the matching ``tool_call`` (used by the
    custom tool-result handler to stamp a ``name`` field so ``_derive_tool_name``
    populates the ``tool_name`` column — issue #133, migration 0022).

    ``focal_channel_at_arrival`` is the SAME value :func:`_lookup_tool_parent_channel`
    resolves — projected here in the SAME row (identical WHERE / ORDER BY /
    LIMIT, same ``events_assistant_tool_calls_idx`` partial index) so the
    ``append_tool_result`` path can pass it as ``tool_parent_channel`` and skip
    the second byte-identical ``@>`` scan (#991): one scan per append, not two.
    """
    row = await conn.fetchrow(
        "SELECT data->'tool_calls' AS tool_calls, focal_channel_at_arrival FROM events "
        "WHERE session_id = $1 "
        "  AND account_id = $3 "
        "  AND kind = 'message' "
        "  AND data->>'role' = 'assistant' "
        "  AND data ? 'tool_calls' "
        "  AND data->'tool_calls' @> jsonb_build_array("
        "    jsonb_build_object('id', $2::text)) "
        "ORDER BY seq DESC LIMIT 1",
        session_id,
        tool_call_id,
        account_id,
    )
    if row is None:
        return None, None
    focal: str | None = row["focal_channel_at_arrival"]
    tool_calls = row["tool_calls"]
    if not isinstance(tool_calls, list):
        return None, focal
    for tc in tool_calls:
        if not isinstance(tc, dict) or tc.get("id") != tool_call_id:
            continue
        function = tc.get("function")
        if not isinstance(function, dict):
            return None, focal
        name = function.get("name")
        return (name if isinstance(name, str) else None), focal
    return None, focal


# A century of seconds — the value callers bind for an effectively unbounded
# confirm-event age when they want every confirmed-unresolved row regardless
# of age. ``confirmed_unresolved_predicate``'s age clause is now a plain
# sargable range (``created_at >= now() - make_interval(...)``) with no
# ``IS NULL`` OR-arm (that form defeated index pruning under a generic plan —
# #1740); any real row's ``created_at`` is always within 100 years of
# ``now()``, so binding this constant is indistinguishable from "unbounded"
# for every actual row while keeping the clause a single sargable comparison
# the planner can push into ``events_tool_confirmed_allow_recent_idx``
# (migration 0134).
_AGE_UNBOUNDED_SECONDS: Final = 3_155_760_000  # 100 years, in seconds


def confirmed_unresolved_predicate(alias: str, age_param: str) -> str:
    """SQL boolean fragment selecting a *confirmed-but-unresolved* dispatch.

    One source for the confirmed-dispatch boolean, consumed by BOTH the sweep's
    cross-session wake detector (``sweep.CONFIRMED_ROWS_SQL``, which projects
    ``DISTINCT session_id``) and the per-session dispatch resolver
    :func:`list_confirmed_unresolved_tool_calls` (which projects the actual
    ``tool_call`` dicts). The two queries differ only in their SELECT/JOIN; the
    WHERE sub-predicate on the ``tool_confirmed`` lifecycle row is THIS shared
    boolean, so detection and dispatch resolve the identical condition by
    construction — no wake-with-no-progress (#155 symptom).

    ``alias`` binds the ``tool_confirmed`` lifecycle row (``lc``). ``age_param``
    is the caller's SQL placeholder for the confirm-event age bound (a
    ``bigint`` seconds value): ``$N`` positional for the resolver, a
    ``{...}``-substituted ``$N`` for the sweep's ``.format``-d text. The clause
    is a plain sargable range comparison — ``created_at >= now() -
    make_interval(secs => age_param)`` — with NO ``IS NULL`` OR-arm, so it is
    prunable at ``events_tool_confirmed_allow_recent_idx`` (migration 0134)
    under a generic plan. A caller that wants an effectively unbounded read
    binds :data:`_AGE_UNBOUNDED_SECONDS` rather than ``NULL`` (see
    :func:`list_confirmed_unresolved_tool_calls`). The bound is keyed on
    ``lc.created_at`` (the CONFIRM event), NOT the assistant turn: a fresh
    confirm of an old proposal is a fresh intent to dispatch (#746).

    The ``NOT EXISTS`` unresolved guard is tenant-scoped
    (``tr.account_id = {alias}.account_id``) — the resolver's correct form;
    the pre-unification sweep copy omitted it (benign, masked by the outer
    ``scope_clause``, but exactly the silent drift two hand-kept copies accrue).
    """
    return (
        f"{alias}.kind = 'lifecycle'\n"
        f"       AND {alias}.data->>'event' = 'tool_confirmed'\n"
        f"       AND {alias}.data->>'result' = 'allow'\n"
        f"       AND {alias}.created_at >= now() - make_interval(secs => {age_param}::bigint)\n"
        f"       AND NOT EXISTS (\n"
        f"           SELECT 1 FROM events tr\n"
        f"            WHERE tr.session_id = {alias}.session_id\n"
        f"              AND tr.account_id = {alias}.account_id\n"
        f"              AND tr.kind = 'message'\n"
        f"              AND tr.role = 'tool'\n"
        f"              AND tr.data->>'tool_call_id' = {alias}.data->>'tool_call_id'\n"
        f"       )"
    )


async def list_confirmed_unresolved_tool_calls(
    conn: asyncpg.Connection[Any],
    session_id: str,
    *,
    account_id: str,
    max_age_seconds: int | None = None,
) -> list[dict[str, Any]]:
    """Return the dispatchable ``tool_call`` dicts for a session: those
    operator-confirmed (``tool_confirmed``/``allow``) whose ``tool_call_id``
    has no paired ``tool_result`` yet, in chronological (parent-assistant
    ``seq``) order.

    This is the dispatch-side resolver of the SAME predicate the sweep uses to
    wake the session for case (c) — ``sweep.CONFIRMED_ROWS_SQL``: a
    ``tool_confirmed``/``allow`` lifecycle event whose ``tool_call_id`` has no
    ``role='tool'`` result, AND (when bounded) whose confirmation is within
    ``max_age_seconds``.  Detection (the sweep, cross-session, projecting
    only ``session_id``) and dispatch (here, per-session, projecting the
    ``tool_call`` dicts) agree BY CONSTRUCTION: both compose the WHERE
    sub-predicate on ``lc`` from the single source
    :func:`confirmed_unresolved_predicate` — they cannot drift.  Re-resolving
    per step is load-bearing against the wake→step TOCTOU window.

    ``max_age_seconds`` is an OPTIONAL age bound on the ``tool_confirmed``
    lifecycle event's (``lc``) ``created_at`` — when set, calls whose
    CONFIRMATION is older than that many seconds are SKIPPED (excluded from
    dispatch, not expired; no synthetic result).  It is keyed on the CONFIRM
    event, NOT the assistant turn: an operator can confirm an OLD proposal,
    which is a FRESH intent to dispatch (#746).  It defaults to ``None`` (no
    bound) for safety/testability; when ``None`` we bind
    :data:`_AGE_UNBOUNDED_SECONDS` (a 100-year constant, indistinguishable from
    "unbounded" for any real row) rather than ``NULL`` — the predicate's age
    clause is a plain sargable range with no ``IS NULL`` OR-arm (#1740), so
    there is no SQL-level "unbounded" sentinel value anymore.  This path is
    dispatch-only (the sole caller is ``_dispatch_confirmed_tools`` via
    ``sessions.py``, no read-model caller), so the dispatch caller always
    passes ``settings.confirmed_dispatch_max_age_seconds`` (``ge=60``, never
    ``None``).  Parallel to the connector backfill bound in
    :func:`_unresolved_tool_calls` (#744).

    Unwindowed otherwise (i.e. when the effective bound is
    :data:`_AGE_UNBOUNDED_SECONDS`) — keyed on ``tool_call_id`` via the
    ``events_tool_confirmed_allow_idx`` partial index (migration 0065), so a
    confirmed tool whose parent assistant has scrolled out of the token window,
    or simply isn't the latest assistant, is still recovered (#737).  The
    ``NOT EXISTS`` result guard means one whose result has itself scrolled out
    is not re-dispatched (no duplicate ``tool_result``; CLAUDE.md invariant
    #4).  The parent-assistant join reuses ``events_assistant_tool_calls_idx``;
    the result check reuses ``events_tool_result_idx``.
    """
    rows = await conn.fetch(
        f"""
        SELECT a.seq AS asst_seq,
               lc.data->>'tool_call_id' AS tool_call_id,
               a.data->'tool_calls' AS tool_calls
          FROM events lc
          JOIN events a
            ON a.session_id = lc.session_id
           AND a.account_id = lc.account_id
           AND a.kind = 'message'
           AND a.role = 'assistant'
           AND a.data ? 'tool_calls'
           AND a.data->'tool_calls' @> jsonb_build_array(
                 jsonb_build_object('id', lc.data->>'tool_call_id'))
         WHERE lc.session_id = $1
           AND lc.account_id = $2
           AND {confirmed_unresolved_predicate("lc", "$3")}
         ORDER BY a.seq ASC
        """,
        session_id,
        account_id,
        max_age_seconds if max_age_seconds is not None else _AGE_UNBOUNDED_SECONDS,
    )
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        tool_call_id: str = row["tool_call_id"]
        if tool_call_id in seen:
            continue
        tool_calls = row["tool_calls"]
        if not isinstance(tool_calls, list):
            continue
        for tc in tool_calls:
            if isinstance(tc, dict) and tc.get("id") == tool_call_id:
                seen.add(tool_call_id)
                out.append(tc)
                break
    return out


async def find_tool_confirmed_seqs(
    conn: asyncpg.Connection[Any],
    session_id: str,
    tool_call_ids: list[str],
    *,
    account_id: str,
) -> dict[str, int]:
    """Return ``{tool_call_id: seq}`` for the ``tool_confirmed``/``allow``
    lifecycle event of each id in *tool_call_ids*.

    Companion lookup for :func:`list_confirmed_unresolved_tool_calls`'s
    result set -- the resolver drops the confirm event's own ``seq`` when it
    projects the ``tool_call`` dict, so the interrupt-vs-confirm ordering
    check in :func:`aios.harness.loop._dispatch_confirmed_tools` (#1756)
    re-fetches it here, scoped to only the ids already known dispatchable.
    Empty *tool_call_ids* short-circuits to ``{}`` with no query. Reuses
    ``events_tool_confirmed_allow_idx`` (migration 0065).
    """
    if not tool_call_ids:
        return {}
    rows = await conn.fetch(
        """
        SELECT data->>'tool_call_id' AS tool_call_id, seq
          FROM events
         WHERE session_id = $1
           AND account_id = $2
           AND kind = 'lifecycle'
           AND data->>'event' = 'tool_confirmed'
           AND data->>'result' = 'allow'
           AND data->>'tool_call_id' = ANY($3::text[])
         ORDER BY seq ASC
        """,
        session_id,
        account_id,
        tool_call_ids,
    )
    # ORDER BY seq ASC + dict overwrite keeps the LATEST confirm seq per id --
    # the correct choice if a call was ever re-confirmed (dict insertion order
    # means the last write for a given key wins).
    out: dict[str, int] = {}
    for row in rows:
        out[row["tool_call_id"]] = row["seq"]
    return out


class _PrecomputedAppend(NamedTuple):
    """The pre-transaction compute result for :func:`append_event` (issue #862,
    #991).

    Carries the two values that must be resolved BEFORE the seq-allocating row
    lock so the LiteLLM tokenizer pass and the tool-parent JSONB ``@>`` scan
    never run under the session lock:

    * ``token_delta`` — the approximate per-event token contribution
      (``_event_token_delta``), 0 for non-message events.
    * ``resolved_tool_channel`` — for tool-role events, the parent assistant's
      ``focal_channel_at_arrival`` (looked up or supplied); ``None`` otherwise.

    Mirrors #986's ``AssistantAppendResult`` precompute-then-pass shape.  The
    two tool-result appenders compute this OUTSIDE their outer ``FOR UPDATE``
    and hand it to :func:`append_event` via ``precomputed=``; every other
    caller leaves ``precomputed=None`` and :func:`append_event` computes it
    inline (byte-identical behavior).
    """

    token_delta: int
    resolved_tool_channel: str | None


async def precompute_event_append(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    session_id: str,
    kind: EventKind,
    data: dict[str, Any],
    orig_channel: str | None = None,
    tool_parent_channel: str | None | EllipsisType = ...,
) -> _PrecomputedAppend:
    """Run :func:`append_event`'s pre-transaction compute and return it.

    The LiteLLM tokenizer pass and the tool-parent JSONB ``@>`` scan are the
    two slowest operations in an append (issue #862).  Resolving them here —
    BEFORE any row lock — keeps concurrent appenders from serializing behind
    the slowest tokenization, and lets the two tool-result appenders (#991)
    run this compute OUTSIDE their outer ``FOR UPDATE`` dedup transaction.

    For tool-role events, ``tool_parent_channel`` either supplies the parent
    channel directly (live builtin/MCP dispatch path) or, left as the default
    ``...`` sentinel, triggers the pre-lock :func:`_lookup_tool_parent_channel`
    scan.  That scan is race-free pre-lock by commit-ordering: the parent
    assistant row is committed before any tool result can arrive (never-delete
    invariant + tool results only arrive for assistant-requested calls), so the
    resolved channel cannot change between this pre-read and the locked INSERT.

    ``sessions`` queries import lazily to avoid a module-load cycle — events.py
    and sessions.py are sibling modules both imported by ``db/queries/__init__.py``.
    """
    from aios.db.queries import sessions as _sessions_q

    delta = 0
    if kind == "message":
        if data.get("role") == "user":
            # USER token count needs the focal channel to render the as-sent
            # form.  This pre-read is OUTSIDE any transaction; a concurrent
            # ``switch_channel`` committing before the lock can make it stale
            # (bounded drift — see ``append_event``'s docstring).  The STORED
            # stamp is always the locked RETURNING value, unaffected by this read.
            pre_focal = await _sessions_q.get_session_focal_channel(
                conn, session_id, account_id=account_id
            )
            delta = _event_token_delta(kind, data, orig_channel, pre_focal)
        else:
            delta = _event_token_delta(kind, data, orig_channel, None)

    # Resolve the tool-parent channel pre-lock too.  The live builtin/MCP
    # dispatch path supplies it directly (default ``...`` → look it up).
    resolved_tool_channel: str | None = None
    if kind == "message" and data.get("role") == "tool":
        resolved_tool_channel = (
            await _lookup_tool_parent_channel(
                conn, session_id, data.get("tool_call_id"), account_id=account_id
            )
            if tool_parent_channel is ...
            else tool_parent_channel
        )

    return _PrecomputedAppend(token_delta=delta, resolved_tool_channel=resolved_tool_channel)


async def count_tool_errors_since_last_user(
    conn: asyncpg.Connection[Any],
    session_id: str,
    tool_name: str,
    error_code: str,
    *,
    account_id: str,
) -> int:
    """Count one tool/error pair in the current direct user turn.

    The append-only event log is the durable counter.  Anchoring at the latest
    direct user message makes a new athlete turn reset the signal without a
    mutable side table, and account scope remains explicit.
    """
    count: int | None = await conn.fetchval(
        "SELECT count(*) FROM events "
        "WHERE session_id = $1 AND account_id = $2 "
        "AND kind = 'message' AND role = 'tool' AND is_error IS TRUE "
        "AND tool_name = $3 AND data->'metadata'->>'mcp_error_code' = $4 "
        "AND seq > COALESCE(("
        "  SELECT max(seq) FROM events "
        "  WHERE session_id = $1 AND account_id = $2 "
        "  AND kind = 'message' AND role = 'user'"
        "), 0)",
        session_id,
        account_id,
        tool_name,
        error_code,
    )
    return count or 0


async def append_event(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    session_id: str,
    kind: EventKind,
    data: dict[str, Any],
    orig_channel: str | None = None,
    tool_parent_channel: str | None | EllipsisType = ...,
    precomputed: _PrecomputedAppend | None = None,
) -> Event:
    """Append an event to ``session_id`` with gapless seq allocation.

    Wraps the seq increment + insert in a single transaction with a row lock
    on the parent session, so concurrent appenders (the API server adding a
    user message while the harness is mid-turn) serialize correctly. Issues
    ``pg_notify`` after the insert so SSE subscribers receive the new event.

    For message events, computes and stores ``cumulative_tokens`` — the
    running total of approximate token counts through this event.  The
    previous cumulative value is fetched inside the same transaction (under
    the session row lock), so the running sum has no race with concurrent
    appenders.  The per-event token DELTA, however, is computed BEFORE the
    lock (issue #862): the LiteLLM tokenizer pass — the slowest part of an
    append — no longer serializes concurrent appenders behind itself.  Only
    the cheap ``_latest_cumulative_tokens`` fetch and the INSERT run under
    the lock.

    Focal-channel stamping (issue #29 redesign): the session's current
    ``focal_channel`` is read from the same UPDATE that allocates the seq
    (via its RETURNING clause) and written to ``focal_channel_at_arrival``
    on the new event row.  Pairing it with the caller-supplied
    ``orig_channel`` (stamped for user events via ``append_user_message``)
    lets the context builder render each event deterministically at arrival
    time without ever needing to re-project past events.

    Derived-channel stamping (issue #52): the new event's ``channel`` column
    is — for user events, ``orig_channel``; for assistant events,
    ``focal_at_arrival``; for tool events, the parent assistant's
    ``focal_channel_at_arrival``.  The tool-parent lookup (a JSONB ``@>``
    scan) is also hoisted out of the transaction (issue #862): the live
    builtin/MCP dispatch path supplies the parent stamp via
    ``tool_parent_channel`` directly (it has the assistant event in hand);
    every other tool-role appender leaves the default ``...`` sentinel and
    the parent is resolved by the pre-transaction
    :func:`_lookup_tool_parent_channel`.

    Drift note (issue #862): a USER message's ``cumulative_tokens`` is
    counted against the focal read BEFORE the lock, so if a ``switch_channel``
    commits between that pre-read and the lock, the token count MAY reflect
    the pre-switch focal — an acceptable, bounded drift in the same class as
    the documented vision/tz drifts below (absorbed by ``model_token_class_ratios``
    calibration).  The STORED ``focal_channel_at_arrival`` is always the
    locked RETURNING value, never the pre-read.
    """
    new_id = make_id(EVENT)
    data_json = json.dumps(data)

    # role/tool_name/is_error/sender_name: indexed-column promotions for
    # events_search (migration 0022); not on the Event model.
    role: str | None = None
    if kind == "message":
        raw_role = data.get("role")
        if isinstance(raw_role, str):
            role = raw_role
    # A user message bumps ``updated_at`` (last-interaction time). It no longer
    # needs to flip a status column: ``status`` is derived from the event log,
    # so an errored session recovers automatically once a user message lands
    # (its seq exceeds the latest error lifecycle event — see
    # ``_SESSION_ERRORED_EXPR``), and the sweep stops skipping it (#39, #353).
    is_user_message = kind == "message" and role == "user"
    # A *stimulus* is any message the assistant must react to: user OR tool
    # (role <> 'assistant'). ``last_stimulus_seq`` tracks its max seq and drives
    # the active predicate. This is deliberately broader than ``is_user_message``
    # (the error latch) — an unreacted tool result keeps the session active, but
    # must NOT clear an error. See ``_SESSION_ACTIVE_EXPR``.
    #
    # Every tool result is a stimulus: the session wakes and the model gets a
    # turn to react to any tool completion (including a ``signal_send`` delivery
    # ack), restoring the standard agentic loop (tool call → result → continue).
    is_stimulus = kind == "message" and role != "assistant"
    # ``is_errored_lifecycle_event`` reads the SAME constant the error latch
    # writes (``harness/loop.py:_latch_errored_turn``). The read is off the JSONB
    # ``data`` (type ``Any``), which cannot bind to the write literal — see the
    # constant's definition (#1084) — so the binding lives in the shared module
    # and is pinned by ``test_errored_lifecycle_coupling.py``.
    is_error_lifecycle = is_errored_lifecycle_event(kind, data)
    is_assistant_message = kind == "message" and role == "assistant"
    tool_call_count_delta = (
        len(data.get("tool_calls") or [])
        if is_assistant_message
        else (-1 if kind == "message" and role == "tool" else 0)
    )
    # The reaction watermark advances to MAX(COALESCE(reacting_to, seq)) over
    # assistant messages — exactly the pre-#732 ``session_max_reacting`` CTE. An
    # assistant message with an explicit ``reacting_to`` uses it; one without
    # (seeded data, or an unprompted assistant turn) falls back to the
    # assistant's OWN new seq (``last_event_seq + 1``). ``turn_ended`` lifecycle
    # events do NOT bump it: a rescheduling ``turn_ended`` appends with no
    # assistant reaction, and bumping the watermark there would falsely mark the
    # still-unreacted user message as reacted-to, flipping a retry-pending
    # session to idle (breaks the litellm/harness-error retry loop — the
    # session must stay active so the sweep re-picks it). Reaction is tracked by
    # assistant ``reacting_to``, never by turn boundaries.
    reacting_to_seq = int(data.get("reacting_to") or 0) if is_assistant_message else 0

    # ── Pre-transaction compute (issue #862, #991) ────────────────────────
    # The LiteLLM tokenizer pass and the tool-parent JSONB lookup are the two
    # slowest operations in an append; they run BEFORE the row lock so
    # concurrent appenders don't serialize behind the slowest tokenization.
    #
    # ``precomputed`` lets a caller resolve that compute OUTSIDE its own outer
    # ``FOR UPDATE`` (the two tool-result appenders, #991) and pass it in — so
    # the tokenizer + cold-path JSONB scan never run under the session lock on
    # the ~100 KB tool-result path that motivated #862.  When ``None`` (the
    # default for every non-tool-result caller), ``append_event`` computes it
    # itself here — byte-identical behavior.
    if precomputed is None:
        precomputed = await precompute_event_append(
            conn,
            account_id=account_id,
            session_id=session_id,
            kind=kind,
            data=data,
            orig_channel=orig_channel,
            tool_parent_channel=tool_parent_channel,
        )
    delta = precomputed.token_delta
    resolved_tool_channel = precomputed.resolved_tool_channel

    # ``chan_candidate`` (issue #1742): the channel address, if any, this
    # append might introduce to the session's maintained ``channels`` set —
    # mirrors ``_resolve_event_channel``'s user/tool arms exactly (the
    # assistant arm is handled inline in the SET clause below via the row's
    # own ``focal_channel``, since that value isn't known until the
    # RETURNING). Non-message / role-less events pass NULL, contributing
    # nothing to the array.
    chan_candidate: str | None = None
    if kind == "message":
        if role == "user":
            chan_candidate = orig_channel
        elif role == "tool":
            chan_candidate = resolved_tool_channel

    async with conn.transaction():
        seq_row = await conn.fetchrow(
            "UPDATE sessions "
            "SET last_event_seq = last_event_seq + 1, "
            "    updated_at = CASE WHEN $3 THEN now() ELSE updated_at END, "
            "    last_user_seq = CASE WHEN $3 THEN last_event_seq + 1 ELSE last_user_seq END, "
            "    last_stimulus_seq = CASE WHEN $8 THEN last_event_seq + 1 "
            "        ELSE last_stimulus_seq END, "
            "    last_error_seq = CASE WHEN $4 THEN last_event_seq + 1 ELSE last_error_seq END, "
            "    open_tool_call_count = GREATEST(open_tool_call_count + $5, 0), "
            "    last_reacted_seq = CASE "
            "        WHEN $7 THEN GREATEST(last_reacted_seq, "
            "            CASE WHEN $6 > 0 THEN $6 ELSE last_event_seq + 1 END) "
            "        ELSE last_reacted_seq END, "
            # ``channels`` (issue #1742): maintain the session's channel set
            # on the SAME row-locked UPDATE that allocates the seq — zero
            # extra round trips, race-free. Mirrors ``_resolve_event_channel``
            # (see cross-reference there): user/tool arms via ``$9``
            # (``chan_candidate``, computed above before the lock); the
            # assistant arm via ``$7 is_assistant_message`` + the row's own
            # ``focal_channel`` (identical to what RETURNING hands the Python
            # stamp below, since this UPDATE never writes ``focal_channel``).
            # ``@>`` membership check avoids appending a duplicate.
            "    channels = CASE "
            "        WHEN $9::text IS NOT NULL AND NOT (channels @> ARRAY[$9::text]) "
            "            THEN channels || $9::text "
            "        WHEN $7 AND focal_channel IS NOT NULL "
            "            AND NOT (channels @> ARRAY[focal_channel]) "
            "            THEN channels || focal_channel "
            "        ELSE channels END "
            "WHERE id = $1 AND account_id = $2 AND archived_at IS NULL "
            "RETURNING last_event_seq, focal_channel",
            session_id,
            account_id,
            is_user_message,
            is_error_lifecycle,
            tool_call_count_delta,
            reacting_to_seq,
            is_assistant_message,
            is_stimulus,
            chan_candidate,
        )
        if seq_row is None:
            # Treat archived as "session no longer exists for write purposes."
            # ``find_sessions_needing_inference`` (harness/sweep.py) already
            # filters ``archived_at IS NULL``, so without this guard a
            # POST to an archived session would return 201 + silently
            # vanish: the row's ``last_event_seq`` increments, the event
            # INSERTs, but the wake-sweep never picks it up. Surfacing as
            # ``NotFoundError`` (→ 404 at the router) gives the caller an
            # honest signal that the post is dropped. Same defect class
            # as PR #521 (archived-connection inbound), one layer deeper.
            raise NotFoundError(f"session {session_id} not found", detail={"id": session_id})
        seq = seq_row["last_event_seq"]
        focal_at_arrival: str | None = seq_row["focal_channel"]

        # cumulative_tokens = prev running sum + the pre-computed per-event
        # delta.  ``prev`` is the ONLY query between the seq-allocating UPDATE
        # and the INSERT (issue #862): the tokenizer pass that produced
        # ``delta`` already ran pre-lock, so concurrent appenders no longer
        # serialize behind it.  The running sum stays race-free because
        # ``prev`` is read under the session row lock.
        #
        # NOTE(vision/tz): the USER ``delta`` was rendered without
        # ``model``/``session_id`` and in the default UTC zone, so inlined
        # images undercount by ~55 LiteLLM tokens each and a non-UTC account's
        # envelope is a few tokens narrower than build time.  Both drifts are
        # bounded and absorbed by ``model_token_class_ratios`` calibration in
        # :func:`read_windowed_events` (see PR #218); exact matching is
        # impossible anyway, since a later tz/vision change re-renders history.
        # cumulative_messages / cumulative_*_mass extend the SAME append-time
        # running-sum machinery (issue #1657): read the prior running state
        # (one index seek on the latest message row), then increment. The
        # per-class mass attributes THIS event's whole token ``delta`` to its
        # dominant content class (``_message_content_class`` mirrors the CASE
        # the old full-slate ``_retained_class_mass`` WindowAgg ran), and
        # ``cumulative_messages`` counts user/assistant messages only — exactly
        # the ``FILTER (role IN ('user','assistant'))`` the omission read used.
        # Every counter is NULL on non-message events (as ``cumulative_tokens``
        # is), so the read path's fallback stays byte-identical for them.
        cum_tokens: int | None = None
        cum_messages: int | None = None
        cum_mass: dict[str, int | None] = {c: None for c in _MESSAGE_CONTENT_CLASSES}
        if kind == "message":
            prev = await _latest_cumulative_state(conn, session_id)
            cum_tokens = (prev.tokens or 0) + delta
            counts_as_message = role in ("user", "assistant")
            cum_messages = (prev.messages or 0) + (1 if counts_as_message else 0)
            cls = _message_content_class(role, data)
            for c in _MESSAGE_CONTENT_CLASSES:
                base = prev.mass.get(c) or 0
                # A negative ``delta`` cannot lower a class below its prior
                # running sum: clamp per-event so the stored cumulative stays
                # monotonic, matching the old query's ``if mass < 0: mass = 0``
                # per-class flooring of the summed deltas.
                add = delta if c == cls else 0
                cum_mass[c] = max(0, base + add)

        channel = _resolve_event_channel(
            kind, data, orig_channel, focal_at_arrival, resolved_tool_channel
        )
        tool_name = _derive_tool_name(kind, data)
        is_error = _derive_is_error(kind, data)
        sender_name = _derive_sender_name(kind, data)

        row = await conn.fetchrow(
            "INSERT INTO events "
            "(id, session_id, seq, kind, data, cumulative_tokens, "
            " cumulative_messages, cumulative_text_mass, "
            " cumulative_tool_result_mass, cumulative_thinking_mass, "
            " cumulative_tool_use_mass, "
            " orig_channel, focal_channel_at_arrival, channel, "
            " role, tool_name, is_error, sender_name, account_id) "
            "VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7, $8, $9, $10, $11, "
            " $12, $13, $14, $15, $16, $17, $18, $19) RETURNING *",
            new_id,
            session_id,
            seq,
            kind,
            data_json,
            cum_tokens,
            cum_messages,
            cum_mass["text"],
            cum_mass["tool_result"],
            cum_mass["thinking"],
            cum_mass["tool_use"],
            orig_channel,
            focal_at_arrival,
            channel,
            role,
            tool_name,
            is_error,
            sender_name,
            account_id,
        )
        assert row is not None

    # NOTIFY happens outside the transaction so subscribers don't see it
    # before the row is committed. Use pg_notify (the function form) rather
    # than the literal NOTIFY statement, because Postgres case-folds unquoted
    # identifiers in NOTIFY <chan> — and our prefixed-ULID session ids
    # contain uppercase letters. asyncpg's add_listener quotes the channel,
    # preserving case, so the two would never match. pg_notify(text, text)
    # treats the channel as a string literal and preserves it byte-for-byte.
    await conn.execute("SELECT pg_notify($1, $2)", f"events_{session_id}", new_id)

    # Connector fan-out: every assistant-with-tool_calls fires
    # ``connector_calls_<type>`` per bound connection. The consumer's
    # backfill filters by ``connector.tools_schema``, so over-fanout
    # (when none of the tool_calls are custom) is harmless and avoids
    # loading agent.tools on the append hot path.
    if (
        kind == "message"
        and role == "assistant"
        and isinstance(data, dict)
        and data.get("tool_calls")
    ):
        for cid, connector in await _list_bound_connection_ids(
            conn, session_id, account_id=account_id
        ):
            await conn.execute(
                "SELECT pg_notify($1, $2)",
                f"connector_calls_{connector}",
                f"{session_id}|{cid}",
            )
    return _row_to_event(row)


async def list_pending_calls_for_connector(
    conn: asyncpg.Connection[Any],
    connector: str,
    *,
    account_id: str,
) -> list[dict[str, Any]]:
    """Pending custom tool calls across every active connection of ``connector`` type.

    Used by the runtime SSE at subscribe-time backfill.  A "pending"
    call is a tool_call on ANY assistant message of a bound session
    whose ``function.name`` is in ``connector.tools_schema`` and has no
    paired tool_result event yet — not just the latest assistant turn, so
    a custom call left pending while the model emitted a later turn is
    still surfaced for execution (the connector-side facet of #741).  No
    dependency on ``stop_reason`` — the source of truth is the event log.

    Each emitted record carries ``connection_id`` so the runtime can
    fan out to the right per-connection worker.

    ``workspace_path`` is the session's host-side bind-mount source for
    ``/workspace`` (the ``workspace_volume_path`` column); the connector
    SDK uses it to resolve ``SandboxPath`` arguments to host paths.

    Output dict shape::

        {
            "session_id": "sess_xxx",
            "tool_call_id": "call_yyy",
            "name": "telegram_send",
            "arguments": "{...}",       # JSON string from the model
            "focal_channel": "telegram/bot1/chat123" | None,
            "connection_id": "conn_zzz",
            "workspace_path": "/var/lib/aios/workspaces/acc_xxx/sess_xxx",
        }
    """
    # The connector type's tool schema gates which tool_calls we surface.
    # ``connectors`` is global per-type; no account scoping on its row.
    cat_row = await conn.fetchrow(
        "SELECT tools_schema AS tools FROM connectors WHERE connector = $1",
        connector,
    )
    if cat_row is None:
        return []
    tools_data = cat_row["tools"]
    name_set = {t["name"] for t in tools_data if isinstance(t, dict) and "name" in t}
    if not name_set:
        return []

    # Find bound sessions of this connector type. Tenant isolation: both
    # ``connections.account_id`` and ``sessions.account_id`` must match the
    # bearer's account, otherwise a runtime token for tenant A could see
    # tool-call arguments from tenants B, C, D under the same connector type.
    bound_rows = await conn.fetch(
        """
        SELECT DISTINCT c.id AS connection_id,
               s.id AS session_id, s.focal_channel,
               s.workspace_volume_path AS workspace_path
          FROM connections c
          JOIN sessions s
            ON s.archived_at IS NULL
           AND s.account_id = $2
           AND (EXISTS (SELECT 1 FROM bindings b
                         WHERE b.connection_id = c.id
                           AND b.archived_at IS NULL
                           AND b.session_id = s.id)
                OR EXISTS (SELECT 1 FROM chat_sessions cs
                            WHERE cs.connection_id = c.id
                              AND cs.session_id = s.id))
         WHERE c.connector = $1
           AND c.archived_at IS NULL
           AND c.account_id = $2
        """,
        connector,
        account_id,
    )
    if not bound_rows:
        return []

    by_session: dict[str, list[tuple[str, str | None]]] = {}
    workspace_path_by_session: dict[str, str] = {}
    for row in bound_rows:
        by_session.setdefault(row["session_id"], []).append(
            (row["connection_id"], row["focal_channel"])
        )
        workspace_path_by_session[row["session_id"]] = row["workspace_path"]

    # Age guard scoped to the transmit/backfill path ONLY (#744): a pending
    # send whose parent assistant turn is older than the threshold is skipped
    # — excluded here, not expired (the event log is left untouched). The
    # sibling read-model (Session.awaiting via _unresolved_tool_calls with no
    # bound) still surfaces stale calls; this only stops the connector from
    # re-transmitting weeks-dormant sends on a reconnect after a worker restart.
    from aios.config import get_settings

    max_age_seconds = get_settings().connector_backfill_max_age_seconds
    raw_by_sid = await _unresolved_tool_calls(
        conn, list(by_session.keys()), account_id=account_id, max_age_seconds=max_age_seconds
    )
    out: list[dict[str, Any]] = []
    for sid, calls in raw_by_sid.items():
        workspace_path = workspace_path_by_session[sid]
        for conn_id, focal in by_session[sid]:
            for tc in calls:
                fn = tc.get("function") or {}
                name = fn.get("name")
                if name not in name_set:
                    continue
                out.append(
                    {
                        "session_id": sid,
                        "tool_call_id": tc["id"],
                        "name": name,
                        "arguments": fn.get("arguments", "{}"),
                        "connection_id": conn_id,
                        "focal_channel": focal,
                        "workspace_path": workspace_path,
                    }
                )
    return out


async def list_pending_calls_for_session_and_connection(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    session_id: str,
    connection_id: str,
) -> list[dict[str, Any]]:
    """Same shape as :func:`list_pending_calls_for_connector` but scoped
    to one session.  Used by the SSE NOTIFY tail to fetch calls only for
    the session that just emitted, instead of re-scanning all bound
    sessions.

    Age-bounded identically to the subscribe-time backfill (#744): the
    NOTIFY tail is a second transmit path into ``runtime_connector_calls_stream``,
    so it passes the same ``settings.connector_backfill_max_age_seconds``
    ceiling to ``_unresolved_tool_calls``.  Without it the tail would
    re-transmit a weeks-stale dormant connector send the instant its
    session emits any new event (firing the per-session NOTIFY) — the
    ``emitted`` dedup in the stream only suppresses calls the backfill
    already yielded, and the backfill now SKIPS stale calls, so they are
    absent from ``emitted`` and would slip through here unbounded.  Both
    emit paths must be bounded by the same setting; neither transmits a
    connector send older than the threshold.  Like the backfill this is
    skip-not-expire (the event log is untouched) and does NOT touch
    ``Session.awaiting`` (the read-model sibling surfaces all unresolved
    calls regardless of age, #741).
    """
    conn_row = await conn.fetchrow(
        f"""
        SELECT cat.tools_schema AS tools, s.focal_channel,
               s.workspace_volume_path AS workspace_path
          FROM connections c
          JOIN connectors cat ON cat.connector = c.connector
          JOIN sessions s
            ON s.id = $3 AND s.archived_at IS NULL AND s.account_id = $2
         WHERE c.id = $1 AND c.archived_at IS NULL AND c.account_id = $2
           AND {
            _session_bound_to_connection_predicate(
                connection_alias="c", session_param_index=3, account_id_param_index=2
            )
        }
        """,
        connection_id,
        account_id,
        session_id,
    )
    if conn_row is None:
        return []
    tools_data = conn_row["tools"]
    name_set = {t["name"] for t in tools_data if isinstance(t, dict) and "name" in t}
    if not name_set:
        return []

    # Same age guard as the subscribe-time backfill (#744): the NOTIFY tail
    # is the second transmit path, so it must bound by the same setting or a
    # stale send re-transmits the moment its session emits a new event.
    from aios.config import get_settings

    max_age_seconds = get_settings().connector_backfill_max_age_seconds
    raw_by_sid = await _unresolved_tool_calls(
        conn, [session_id], account_id=account_id, max_age_seconds=max_age_seconds
    )
    focal = conn_row["focal_channel"]
    workspace_path = conn_row["workspace_path"]
    out: list[dict[str, Any]] = []
    for tc in raw_by_sid.get(session_id, []):
        fn = tc.get("function") or {}
        name = fn.get("name")
        if name not in name_set:
            continue
        out.append(
            {
                "session_id": session_id,
                "tool_call_id": tc["id"],
                "name": name,
                "arguments": fn.get("arguments", "{}"),
                "connection_id": connection_id,
                "focal_channel": focal,
                "workspace_path": workspace_path,
            }
        )
    return out


async def _unresolved_tool_calls(
    conn: asyncpg.Connection[Any],
    session_ids: list[str],
    *,
    account_id: str,
    max_age_seconds: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Return ``{session_id: [tool_call_dict]}`` for EVERY assistant's
    tool_calls (per session) that have no paired tool_result event, in
    chronological (seq-ascending) order.

    Spans all assistant turns, not just the latest: an ``always_ask``
    tool_call on an earlier assistant can stay unresolved while a later
    assistant emits other tool_calls (e.g. the model reacts to an impatient
    user message before the operator confirms).  Restricting to the latest
    assistant (a ``DISTINCT ON (session_id) ... ORDER BY seq DESC``) hid such
    still-pending calls from ``Session.awaiting`` — the read-model sibling of
    the dispatch-side window-edge bug #737 (#741).

    Pending-ness is purely an event-log property — the session row's
    ``status`` and ``stop_reason`` are irrelevant. Tool_call dicts are
    returned as-is from the assistant's ``data->'tool_calls'`` array.

    ``max_age_seconds`` is an OPTIONAL age bound on the parent assistant
    turn's ``created_at`` — when set, tool_calls whose assistant event is
    older than that many seconds are excluded.  It defaults to ``None``
    (no age filter) so the ``Session.awaiting`` / unresolved-read-model
    callers keep surfacing ALL unresolved calls regardless of age (#741).
    BOTH connector-SSE transmit paths pass a bound (#744): the
    subscribe-time backfill (:func:`list_pending_calls_for_connector`)
    AND the NOTIFY tail (:func:`list_pending_calls_for_session_and_connection`),
    so neither re-transmits a weeks-dormant connector send — on reconnect
    (backfill) or on session re-activation (tail).
    """
    if not session_ids:
        return {}
    # ``data ? 'tool_calls'`` is the partial-index predicate on
    # ``events_assistant_tool_calls_idx``; the ``jsonb_array_length > 0``
    # post-filter narrows to non-empty arrays (the index admits
    # ``null`` / ``[]`` too).  Without the ``?`` conjunct the planner
    # falls back to the wider btree backing the ``events``
    # ``UNIQUE (session_id, seq)`` constraint.
    #
    # ``$3`` carries the optional age bound (seconds); NULL disables it so
    # the awaiting read-model path is byte-for-byte unchanged (#741), while
    # the connector backfill (#744) passes a positive value to drop stale
    # sends.  ``make_interval`` keeps the bound parameterized rather than
    # string-interpolated into the SQL.
    asst_rows = await conn.fetch(
        """
        SELECT session_id, data, created_at
          FROM events
         WHERE session_id = ANY($1::text[])
           AND account_id = $2
           AND kind = 'message'
           AND role = 'assistant'
           AND data ? 'tool_calls'
           AND jsonb_array_length(
                 COALESCE(NULLIF(data->'tool_calls','null'::jsonb), '[]'::jsonb)
               ) > 0
           AND (
                 $3::bigint IS NULL
                 OR created_at >= now() - make_interval(secs => $3::bigint)
               )
         ORDER BY session_id, seq ASC
        """,
        session_ids,
        account_id,
        max_age_seconds,
    )
    if not asst_rows:
        return {}
    results_by_sid = await _tool_result_ids_by_session(conn, session_ids, account_id=account_id)
    out: dict[str, list[dict[str, Any]]] = {}
    for row in asst_rows:
        sid: str = row["session_id"]
        data = row["data"]
        completed: set[str] = results_by_sid.get(sid, set())
        for tc in data.get("tool_calls") or []:
            if tc.get("id") and tc["id"] not in completed:
                # Shallow copy so the read-model carries the parent
                # assistant turn's created_at without mutating the parsed
                # source dict (the jsonb codec may return a shared reference).
                # Connector-SSE consumers
                # build their own explicit output dicts, so this extra key
                # never leaks into their payloads.
                out.setdefault(sid, []).append({**tc, "_pending_since": row["created_at"]})
    return out


async def _tool_result_ids_by_session(
    conn: asyncpg.Connection[Any],
    session_ids: list[str],
    *,
    account_id: str,
) -> dict[str, set[str]]:
    """Map ``session_id → {tool_call_id}`` for every tool-role event."""
    rows = await conn.fetch(
        """
        SELECT session_id, data->>'tool_call_id' AS tool_call_id
          FROM events
         WHERE session_id = ANY($1::text[])
           AND account_id = $2
           AND kind = 'message'
           AND role = 'tool'
        """,
        session_ids,
        account_id,
    )
    out: dict[str, set[str]] = {}
    for r in rows:
        tcid = r["tool_call_id"]
        if tcid:
            out.setdefault(r["session_id"], set()).add(tcid)
    return out


async def list_unresolved_tool_calls_batch(
    conn: asyncpg.Connection[Any],
    session_ids: list[str],
    *,
    account_id: str,
) -> dict[str, list[dict[str, Any]]]:
    """For each session, return every assistant's tool_calls that have no
    paired tool_result, annotated with allow-lifecycle presence.

    Spans all assistant turns, not just the latest, so a tool_call left
    unresolved on an earlier turn still appears in ``Session.awaiting``
    (#741).  Used by :func:`services.sessions.compute_awaiting` to build the
    ``Session.awaiting`` derived view. Returned dicts have keys
    ``tool_call_id``, ``name``, ``arguments``, ``has_allow_lifecycle``,
    ``pending_since`` (the parent assistant event's ``created_at``)
    — the caller classifies kind / needs_confirm using ``agent`` (and
    the tool's ``classify_permission`` for arg-aware routes like
    ``http_request``).
    """
    raw = await _unresolved_tool_calls(conn, session_ids, account_id=account_id)
    if not raw:
        return {}
    allow_rows = await conn.fetch(
        """
        SELECT session_id, data->>'tool_call_id' AS tool_call_id
          FROM events
         WHERE session_id = ANY($1::text[])
           AND account_id = $2
           AND kind = 'lifecycle'
           AND data->>'event' = 'tool_confirmed'
           AND data->>'result' = 'allow'
        """,
        session_ids,
        account_id,
    )
    allows_by_sid: dict[str, set[str]] = {}
    for r in allow_rows:
        tcid = r["tool_call_id"]
        if tcid:
            allows_by_sid.setdefault(r["session_id"], set()).add(tcid)

    out: dict[str, list[dict[str, Any]]] = {}
    for sid, calls in raw.items():
        allows = allows_by_sid.get(sid, set())
        entries: list[dict[str, Any]] = []
        for tc in calls:
            fn = tc.get("function") or {}
            name = fn.get("name")
            if not name:
                continue
            entries.append(
                {
                    "tool_call_id": tc["id"],
                    "name": name,
                    "arguments": fn.get("arguments", "{}"),
                    "has_allow_lifecycle": tc["id"] in allows,
                    "pending_since": tc["_pending_since"],
                }
            )
        if entries:
            out[sid] = entries
    return out


async def _list_bound_connection_ids(
    conn: asyncpg.Connection[Any], session_id: str, *, account_id: str
) -> list[tuple[str, str]]:
    """``(connection_id, connector)`` pairs for active connections bound to ``session_id``.

    Called from :func:`append_event` when an assistant message with
    tool_calls lands, to fan a per-connection notification on the
    ``connector_calls_<connector>`` channel.  Tools-less connections
    receive notifications and harmlessly no-op them on the consumer side.
    """
    rows = await conn.fetch(
        f"""
        SELECT c.id, c.connector
          FROM connections c
         WHERE c.archived_at IS NULL
           AND c.account_id = $2
           AND {
            _session_bound_to_connection_predicate(
                connection_alias="c", session_param_index=1, account_id_param_index=2
            )
        }
        """,
        session_id,
        account_id,
    )
    return [(row["id"], row["connector"]) for row in rows]


async def is_session_bound_to_connection(
    conn: asyncpg.Connection[Any], *, account_id: str, connection_id: str, session_id: str
) -> bool:
    """True iff ``connection_id`` is bound to ``session_id`` via either
    of the two lineage paths:

    * Active single_session binding on this connection whose
      ``bindings.session_id`` matches.
    * Row in ``chat_sessions`` for this ``(connection_id, session_id)``.

    Centralised so route handlers don't inline the union of branches
    every time they need to authorise a connector-driven write.
    """
    row = await conn.fetchval(
        f"""
        SELECT 1
          FROM connections c
         WHERE c.id = $1
           AND c.archived_at IS NULL
           AND c.account_id = $3
           AND {
            _session_bound_to_connection_predicate(
                connection_alias="c", session_param_index=2, account_id_param_index=3
            )
        }
         LIMIT 1
        """,
        connection_id,
        session_id,
        account_id,
    )
    return row is not None


async def read_events(
    conn: asyncpg.Connection[Any],
    session_id: str,
    *,
    account_id: str,
    after_seq: int = 0,
    before: int | None = None,
    kind: EventKind | None = None,
    channels: list[str] | None = None,
    chat_type: ChatType | None = None,
    limit: int = 200,
    newest_first: bool = False,
    error_only: bool = False,
) -> list[Event]:
    # Bounds define the requested window; ``newest_first`` alone defines its
    # traversal direction.  A forward bounded window (``after_seq`` + ``before``)
    # must remain chronological rather than becoming backward merely because it
    # has an upper bound.
    order = "DESC" if newest_first else "ASC"

    def _build_query(lower_bound: int, upper_bound: int | None) -> tuple[str, list[Any]]:
        params: list[Any] = [session_id, account_id]
        where = "session_id = $1 AND account_id = $2"
        if lower_bound:
            params.append(lower_bound)
            where += f" AND seq > ${len(params)}"
        if upper_bound is not None:
            params.append(upper_bound)
            where += f" AND seq < ${len(params)}"
        if kind is not None:
            params.append(kind)
            where += f" AND kind = ${len(params)}"
        if channels:
            # #1613: index-backed channel filter against the partial index
            # ``events_session_channel_seq_idx ON events(session_id, channel,
            # seq) WHERE channel IS NOT NULL`` (migration 0022). NULL-channel
            # rows (lifecycle/span/switch_channel) are excluded by design: a
            # channel-scoped LIST is message-row-only.
            #
            # Single channel (the relay/cockpit hot path) emits a scalar
            # ``channel = $n`` so the planner can walk the index in
            # ``(session_id, channel, seq)`` order and satisfy ``ORDER BY seq``
            # directly — an ordered Index Scan, no Sort. ``channel = ANY($n)``
            # over a set cannot yield globally seq-ordered output from that
            # index, so multi-channel keeps the OR form (still index-backed,
            # with a Sort/merge on top).
            params.append(channels[0] if len(channels) == 1 else channels)
            if len(channels) == 1:
                where += f" AND channel = ${len(params)}"
            else:
                where += f" AND channel = ANY(${len(params)})"
        if error_only:
            where += " AND is_error IS TRUE"
        params.append(limit)
        return (
            f"SELECT * FROM events WHERE {where} ORDER BY seq {order} LIMIT ${len(params)}",
            params,
        )

    if chat_type is None:
        sql, params = _build_query(after_seq, before)
        rows = await conn.fetch(sql, *params)
    else:
        # #1613: ``chat_type`` is a pure function of the channel address
        # (``chat_type_of``), not a column — so it is applied as a post-filter.
        # To preserve LIMIT semantics across that filter, page forward (or
        # backward) by ``seq`` until ``limit`` matching rows are collected or
        # the log is exhausted. Each batch still rides the same index-backed
        # WHERE; this only adds extra batches when most rows are filtered out.
        collected: list[asyncpg.Record] = []
        lo, hi = after_seq, before
        while True:
            sql, params = _build_query(lo, hi)
            batch = await conn.fetch(sql, *params)
            if not batch:
                break
            for r in batch:
                if chat_type_of(r["channel"]) == chat_type:
                    collected.append(r)
                    if len(collected) >= limit:
                        break
            if len(collected) >= limit or len(batch) < limit:
                break
            # Advance the cursor past this batch's last seq (forward) or before
            # its last seq (backward) to fetch the next page.
            last_seq = int(batch[-1]["seq"])
            if order == "ASC":
                lo = last_seq
            else:
                hi = last_seq
        rows = collected
    return [_row_to_event(r) for r in rows]


async def get_event(
    conn: asyncpg.Connection[Any], session_id: str, event_id: str, *, account_id: str
) -> Event:
    row = await conn.fetchrow(
        "SELECT * FROM events WHERE id = $1 AND session_id = $2 AND account_id = $3",
        event_id,
        session_id,
        account_id,
    )
    if row is None:
        raise NotFoundError(f"event {event_id} not found", detail={"id": event_id})
    return _row_to_event(row)


async def replace_event_data(
    conn: asyncpg.Connection[Any],
    session_id: str,
    event_id: str,
    data: dict[str, Any],
    *,
    account_id: str,
) -> bool:
    """Overwrite ``events.data`` for a single row (issue #1745 Part C).

    The ONLY in-place event mutation in the codebase — every other write
    path is an append. It exists for exactly one caller,
    :func:`aios.harness.context_persist.persist_clamped_image_parts`: a
    deterministic, idempotent, account-scoped self-heal that shrinks an
    oversize persisted image part once so the render-time clamp pass
    (:func:`aios.harness.context._clamp_oversize_image_data_urls`) doesn't
    have to re-decode + re-downsample it on every future build. ``seq``,
    ``created_at``, and ``kind`` are never touched, so rendered-context
    monotonicity holds — only the JSONB payload changes, and only to bytes
    that are byte-equal to what the in-memory clamp pass already produces.

    Account-scoping mirrors :func:`get_event`. Returns whether one row was
    updated; a mismatched id/session/account returns ``False``.
    """
    status = await conn.execute(
        "UPDATE events SET data = $1::jsonb WHERE id = $2 AND session_id = $3 AND account_id = $4",
        json.dumps(data),
        event_id,
        session_id,
        account_id,
    )
    return bool(status == "UPDATE 1")


async def get_session_event_stats(
    conn: asyncpg.Connection[Any], session_id: str, *, account_id: str
) -> tuple[int, datetime | None]:
    row = await conn.fetchrow(
        "SELECT COUNT(*) AS total, MAX(created_at) AS last_at FROM events "
        "WHERE session_id = $1 AND account_id = $2",
        session_id,
        account_id,
    )
    assert row is not None  # COUNT(*) always returns a row
    return int(row["total"]), row["last_at"]


async def read_message_events(
    conn: asyncpg.Connection[Any], session_id: str, *, account_id: str
) -> list[Event]:
    """Read every message-kind event for a session in chronological order.

    Used by callers that need the full unwindowed log (e.g.
    ``confirm_tool_deny`` searching for a tool_call_id).
    """
    rows = await conn.fetch(
        "SELECT * FROM events WHERE session_id = $1 AND account_id = $2 "
        "AND kind = 'message' ORDER BY seq ASC",
        session_id,
        account_id,
    )
    return [_row_to_event(r) for r in rows]


async def list_session_channels(
    conn: asyncpg.Connection[Any], session_id: str, *, account_id: str
) -> list[str]:
    """Channel addresses the session has interacted with, sorted.

    Issue #1742: this used to run ``SELECT DISTINCT channel FROM events``
    (O(session-size)) on EVERY step inside the harness's pre-inference
    gather. It now reads the ``channels`` array maintained on the
    ``sessions`` row itself — a primary-key point read, no ``events``
    scan — by :func:`append_event`'s seq-allocating UPDATE (mirrors
    :func:`_resolve_event_channel`; see the cross-reference there).
    Array ``||`` append order is insertion order, not sorted order, so
    this sorts before returning — preserving the old ``ORDER BY channel``
    contract byte-for-byte.
    """
    row = await conn.fetchrow(
        "SELECT channels FROM sessions WHERE id = $1 AND account_id = $2",
        session_id,
        account_id,
    )
    if row is None:
        raise NotFoundError(f"session {session_id} not found", detail={"id": session_id})
    return sorted(row["channels"] or [])


async def recompute_session_channels(
    conn: asyncpg.Connection[Any], session_id: str, *, account_id: str
) -> list[str]:
    """Recompute the channel set straight from the event log (fallback).

    This is the OLD ``list_session_channels`` DISTINCT-scan query, kept
    as the ground-truth recomputation for the repair path (issue #1742):
    a rolling-deploy window where an old (pre-migration-code) container
    appends a new channel after the ``channels`` column has already been
    backfilled leaves the stored array stale relative to the event log.
    Callers use this ONLY at cold hard-reject sites (``switch_channel``'s
    membership check, the POST ``metadata.channel`` bound-check) — never
    in the hot per-step loop path, which stays an O(1) ``sessions`` read
    via :func:`list_session_channels`.
    """
    rows = await conn.fetch(
        """
        SELECT DISTINCT channel
          FROM events
         WHERE session_id = $1
           AND account_id = $2
           AND kind = 'message'
           AND channel IS NOT NULL
         ORDER BY channel
        """,
        session_id,
        account_id,
    )
    return [str(r["channel"]) for r in rows]


async def read_windowed_context_events(
    conn: asyncpg.Connection[Any],
    session_id: str,
    *,
    account_id: str,
    drop: int | None = None,
) -> list[Event]:
    """Events the context builder needs, in seq order: message events plus
    the model-visible FS-loss notices (``kind='lifecycle'`` whose ``event``
    is in :data:`MODEL_VISIBLE_LIFECYCLE_EVENTS`).

    ``drop=None`` loads the full log. ``drop=N`` keeps messages with
    ``cumulative_tokens > N`` plus notices past the dropped-message prefix
    (``seq`` greater than the max seq among dropped messages). The notices
    carry NULL ``cumulative_tokens``, so they window out by *seq* alongside
    their surrounding messages, not by the token boundary — a notice scrolls
    out of context exactly when the messages around its reset point do.

    ``read_message_events`` stays message-only (its other callers — e.g.
    ``confirm_tool_deny`` — must not see lifecycle rows); this is the
    windowing-specific read that feeds :func:`build_messages`.
    """
    allowlist = list(MODEL_VISIBLE_LIFECYCLE_EVENTS)
    # UNION ALL (not an OR across kinds) so each arm keeps its own index plan:
    # the message arm stays a clean ``cumulative_tokens`` partial-index range
    # scan. An ``OR`` spanning both kinds would defeat that index on every
    # windowed wake, even for the common session with no FS-loss notices. The
    # arms are disjoint by ``kind``, so ALL (no dedup) is correct and cheaper.
    if drop is None:
        rows = await conn.fetch(
            "SELECT * FROM events "
            "WHERE session_id = $1 AND account_id = $2 AND kind = 'message' "
            "UNION ALL "
            "SELECT * FROM events "
            "WHERE session_id = $1 AND account_id = $2 "
            "AND kind = 'lifecycle' AND data->>'event' = ANY($3) "
            "ORDER BY seq ASC",
            session_id,
            account_id,
            allowlist,
        )
    else:
        # Notices are seq-bounded, not token-bounded: include those past the
        # last dropped message (COALESCE handles "nothing dropped" → seq > 0).
        rows = await conn.fetch(
            "SELECT * FROM events "
            "WHERE session_id = $1 AND account_id = $3 "
            "AND kind = 'message' AND cumulative_tokens > $2 "
            "UNION ALL "
            "SELECT * FROM events "
            "WHERE session_id = $1 AND account_id = $3 "
            "AND kind = 'lifecycle' AND data->>'event' = ANY($4) "
            "AND seq > COALESCE("
            "    (SELECT max(seq) FROM events "
            "     WHERE session_id = $1 AND account_id = $3 "
            "     AND kind = 'message' AND cumulative_tokens <= $2), 0) "
            "ORDER BY seq ASC",
            session_id,
            drop,
            account_id,
            allowlist,
        )
    return [_row_to_event(r) for r in rows]


class _NormalizedOverhead(NamedTuple):
    system: int
    tools: int
    reserves: int

    @property
    def total(self) -> int:
        return self.system + self.tools + self.reserves


def _normalize_overhead(overhead_local: int | OverheadLocalSplit) -> _NormalizedOverhead:
    """Coerce the windower's ``overhead_local`` argument to a class split.

    Accepts both the new :class:`OverheadLocalSplit` (system/tools/reserves)
    and the legacy bare ``int`` (preview tooling / test scaffolds that pass
    a single scalar, e.g. ``0``). A bare int is treated as undifferentiated
    ``text``-class overhead via the ``reserves`` slot, so the blend weights
    it neutrally — preserving the legacy contract for those callers.
    """
    if isinstance(overhead_local, int):
        return _NormalizedOverhead(system=0, tools=0, reserves=overhead_local)
    return _NormalizedOverhead(
        system=int(overhead_local.system),
        tools=int(overhead_local.tools),
        reserves=int(overhead_local.reserves),
    )


async def _retained_class_mass(
    conn: asyncpg.Connection[Any],
    session_id: str,
    *,
    account_id: str,
) -> dict[str, float]:
    """Per-content-class neutral token mass of the session's message slate.

    Pooled over the WHOLE session's message slate (not just the retained
    tail): the composition is what selects ``R_eff``, and the full-slate
    mix is a stable, prefix-cache-friendly estimator of it -- it does not
    jump as the drop boundary advances within a snap chunk.

    O(1) read (issue #1657). The four per-class running sums are maintained
    at append time (``append_event`` increments ``cumulative_*_mass`` from
    the prior message's totals, classifying by role+data with
    ``_message_content_class``). Because the mass pools over the whole slate,
    the answer is just the LATEST message row's four cumulative totals -- one
    index seek on ``events_session_cumtokens_idx`` (``ORDER BY seq DESC``),
    replacing the O(session-size) full-slate ``LAG() OVER (ORDER BY seq)``
    WindowAgg (measured 3.8s / 90k rows / 46 MB spill on Ultron) entirely.

    Returns an empty dict when the latest message carries no per-class
    cumulative data (pre-backfill sessions / rolling deploys, where every
    ``cumulative_*_mass`` is ``NULL``): the caller then blends over the
    overhead split alone, the same neutral behavior as before backfill. Any
    class whose stored running sum is ``NULL`` is simply omitted (unseen).
    """
    row = await conn.fetchrow(
        "SELECT cumulative_text_mass, cumulative_tool_result_mass, "
        "       cumulative_thinking_mass, cumulative_tool_use_mass "
        "FROM events "
        "WHERE session_id = $1 AND account_id = $2 "
        "AND kind = 'message' AND cumulative_tokens IS NOT NULL "
        "ORDER BY seq DESC LIMIT 1",
        session_id,
        account_id,
    )
    if row is None:
        return {}
    stored = {
        "text": row["cumulative_text_mass"],
        "tool_result": row["cumulative_tool_result_mass"],
        "thinking": row["cumulative_thinking_mass"],
        "tool_use": row["cumulative_tool_use_mass"],
    }
    out: dict[str, float] = {}
    for cls, mass in stored.items():
        if mass is None:
            continue
        out[cls] = float(mass) if mass > 0 else 0.0
    return out


async def read_windowed_events(
    conn: asyncpg.Connection[Any],
    session_id: str,
    *,
    account_id: str,
    window_min: int,
    window_max: int,
    model: str,
    overhead_local: int | OverheadLocalSplit,
) -> WindowedEvents:
    """Read message events for the session's trailing context window.

    Uses the ``cumulative_tokens`` column to compute the chunked-window
    snap boundary (the shared snap math, :func:`~aios.harness.tokens.tokens_to_drop`)
    and loads only the events past that boundary.

    ``cumulative_tokens`` is stored in model-agnostic units (see
    :func:`aios.harness.tokens.approx_tokens`), so the raw value
    systematically diverges from what the provider actually counts —
    ~18 % low on Sonnet 4.6, ~34 % low on Opus 4.7.  This function
    corrects for that at read time: ``window_min`` / ``window_max`` are
    interpreted as provider tokens, ``total_effective = total_local * R``
    where ``R`` is the composition-weighted class ratio, and the drop boundary is
    translated back to local units for the ``cumulative_tokens`` index
    scan.  When the model has fewer than ``model_token_class_ratios``' sample
    threshold, ``R`` is ``1.0`` and the math reduces to the plain
    chunked-snap algorithm.

    ``overhead_local`` is the token cost the caller will add on top of
    the returned events — system prompt plus tool schemas — in local
    (``approx_tokens``) units.  It is NOT included in
    ``cumulative_tokens``, so the windower has to subtract it from the
    effective budget up-front or the sent prompt will exceed
    ``window_max`` by the overhead amount.  Callers that don't have any
    such overhead (preview tooling, test scaffolds) pass ``0``.

    ``model`` must be the session's currently-active model string —
    ``agent.model`` on the session's pinned agent/version.  The same
    string is what :func:`~aios.harness.loop.run_session_step` stamps on
    ``model_request_end`` spans, so stamp-side and query-side stay
    partitioned on identical keys.

    Prefix-cache invariant: the plain chunked-snap algorithm gave a
    *strict* guarantee of byte-identical prompt prefix within a snap
    chunk.  With the ratio correction this remains stable in practice
    because :func:`model_token_class_ratios` uses a lifetime aggregate and
    standard-error bucketing, so mature calibrations do not drift on every
    new sample.  Early calibrations are coarse by design and converge as
    the sample count grows.

    Falls back to :func:`read_message_events` (loading all events) when
    cumulative data is not available (pre-backfill sessions or rolling
    deploys) or when the entire session fits within ``window_max``.

    When the boundary excludes message events, the result carries a
    :class:`~aios.harness.window.WindowOmission` (issue #738), computed
    against the same ``cumulative_tokens`` boundary as the retained scan
    — exact complements.  Cache-stability rationale lives on the class.
    """
    # Index seek: total cumulative tokens from the latest message event.
    total = await _latest_cumulative_tokens(conn, session_id)

    # Fallback: no cumulative data yet — load everything.
    if total is None:
        return WindowedEvents(
            events=await queries.read_windowed_context_events(
                conn, session_id, account_id=account_id
            ),
            omission=None,
        )

    # Per-class coefficients (issue #1609). The windower consumes only the
    # composition-weighted *blended* R_eff — the raw per-class coefficients
    # are not individually load-bearing (collinear; see the issue's
    # robustness note). When the model is below the calibration sample
    # threshold (or any class unseen), every coefficient is 1.0 and R_eff
    # reduces to today's neutral behavior, byte-identical.
    coefs = await queries.model_token_class_ratios(conn, model, account_id=account_id)

    # Re-derive the session's retained-slate class mix in-process from
    # role+data already in the log (per-class neutral token mass via
    # cumulative_tokens deltas). Combined with the overhead split, this is
    # the composition R_eff is blended over.
    overhead = _normalize_overhead(overhead_local)
    events_mass = await _retained_class_mass(conn, session_id, account_id=account_id)

    # Build the full composition vector: events (text/tool_result/thinking/
    # tool_use) plus the overhead split (system/tools, and reserves folded
    # into text as conservative text-shaped padding).
    composition: dict[str, float] = dict(events_mass)
    composition["system"] = composition.get("system", 0.0) + overhead.system
    composition["tools"] = composition.get("tools", 0.0) + overhead.tools
    composition["text"] = composition.get("text", 0.0) + overhead.reserves

    ratio = blended_r_eff(coefs, composition)

    # SAFETY MARGIN (acceptance #4): inflate the budgeted provider total by
    # the per-class residual MAX (~30%) so a composition shift toward the
    # heaviest class degrades to "drops one chunk early," never a cap
    # breach. ``effective`` units are now "budgeted provider tokens incl.
    # safety slack"; the snap math operates entirely in this space.
    #
    # The margin is gated on a CALIBRATED fit. When the model is below the
    # sample threshold every coefficient is exactly the neutral 1.0; in that
    # case we keep margin == 1.0 so the windower reproduces today's
    # byte-identical drop boundary (acceptance #5) — the slack ships only
    # together with the Layer-2 calibrated coefficients it protects.
    calibrated = any(c != 1.0 for c in coefs.values())
    margin = (1.0 + _WINDOW_SAFETY_MARGIN) if calibrated else 1.0
    # Two-way guard: calibration may make the conversion larger, never
    # smaller than neutral.  A low fitted ratio must not inflate retention.
    effective_ratio = max(1.0, ratio * margin)

    # Shrink the effective window by the caller's overhead contribution.
    # Apply R_eff (x margin) to the overhead up-front so the subtraction
    # happens in the same effective space tokens_to_drop operates in.
    overhead_effective = round(overhead.total * effective_ratio)
    events_window_max = window_max - overhead_effective
    events_window_min = max(0, window_min - overhead_effective)
    if events_window_max <= 0:
        raise ValueError(
            f"system+tools overhead ({overhead_effective} provider tokens) "
            f"exceeds window_max ({window_max}); no budget remains for events"
        )

    total_effective = round(total * effective_ratio)
    if total_effective <= events_window_max:
        return WindowedEvents(
            events=await queries.read_windowed_context_events(
                conn, session_id, account_id=account_id
            ),
            omission=None,
        )

    from aios.harness.tokens import tokens_to_drop

    # Forward-convert local → effective with plain rounding: best-estimate
    # of the provider-token total.  Back-convert effective → local with
    # ceil: deliberately asymmetric so the post-drop remaining fits under
    # ``window_max`` even when ratio error would otherwise leave one
    # message straddling the boundary.
    drop_effective = tokens_to_drop(
        total_effective, window_min=events_window_min, window_max=events_window_max
    )
    if drop_effective == 0:
        return WindowedEvents(
            events=await queries.read_windowed_context_events(
                conn, session_id, account_id=account_id
            ),
            omission=None,
        )

    drop = math.ceil(drop_effective / effective_ratio)

    # Never drop the entire window: the window must keep a non-empty tail.
    # Here ``events_window_min`` can clamp to 0 when overhead exceeds
    # ``window_min`` (above), and the asymmetric ceil back-conversion can then
    # push ``drop`` up to ``total`` — the retained scan (``cumulative_tokens >
    # drop``) would match zero rows while the omission complement still matches
    # every row. That pairing (empty events + a non-None omission) crashes
    # ``build_messages``, which reads ``events[0].created_at`` to anchor the
    # omission marker and relies on the inverse invariant. Clamp so the most
    # recent event always survives (its ``cumulative_tokens == total``) — the
    # retain-the-tail-even-when-oversized guarantee.
    drop = min(drop, total - 1)

    # Bounded range scan: messages past the boundary, plus the FS-loss
    # notices past the dropped-message prefix. Bare call (not via ``queries``)
    # so the fallback stub on the package attribute does not intercept the
    # retained-window read — keeping the unit FakeConn path exercised.
    events = await read_windowed_context_events(conn, session_id, account_id=account_id, drop=drop)

    # The omitted complement: same boundary expression as the retained
    # scan (``<=`` vs ``>``), a seq-prefix of the log. Its ``omitted_messages``
    # (user+assistant only) is now the boundary row's ``cumulative_messages``
    # running count -- O(1), the escape hatch the in-code comment named
    # (issue #1657): the boundary row is the message with the greatest
    # ``cumulative_tokens <= drop`` (one index seek on the same
    # ``events_session_cumtokens_idx``), and its running count IS the count of
    # user/assistant messages through it = the omitted count. ``began_at`` is
    # the conversation start = the FIRST message's ``created_at`` (a seq-ASC
    # LIMIT 1 index seek, not a ``min()`` aggregate scan). NULL boundary row
    # means the drop excludes no message (oversized first event straddling it)
    # -> no omission.
    boundary_row = await conn.fetchrow(
        "SELECT cumulative_messages, created_at "
        "FROM events "
        "WHERE session_id = $1 AND account_id = $3 AND kind = 'message' "
        "AND cumulative_tokens <= $2 "
        "ORDER BY cumulative_tokens DESC LIMIT 1",
        session_id,
        drop,
        account_id,
    )
    if boundary_row is None:
        # Nothing omitted.
        return WindowedEvents(events=events, omission=None)

    omitted_messages = boundary_row["cumulative_messages"]
    if omitted_messages is None:
        # Pre-backfill boundary row (``cumulative_messages`` is NULL, as it is
        # on every message that predates migration 0127). Fall back to the
        # role-filtered ``count(*)`` over the omitted prefix -- the same value,
        # just O(omitted-prefix). Bounded by the ``cumulative_tokens <= drop``
        # index cond, and only for the un-backfilled tail (transient across a
        # rolling deploy), so it never re-introduces the O(session-size) term.
        omitted_messages = await conn.fetchval(
            "SELECT count(*) FILTER (WHERE role IN ('user', 'assistant')) "
            "FROM events "
            "WHERE session_id = $1 AND account_id = $3 AND kind = 'message' "
            "AND cumulative_tokens <= $2",
            session_id,
            drop,
            account_id,
        )

    # Conversation start: the first message's ``created_at``. A seq-ASC index
    # seek, not an aggregate over the omitted span. Guaranteed present here
    # (``boundary_row`` proves at least one omitted message exists).
    began_at = await conn.fetchval(
        "SELECT created_at FROM events "
        "WHERE session_id = $1 AND account_id = $2 AND kind = 'message' "
        "ORDER BY seq ASC LIMIT 1",
        session_id,
        account_id,
    )
    omission = WindowOmission(began_at=began_at, omitted_messages=omitted_messages)
    return WindowedEvents(events=events, omission=omission)


async def find_latest_model_workflow_park(
    conn: asyncpg.Connection[Any],
    session_id: str,
    *,
    account_id: str,
) -> dict[str, Any] | None:
    """Return the ``data`` of the most recent *un-consumed* ``model_workflow_park`` span, or ``None``.

    The workflow-bound model dispatch (#1634) journals one park per inference it
    defers; the latest one names the run the next step harvests. The harvest read
    (:func:`aios.harness.model_workflow.take_pending_harvest`) pairs this park with
    its matching harvest event.

    A park is **consumed** once its harvest has been folded into a turn — the fold
    step appends exactly one ``model_workflow_harvest_end`` span carrying the
    park's ``run_id`` (on every fold path: a folded assistant turn AND the
    errored / invalid-shape latches). This query EXCLUDES any park whose ``run_id``
    already has that consumed-marker span, so a session that has finished a parked
    turn reads ``None`` and the caller launches a *fresh* park for the next
    stimulus. Without this filter the unconditional ``ORDER BY seq DESC LIMIT 1``
    would keep returning the most recent (already-harvested) park, and a later
    stimulus would re-fold the stale inner answer forever — the park-time
    ``reacting_to`` never reaches the new stimulus seq, so the session stays a
    sweep candidate and re-wakes onto the same stale park (a permanent wedge).
    The exclusion is what makes the consumed park un-returnable; it is the
    enforcing mechanism, not a narrative assumption about ``reacting_to``.
    """
    row = await conn.fetchrow(
        "SELECT p.data FROM events p "
        "WHERE p.session_id = $1 AND p.account_id = $2 AND p.kind = 'span' "
        "AND p.data->>'event' = 'model_workflow_park' "
        "AND NOT EXISTS ("
        "    SELECT 1 FROM events h "
        "    WHERE h.session_id = $1 AND h.account_id = $2 AND h.kind = 'span' "
        "    AND h.data->>'event' = 'model_workflow_harvest_end' "
        "    AND h.data->>'run_id' = p.data->>'run_id'"
        ") "
        "ORDER BY p.seq DESC LIMIT 1",
        session_id,
        account_id,
    )
    if row is None:
        return None
    data = row["data"]
    return data if isinstance(data, dict) else None


# The crash-recovery park scan (#1635). Held as a module constant — rather than
# inlined in :func:`find_unharvested_model_dispatch_parks` — so the perf test can
# EXPLAIN the exact production query text and assert an index scan (not a
# ``Seq Scan on events``). ``{scope_clause}`` is empty on the cross-session
# periodic path, so the ``model_workflow_park`` partial index (migration 0131) is
# what restricts the outer scan there; the two ``NOT EXISTS`` anti-joins are served
# by the companion ``model_workflow_harvest_end`` / ``model_workflow_harvest``
# partial indexes added in the same migration.
UNHARVESTED_MODEL_DISPATCH_PARKS_SQL = (
    "SELECT DISTINCT ON (p.session_id) p.session_id, p.account_id, "
    "    p.data->>'run_id' AS run_id "
    "FROM events p "
    "WHERE p.kind = 'span' AND p.data->>'event' = 'model_workflow_park' "
    "{scope_clause} "
    # not yet folded (consumed)
    "AND NOT EXISTS ("
    "    SELECT 1 FROM events c "
    "    WHERE c.session_id = p.session_id AND c.account_id = p.account_id "
    "    AND c.kind = 'span' AND c.data->>'event' = 'model_workflow_harvest_end' "
    "    AND c.data->>'run_id' = p.data->>'run_id'"
    ") "
    # not yet harvested (the run's terminal state was never written back)
    "AND NOT EXISTS ("
    "    SELECT 1 FROM events h "
    "    WHERE h.session_id = p.session_id AND h.account_id = p.account_id "
    "    AND h.kind = 'span' AND h.data->>'event' = 'model_workflow_harvest' "
    "    AND h.data->>'run_id' = p.data->>'run_id'"
    ") "
    "AND p.data->>'run_id' IS NOT NULL "
    "ORDER BY p.session_id, p.seq DESC"
)


async def find_unharvested_model_dispatch_parks(
    conn: asyncpg.Connection[Any],
    *,
    session_id: str | None = None,
) -> list[tuple[str, str, str]]:
    """Return ``(session_id, run_id, account_id)`` for every **stranded** model-dispatch park.

    The crash-recovery query (#1635). A model-dispatch park is *stranded* when:

    * it is the session's latest **un-consumed** park — no ``model_workflow_harvest_end``
      span for its ``run_id`` (it has not been folded), AND
    * its run has **not been harvested** — no ``model_workflow_harvest`` span for its
      ``run_id`` (the bound run's terminal state was never written back to the session).

    Such a park owes a harvest task that this worker (or its predecessor) must run. On a
    normal park the live in-process task writes the harvest; after a worker crash that
    task is gone, so the sweep re-derives this set and re-parks each one. The harvest
    write is idempotent on ``run_id`` (:func:`aios.harness.model_workflow.write_harvest_event`),
    so re-parking a park whose harvest a racing task is about to write is harmless.

    Unlike the ``call_*`` ghost scan this does NOT key on a ``tool_call_id`` in a
    persisted assistant message — a model-dispatch park has neither (the assistant
    message is the run's output, produced only once the park resolves). It keys on the
    ``model_workflow_park`` span itself, the durable park record.

    Scoped to one ``session_id`` when given (the per-session sweep); otherwise
    cross-session (the boot / periodic sweep). Returns at most one row per session: the
    park/harvest pair keys off the run id and a session owes at most one parked inference
    at a time, so the latest un-consumed, un-harvested park is the only stranded one.
    """
    scope_clause = "AND p.session_id = $1" if session_id else ""
    params: list[Any] = [session_id] if session_id else []
    rows = await conn.fetch(
        UNHARVESTED_MODEL_DISPATCH_PARKS_SQL.format(scope_clause=scope_clause),
        *params,
    )
    return [(r["session_id"], r["run_id"], r["account_id"]) for r in rows]


async def find_model_workflow_harvest(
    conn: asyncpg.Connection[Any],
    session_id: str,
    *,
    run_id: str,
    account_id: str,
) -> dict[str, Any] | None:
    """Return the ``data`` of the ``model_workflow_harvest`` span for ``run_id``, or ``None``.

    ``None`` while the bound run has not resolved (the step ends owing the message
    again). Keyed on ``run_id`` so the dedup guard on the harvest write is exact.
    """
    row = await conn.fetchrow(
        "SELECT data FROM events "
        "WHERE session_id = $1 AND account_id = $2 AND kind = 'span' "
        "AND data->>'event' = 'model_workflow_harvest' "
        "AND data->>'run_id' = $3 "
        "ORDER BY seq DESC LIMIT 1",
        session_id,
        account_id,
        run_id,
    )
    if row is None:
        return None
    data = row["data"]
    return data if isinstance(data, dict) else None
