"""The ``tier:`` model-string scheme (docs/rlm.md).

Mirrors the ``workflow:`` scheme in :mod:`aios.harness.model_binding`: a
prefix constant, an ``is_``/``resolve_`` pair, fail-loud on malformed input.
``tier:<name>`` names an entry in ``Settings.model_tiers`` and resolves to
that raw provider model string **late** — at the surface chokepoint
(``services.agents._surface_from_agent``) and the workflow ``call_llm``
leaf — so a config remap retargets every future step with zero per-site
edits, and no cached provider-sniff (``model_descriptor``,
``_derive_provider``) ever sees an unresolved tier name.

Tier values are validated at Settings construction to never themselves be
``tier:``/``workflow:`` strings, so resolution is single-step and the
model-binding privilege guards cannot be bypassed by indirection.
"""

from __future__ import annotations

from collections.abc import Mapping

TIER_MODEL_PREFIX = "tier:"


class UnknownModelTierError(ValueError):
    """A ``tier:<name>`` referenced a tier absent from ``Settings.model_tiers``."""

    def __init__(self, tier: str, configured: Mapping[str, str]) -> None:
        self.tier = tier
        super().__init__(
            f"unknown model tier {tier!r}; configured tiers: "
            f"{sorted(configured) if configured else '(none — set AIOS_MODEL_TIERS)'}"
        )


def is_tier_model(model: str) -> bool:
    """True iff ``model`` uses the ``tier:`` scheme."""
    return model.startswith(TIER_MODEL_PREFIX)


def tier_model_string(tier: str) -> str:
    """The stampable model string for ``tier`` (e.g. ``"sub"`` → ``"tier:sub"``)."""
    return f"{TIER_MODEL_PREFIX}{tier}"


def resolve_tier_model(model: str, tiers: Mapping[str, str]) -> str:
    """Resolve a ``tier:`` scheme string to its raw model; passthrough otherwise.

    Raises :class:`UnknownModelTierError` on an unconfigured tier name — a
    misconfigured tier fails the step loudly rather than silently degrading
    provider sniffing, prompt caching, and auth resolution downstream.
    """
    if not is_tier_model(model):
        return model
    tier = model[len(TIER_MODEL_PREFIX) :]
    resolved = tiers.get(tier)
    if resolved is None:
        raise UnknownModelTierError(tier, tiers)
    return resolved
