"""Unit tests for the ``tier:`` model-string scheme (docs/rlm.md).

Covers :mod:`aios.harness.model_tier` (predicate, constructor, resolution,
the fail-loud unknown-tier error) and the ``Settings.model_tiers`` startup
validator that keeps tier values raw provider strings so resolution stays
single-step and the model-binding privilege guards can't be bypassed by
indirection.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aios.harness.model_tier import (
    TIER_MODEL_PREFIX,
    UnknownModelTierError,
    is_tier_model,
    resolve_tier_model,
    tier_model_string,
)

# ─── scheme predicate + constructor ──────────────────────────────────────────


def test_is_tier_model() -> None:
    assert is_tier_model("tier:sub") is True
    assert is_tier_model("openrouter/foo") is False
    assert is_tier_model("workflow:gpt") is False


def test_tier_model_string_round_trips() -> None:
    assert tier_model_string("sub") == f"{TIER_MODEL_PREFIX}sub"
    assert is_tier_model(tier_model_string("sub"))


# ─── resolution ──────────────────────────────────────────────────────────────


def test_resolve_passthrough_for_raw_model() -> None:
    """A non-``tier:`` string is returned unchanged, even with tiers configured."""
    assert resolve_tier_model("openrouter/foo", {"sub": "x"}) == "openrouter/foo"


def test_resolve_tier_to_configured_model() -> None:
    assert resolve_tier_model("tier:sub", {"sub": "openrouter/foo"}) == "openrouter/foo"


def test_unknown_tier_error_lists_configured_names() -> None:
    with pytest.raises(UnknownModelTierError) as exc_info:
        resolve_tier_model("tier:frontier", {"sub": "m", "verify": "n"})
    assert exc_info.value.tier == "frontier"
    message = str(exc_info.value)
    assert "frontier" in message
    assert "sub" in message and "verify" in message


def test_unknown_tier_error_empty_config_hints_at_env() -> None:
    with pytest.raises(UnknownModelTierError, match="AIOS_MODEL_TIERS"):
        resolve_tier_model("tier:sub", {})


# ─── Settings.model_tiers validator ──────────────────────────────────────────


def _secrets(tmp_path: Path) -> Path:
    secrets = tmp_path / "secrets.env"
    secrets.write_text("AIOS_VAULT_KEY=v\nAIOS_EGRESS_CA_KEY=e\nAIOS_DB_URL=postgresql://x/y\n")
    return secrets


@pytest.mark.parametrize("value", ["tier:other", "workflow:gpt"])
def test_settings_rejects_scheme_valued_tiers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """A tier value that is itself a scheme string fails Settings construction
    loudly — resolution must never be recursive or guard-bypassing."""
    from pydantic import ValidationError

    from aios.config import Settings

    monkeypatch.setenv("AIOS_MODEL_TIERS", json.dumps({"sub": value}))
    with pytest.raises(ValidationError, match="raw provider model strings"):
        Settings(_env_file=(str(_secrets(tmp_path)),))


def test_settings_accepts_raw_tier_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aios.config import Settings

    monkeypatch.setenv("AIOS_MODEL_TIERS", json.dumps({"sub": "openrouter/foo"}))
    settings = Settings(_env_file=(str(_secrets(tmp_path)),))
    assert settings.model_tiers == {"sub": "openrouter/foo"}
