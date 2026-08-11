"""Guard for security-relevant dependency floors (#1714).

Four dependency constraints were behind their security-relevant minimums.
This test pins both the *manifest* floors (``pyproject.toml``) and the
*resolved* versions (``uv.lock``) so a future re-resolve can't silently
regress into a known-vulnerable version.

Covered items (see #1714):

* **litellm** ``>=1.83.7`` — CVE-2026-42208 (LiteLLM Proxy pre-auth SQLi).
  aios uses litellm as a client library, not the Proxy, so the vulnerable
  path is unreachable; the floor is SCA hygiene.
* **aiohttp** ``>=3.14.1`` (resolved-only; transitive via litellm) — clears
  the incomplete-websocket-frame DoS (CVE-2026-54274), multipart CRLF
  (CVE-2026-50269) and CookieJar pickle (CVE-2026-34993) advisories.
* **h11** ``>=0.16`` — CVE-2025-43859 (request-smuggling-vulnerable parser).
* **mcp** ``>=1.20,<2`` — SDK v2 is a breaking rearchitecture; cap defends
  against an unattended major bump.

Pure file parsing (``tomllib`` + ``packaging``); no DB, no network.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
UV_LOCK = PROJECT_ROOT / "uv.lock"

# Minimum *resolved* versions the lock must satisfy. These are the
# security-relevant floors from #1714.
LOCKED_FLOORS = {
    "litellm": Version("1.83.7"),
    "aiohttp": Version("3.14.1"),
    "h11": Version("0.16"),
}

# mcp must stay below the v2 breaking rearchitecture.
MCP_MAX_EXCLUSIVE = Version("2")


def _direct_requirements() -> dict[str, Requirement]:
    data = tomllib.loads(PYPROJECT.read_text())
    reqs: dict[str, Requirement] = {}
    for raw in data["project"]["dependencies"]:
        req = Requirement(raw)
        reqs[req.name.lower()] = req
    return reqs


def _locked_versions() -> dict[str, Version]:
    data = tomllib.loads(UV_LOCK.read_text())
    return {
        pkg["name"].lower(): Version(pkg["version"])
        for pkg in data.get("package", [])
        if "version" in pkg
    }


def test_manifest_floors_declared() -> None:
    """pyproject.toml must declare the hardened floors / cap."""
    reqs = _direct_requirements()

    litellm = reqs["litellm"]
    assert not litellm.specifier.contains("1.91.2", prereleases=True), (
        "litellm floor must exclude <1.91.3 (known proxy-code criticals)"
    )
    assert litellm.specifier.contains("1.91.3", prereleases=True)
    assert not litellm.specifier.contains("1.92.0", prereleases=True), (
        "litellm must stay below 1.92 until portable wheels return"
    )

    h11 = reqs["h11"]
    assert not h11.specifier.contains("0.15", prereleases=True), (
        "h11 floor must exclude <0.16 (CVE-2025-43859)"
    )
    assert h11.specifier.contains("0.16", prereleases=True)

    mcp = reqs["mcp"]
    assert not mcp.specifier.contains("2.0.0", prereleases=True), (
        "mcp must be capped <2 (v2 is a breaking rearchitecture)"
    )
    assert mcp.specifier.contains("1.20", prereleases=True)


def test_locked_versions_meet_floors() -> None:
    """uv.lock must resolve to versions at or above the security floors."""
    locked = _locked_versions()

    for name, floor in LOCKED_FLOORS.items():
        assert name in locked, f"{name} missing from uv.lock"
        assert locked[name] >= floor, f"{name} locked at {locked[name]}, must be >= {floor}"

    assert "mcp" in locked, "mcp missing from uv.lock"
    assert locked["mcp"] < MCP_MAX_EXCLUSIVE, (
        f"mcp locked at {locked['mcp']}, must be < {MCP_MAX_EXCLUSIVE}"
    )


def test_procrastinate_pin_untouched() -> None:
    """#1714 explicitly forbids touching the intentional procrastinate pin."""
    reqs = _direct_requirements()
    assert str(reqs["procrastinate"].specifier) == "==3.8.1"
