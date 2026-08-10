"""Unit coverage for issue #1745: attachment render cache (A), clamp-pass
fit-verdict cache (B), persist-once (C), and the loop-thread offload (D).
"""

from __future__ import annotations

import base64
import hashlib
import io
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from aios.config import get_settings
from aios.harness import context, image_resize, vision
from aios.harness.context import (
    _ATTACHMENT_CACHE,
    _CLAMP_CACHE,
    _apply_attachments,
    _clamp_cache_key,
    _clear_attachment_cache,
    _clear_clamp_cache,
    build_messages,
)
from aios.harness.context_persist import persist_clamped_image_parts
from aios.harness.image_resize import ImageDownsampleError
from aios.models.events import Event
from aios.sandbox.volumes import session_attachments_dir
from tests.helpers.images import valid_jpeg_bytes

_CREATED_AT = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
_VALID_JPEG = valid_jpeg_bytes()


@pytest.fixture(autouse=True)
def _reset_caches() -> Any:
    _clear_attachment_cache()
    _clear_clamp_cache()
    yield
    _clear_attachment_cache()
    _clear_clamp_cache()


@pytest.fixture
def temp_workspace_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    settings = get_settings()
    monkeypatch.setattr(settings, "workspace_root", tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def _stub_supports_vision(monkeypatch: pytest.MonkeyPatch) -> Any:
    saved = dict(vision._VISION_OVERRIDES)
    vision._VISION_OVERRIDES.clear()
    vision._VISION_OVERRIDES["model/vision"] = True
    yield
    vision._VISION_OVERRIDES.clear()
    vision._VISION_OVERRIDES.update(saved)


def _stage_image(
    workspace_root: Path, session_id: str, connector: str, name: str, payload: bytes
) -> tuple[str, Path]:
    session_dir = session_attachments_dir(session_id) / connector
    session_dir.mkdir(parents=True, exist_ok=True)
    file_path = session_dir / name
    file_path.write_bytes(payload)
    return f"/mnt/attachments/{connector}/{name}", file_path


def _user_event(*, content: str = "hi", attachments: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "role": "user",
        "content": content,
        "metadata": {"channel": "echo/acct/chat-1", "attachments": attachments},
    }


class TestAttachmentRenderCache:
    def test_second_render_same_identity_zero_read_bytes(
        self, temp_workspace_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sandbox_path, _host_path = _stage_image(
            temp_workspace_root, "sess-1", "echo", "photo.jpg", _VALID_JPEG
        )
        record = {
            "filename": "photo.jpg",
            "content_type": "image/jpeg",
            "size": len(_VALID_JPEG),
            "in_sandbox_path": sandbox_path,
        }

        read_calls = {"n": 0}
        orig_read_bytes = Path.read_bytes

        def counting_read_bytes(self: Path) -> bytes:
            read_calls["n"] += 1
            return orig_read_bytes(self)

        monkeypatch.setattr(Path, "read_bytes", counting_read_bytes)

        msg1: dict[str, Any] = {"content": ""}
        _apply_attachments(msg1, [record], model="model/vision", session_id="sess-1")
        assert read_calls["n"] == 1
        part1 = msg1["content"][0]

        msg2: dict[str, Any] = {"content": ""}
        _apply_attachments(msg2, [record], model="model/vision", session_id="sess-1")
        # Zero additional read_bytes calls on the cache hit.
        assert read_calls["n"] == 1
        part2 = msg2["content"][0]

        assert part1 == part2
        # Never share the cached dict — each hit rebuilds the part.
        assert part1 is not part2
        assert part1["image_url"] is not part2["image_url"]

    def test_mtime_bump_triggers_reread(
        self, temp_workspace_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sandbox_path, host_path = _stage_image(
            temp_workspace_root, "sess-1", "echo", "photo.jpg", _VALID_JPEG
        )
        record = {
            "filename": "photo.jpg",
            "content_type": "image/jpeg",
            "size": len(_VALID_JPEG),
            "in_sandbox_path": sandbox_path,
        }
        read_calls = {"n": 0}
        orig_read_bytes = Path.read_bytes

        def counting_read_bytes(self: Path) -> bytes:
            read_calls["n"] += 1
            return orig_read_bytes(self)

        monkeypatch.setattr(Path, "read_bytes", counting_read_bytes)

        msg: dict[str, Any] = {"content": ""}
        _apply_attachments(msg, [record], model="model/vision", session_id="sess-1")
        assert read_calls["n"] == 1

        # Bump mtime by rewriting with different bytes (same size class ok).
        import os
        import time

        time.sleep(0.01)
        host_path.write_bytes(_VALID_JPEG)
        os.utime(host_path, None)

        msg2: dict[str, Any] = {"content": ""}
        _apply_attachments(msg2, [record], model="model/vision", session_id="sess-1")
        assert read_calls["n"] == 2

    def test_byte_cap_eviction(
        self, temp_workspace_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AIOS_CONTEXT_IMAGE_CACHE_MAX_BYTES", str(len(_VALID_JPEG) * 2 + 10))
        paths = []
        for i in range(5):
            sandbox_path, _ = _stage_image(
                temp_workspace_root, "sess-1", "echo", f"photo{i}.jpg", _VALID_JPEG
            )
            paths.append(sandbox_path)

        for i, sandbox_path in enumerate(paths):
            record = {
                "filename": f"photo{i}.jpg",
                "content_type": "image/jpeg",
                "size": len(_VALID_JPEG),
                "in_sandbox_path": sandbox_path,
            }
            msg: dict[str, Any] = {"content": ""}
            _apply_attachments(msg, [record], model="model/vision", session_id="sess-1")

        # Cache stayed bounded — not all 5 entries retained.
        assert len(_ATTACHMENT_CACHE) < 5

    def test_marker_verdict_cached(
        self, temp_workspace_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        corrupt = b"\xff\xd8\xff" + b"\x00" * 20  # JPEG magic, undecodable body
        sandbox_path, _ = _stage_image(temp_workspace_root, "sess-1", "echo", "bad.jpg", corrupt)
        record = {
            "filename": "bad.jpg",
            "content_type": "image/jpeg",
            "size": len(corrupt),
            "in_sandbox_path": sandbox_path,
        }
        read_calls = {"n": 0}
        orig_read_bytes = Path.read_bytes

        def counting_read_bytes(self: Path) -> bytes:
            read_calls["n"] += 1
            return orig_read_bytes(self)

        monkeypatch.setattr(Path, "read_bytes", counting_read_bytes)

        msg1: dict[str, Any] = {"content": ""}
        _apply_attachments(msg1, [record], model="model/vision", session_id="sess-1")
        assert read_calls["n"] == 1
        assert isinstance(msg1["content"], str)
        assert "[image: bad.jpg" in msg1["content"]

        msg2: dict[str, Any] = {"content": ""}
        _apply_attachments(msg2, [record], model="model/vision", session_id="sess-1")
        # Marker verdict cached: no second read.
        assert read_calls["n"] == 1
        assert "[image: bad.jpg" in msg2["content"]


def _big_png_data_url() -> tuple[str, bytes]:
    buf = io.BytesIO()
    Image.new("RGB", (4000, 4000), (10, 20, 30)).save(buf, format="PNG")
    raw = buf.getvalue()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/png;base64,{b64}", raw


def _tool_event_with_image(url: str) -> Event:
    return Event(
        id="evt_img",
        session_id="sess_01TEST",
        seq=3,
        kind="message",
        data={
            "role": "tool",
            "tool_call_id": "a",
            "content": [
                {"type": "text", "text": "Image: huge.png"},
                {"type": "image_url", "image_url": {"url": url}},
            ],
        },
        created_at=_CREATED_AT,
        orig_channel=None,
        focal_channel_at_arrival=None,
    )


def _preceding_events() -> list[Event]:
    return [
        Event(
            id="evt_1",
            session_id="sess_01TEST",
            seq=1,
            kind="message",
            data={"role": "user", "content": "show photo"},
            created_at=_CREATED_AT,
            orig_channel=None,
            focal_channel_at_arrival=None,
        ),
        Event(
            id="evt_2",
            session_id="sess_01TEST",
            seq=2,
            kind="message",
            data={
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "a", "type": "function", "function": {"name": "read", "arguments": "{}"}}
                ],
            },
            created_at=_CREATED_AT,
            orig_channel=None,
            focal_channel_at_arrival=None,
        ),
    ]


class TestClampFitVerdictCache:
    def test_second_build_zero_full_decode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Second build over the SAME (unpersisted, still-oversize) window
        # re-derives the same result but must not re-decode: the fit
        # verdict for a DEGRADE-shrunk-in-memory part isn't cached as FITS
        # (spec: only persisted-fit or DEGRADE verdicts are cached), but a
        # part that already FITS IS a cache hit. Exercise that directly:
        # build twice over a window whose part already fits, counting only
        # FULL-payload decodes (the mime-corrector's own unconditional
        # 24-char header sniff is orthogonal and out of scope — #1745
        # explicitly leaves it unchanged).
        small_buf = io.BytesIO()
        Image.new("RGB", (64, 64), (1, 2, 3)).save(small_buf, format="PNG")
        small_b64 = base64.b64encode(small_buf.getvalue()).decode("ascii")
        small_url = f"data:image/png;base64,{small_b64}"
        small_events = [*_preceding_events(), _tool_event_with_image(small_url)]

        full_decode_calls = {"n": 0}
        orig_decode = base64.b64decode

        def counting_decode(data: Any, *args: Any, **kwargs: Any) -> bytes:
            if isinstance(data, str) and len(data) == len(small_b64):
                full_decode_calls["n"] += 1
            return orig_decode(data, *args, **kwargs)

        monkeypatch.setattr(base64, "b64decode", counting_decode)

        is_oversize_calls = {"n": 0}
        orig_is_oversize = image_resize.is_oversize_image

        def counting_is_oversize(*args: Any, **kwargs: Any) -> bool:
            is_oversize_calls["n"] += 1
            return orig_is_oversize(*args, **kwargs)

        monkeypatch.setattr(context, "is_oversize_image", counting_is_oversize)

        build_messages(small_events, system_prompt=None)
        assert full_decode_calls["n"] == 1
        assert is_oversize_calls["n"] == 1

        full_decode_calls["n"] = 0
        is_oversize_calls["n"] = 0
        build_messages(small_events, system_prompt=None)
        # Cache hit: zero FULL decode, zero is_oversize_image call.
        assert full_decode_calls["n"] == 0
        assert is_oversize_calls["n"] == 0

    def test_clamp_key_uses_stable_blake2b_digest(self) -> None:
        data_b64 = "same-length-payload"
        length, digest = _clamp_cache_key(data_b64)

        assert length == len(data_b64)
        assert digest == hashlib.blake2b(data_b64.encode("ascii"), digest_size=16).digest()
        assert isinstance(digest, bytes)

    def test_clamp_key_non_ascii_payload_does_not_raise(self) -> None:
        # Regression (PR#1829): the data-url payload is untrusted and may
        # contain non-ASCII characters, or lone surrogates reachable via JSON
        # \uD escapes. ``.encode("ascii")`` raised UnicodeEncodeError on those,
        # crashing context composition BEFORE the guarded b64decode could
        # degrade the malformed part — a permanent per-session wedge because
        # the offending event is persisted and replayed every build. The key
        # must be derivable (total codec) for ANY str payload without raising.
        for payload in (
            "iVBORw0KGgoé=",  # non-ASCII (latin-1) char in the payload
            "AAAA\ud800BBBB",  # lone surrogate (JSON \uD800), plain utf-8 fails
            "\U0001f600multibyte",  # astral-plane emoji
        ):
            length, digest = _clamp_cache_key(payload)
            assert length == len(payload)
            assert isinstance(digest, bytes)
            assert len(digest) == 16
            # Stable + deterministic: same payload always keys the same slot.
            assert _clamp_cache_key(payload) == (length, digest)

    def test_degrade_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        url, _raw = _big_png_data_url()
        events = [*_preceding_events(), _tool_event_with_image(url)]

        def raise_downsample(*args: Any, **kwargs: Any) -> Any:
            raise ImageDownsampleError("boom")

        monkeypatch.setattr(context, "_blocking_downsample", raise_downsample)

        msgs1 = build_messages(events, system_prompt=None).messages
        tool_msg1 = msgs1[2]
        assert tool_msg1["content"][1] == {
            "type": "text",
            "text": "[image omitted: too large to display inline]",
        }

        cache_key = _clamp_cache_key(url.partition(",")[2])
        assert cache_key in _CLAMP_CACHE

        # Second build: cached DEGRADE, no re-attempt at downsample.
        calls = {"n": 0}

        def raise_again(*args: Any, **kwargs: Any) -> Any:
            calls["n"] += 1
            raise ImageDownsampleError("boom")

        monkeypatch.setattr(context, "_blocking_downsample", raise_again)
        msgs2 = build_messages(events, system_prompt=None).messages
        assert calls["n"] == 0
        assert msgs2[2]["content"][1] == {
            "type": "text",
            "text": "[image omitted: too large to display inline]",
        }

    def test_size_gate_short_circuits_byte_oversize_no_decode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A part whose base64 length alone proves byte-oversize skips the
        redundant Pillow header check (``is_oversize_image``)."""
        import random

        from aios.harness.vision import INLINE_SIZE_CAP_BYTES

        # A noisy (incompressible) 2000x2000 RGB PNG lands well over the
        # 3.75 MiB byte cap while still being <=2000px/side — genuine
        # BYTE-oversize (not dimension-oversize), the case the size-gate
        # exists for.
        data = random.Random("byte-oversize-seed").randbytes(2000 * 2000 * 3)
        img = Image.frombytes("RGB", (2000, 2000), data)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        raw = buf.getvalue()
        assert len(raw) > INLINE_SIZE_CAP_BYTES
        b64 = base64.b64encode(raw).decode("ascii")
        url = f"data:image/png;base64,{b64}"
        events = [*_preceding_events(), _tool_event_with_image(url)]

        is_oversize_calls = {"n": 0}
        orig_is_oversize = image_resize.is_oversize_image

        def counting_is_oversize(*args: Any, **kwargs: Any) -> bool:
            is_oversize_calls["n"] += 1
            return orig_is_oversize(*args, **kwargs)

        monkeypatch.setattr(context, "is_oversize_image", counting_is_oversize)
        build_messages(events, system_prompt=None)
        # Byte-oversize proven from length alone — size-gate short-circuits
        # without calling is_oversize_image at all.
        assert is_oversize_calls["n"] == 0


class TestPersistClampedImageParts:
    @staticmethod
    def _fake_pool_and_conn() -> tuple[Any, Any, list[tuple[str, str, dict[str, Any]]]]:
        updates: list[tuple[str, str, dict[str, Any]]] = []

        class _Conn:
            pass

        conn = _Conn()

        class _Cm:
            async def __aenter__(self) -> _Conn:
                return conn

            async def __aexit__(self, *exc: object) -> None:
                return None

        class _Pool:
            def acquire(self) -> _Cm:
                return _Cm()

        return _Pool(), conn, updates

    @pytest.mark.asyncio
    async def test_oversize_event_persists_exactly_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        url, _raw = _big_png_data_url()
        events = [*_preceding_events(), _tool_event_with_image(url)]
        pool, _conn, _ = self._fake_pool_and_conn()

        update_calls: list[dict[str, Any]] = []

        async def fake_replace(
            conn_arg: Any, session_id: str, event_id: str, data: Any, *, account_id: str
        ) -> bool:
            update_calls.append({"event_id": event_id, "data": data, "account_id": account_id})
            return True

        monkeypatch.setattr(
            "aios.db.queries.replace_event_data", AsyncMock(side_effect=fake_replace)
        )

        await persist_clamped_image_parts(
            pool, events, session_id="sess_01TEST", account_id="acct_1"
        )
        assert len(update_calls) == 1
        stored_part = update_calls[0]["data"]["content"][1]
        rendered = build_messages(events, system_prompt=None).messages[2]["content"][1]
        assert stored_part == rendered

        # In-memory event updated so THIS step renders persisted bytes.
        assert events[2].data["content"][1] == stored_part

        # Next compose: zero UPDATE, zero downsample (already fits).
        update_calls.clear()
        downsample_calls = {"n": 0}
        orig = image_resize._blocking_downsample

        def counting(*args: Any, **kwargs: Any) -> Any:
            downsample_calls["n"] += 1
            return orig(*args, **kwargs)

        monkeypatch.setattr("aios.harness.context_persist._blocking_downsample", counting)
        await persist_clamped_image_parts(
            pool, events, session_id="sess_01TEST", account_id="acct_1"
        )
        assert len(update_calls) == 0
        assert downsample_calls["n"] == 0

    @pytest.mark.asyncio
    async def test_undownsampleable_never_persisted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        url, _raw = _big_png_data_url()
        events = [*_preceding_events(), _tool_event_with_image(url)]
        pool, _conn, _ = self._fake_pool_and_conn()

        def raise_downsample(*args: Any, **kwargs: Any) -> Any:
            raise ImageDownsampleError("boom")

        monkeypatch.setattr("aios.harness.context_persist._blocking_downsample", raise_downsample)
        update_mock = AsyncMock()
        monkeypatch.setattr("aios.db.queries.replace_event_data", update_mock)

        await persist_clamped_image_parts(
            pool, events, session_id="sess_01TEST", account_id="acct_1"
        )
        update_mock.assert_not_called()
        # Original bytes untouched in memory.
        assert events[2].data["content"][1]["image_url"]["url"] == url

    @pytest.mark.asyncio
    async def test_persist_image_rewrites_false_skips_persist_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``compose_step_context``'s ``persist_image_rewrites`` gate lives
        at the call site (not inside ``persist_clamped_image_parts``
        itself): when ``False``, the composer must not call this function
        at all. Pinned via the composer's own wiring."""
        from aios.harness.step_context import compose_step_context

        persist_spy = AsyncMock()
        monkeypatch.setattr("aios.harness.step_context.persist_clamped_image_parts", persist_spy)

        from aios.models.agents import AgentBinding, StepSurface

        agent = StepSurface(
            model="model/vision",
            system="sys",
            tools=[],
            skills=[],
            mcp_servers=[],
            http_servers=[],
            litellm_extra={},
            window_min=1,
            window_max=10,
            preempt_policy="wait",
            binding=AgentBinding(agent_id="agt_1", version=1),
        )

        class _Prelude:
            system_prompt = "sys"
            tools: ClassVar[list[Any]] = []
            skill_versions: ClassVar[list[Any]] = []
            obligations: ClassVar[list[Any]] = []
            context_variables: ClassVar[list[Any]] = []

        class _Session:
            id = "sess_gate"
            focal_channel = None

        pool = MagicMock()

        monkeypatch.setattr(
            "aios.services.sessions.load_session_workspace_path",
            AsyncMock(return_value=None),
        )
        monkeypatch.setattr(
            "aios.services.accounts.resolve_effective_timezone",
            AsyncMock(return_value="UTC"),
        )

        await compose_step_context(
            pool=pool,
            session=_Session(),  # type: ignore[arg-type]
            account_id="acct_1",
            agent=agent,
            channels=[],
            prelude=_Prelude(),  # type: ignore[arg-type]
            events=[],
            persist_image_rewrites=False,
        )
        persist_spy.assert_not_called()

        await compose_step_context(
            pool=pool,
            session=_Session(),  # type: ignore[arg-type]
            account_id="acct_1",
            agent=agent,
            channels=[],
            prelude=_Prelude(),  # type: ignore[arg-type]
            events=[],
            persist_image_rewrites=True,
        )
        persist_spy.assert_called_once()


class TestBuildMessagesOffloadedFromLoop:
    @pytest.mark.asyncio
    async def test_build_messages_runs_in_thread(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``compose_step_context`` runs ``build_messages`` via
        ``asyncio.to_thread`` — assert the sync function executes on a
        different thread than the event loop's."""
        import threading

        from aios.harness.step_context import compose_step_context

        loop_thread_id = threading.get_ident()
        seen_thread_id: dict[str, int] = {}

        orig_build_messages = context.build_messages

        def spy_build_messages(*args: Any, **kwargs: Any) -> Any:
            seen_thread_id["id"] = threading.get_ident()
            return orig_build_messages(*args, **kwargs)

        monkeypatch.setattr("aios.harness.step_context.build_messages", spy_build_messages)

        from aios.models.agents import AgentBinding, StepSurface

        agent = StepSurface(
            model="model/vision",
            system="sys",
            tools=[],
            skills=[],
            mcp_servers=[],
            http_servers=[],
            litellm_extra={},
            window_min=1,
            window_max=10,
            preempt_policy="wait",
            binding=AgentBinding(agent_id="agt_1", version=1),
        )

        class _Prelude:
            system_prompt = "sys"
            tools: ClassVar[list[Any]] = []
            skill_versions: ClassVar[list[Any]] = []
            obligations: ClassVar[list[Any]] = []
            context_variables: ClassVar[list[Any]] = []

        class _Session:
            id = "sess_thread"
            focal_channel = None

        pool = MagicMock()

        async def fake_load_workspace(*args: Any, **kwargs: Any) -> None:
            return None

        async def fake_tz(*args: Any, **kwargs: Any) -> str:
            return "UTC"

        monkeypatch.setattr(
            "aios.services.sessions.load_session_workspace_path",
            AsyncMock(side_effect=fake_load_workspace),
        )
        monkeypatch.setattr(
            "aios.services.accounts.resolve_effective_timezone",
            AsyncMock(side_effect=fake_tz),
        )

        await compose_step_context(
            pool=pool,
            session=_Session(),  # type: ignore[arg-type]
            account_id="acct_1",
            agent=agent,
            channels=[],
            prelude=_Prelude(),  # type: ignore[arg-type]
            events=[],
            persist_image_rewrites=False,
        )
        assert seen_thread_id["id"] != loop_thread_id
