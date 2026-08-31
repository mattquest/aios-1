"""Worker-scoped MCP session pool — per-call session checkout.

Holds initialized ``ClientSession`` instances per ``(url, vault_id, headers_key)``
key (``headers_key`` hashes only the static spec headers, so the key is stable
across OAuth token rotation — see :meth:`acquire` and #459) so tool
discovery and execution can reuse an already-initialized MCP connection
instead of opening a fresh one on every call.

Checkout model: each tool call :meth:`acquire`\\ s an entry for **exclusive**
use and :meth:`release`\\ s (healthy) or :meth:`discard`\\ s (broken) it when
done. Because an entry — and its :class:`HttpErrorSink` — is never shared
while checked out, two concurrent calls to the same server can't cross-attribute
each other's HTTP errors (the bug a single shared session caused). Released
entries return to a per-key idle pool and are reused on the next acquire (warm
reuse); a fresh session is opened only when none are idle.

Pool lifecycle:

- Created at worker startup, stashed on :mod:`aios.harness.runtime`.
- :meth:`acquire` pops an idle entry or opens one. At most
  ``MAX_SESSIONS_PER_KEY`` live sessions exist per key — beyond the cap the
  caller waits on a per-key :class:`asyncio.Condition` until a release/discard
  frees a slot. The Condition is held across the open so the cap is strict.
- :meth:`release` returns a healthy session to idle; :meth:`discard` drops a
  broken one (fire-and-forget close). A benign HTTP error (e.g. 403) releases —
  the transport is fine; only transport failures discard.
- The idle reaper closes sessions idle longer than the TTL (idle entries only —
  an in-use entry is by definition active).
- :meth:`close_all` is called from ``worker_main``'s ``finally`` at shutdown and
  closes **both** idle and in-use sessions (in-use owner tasks must receive the
  shutdown signal or they leak).

Owner-task model (#425): each entry's ``AsyncExitStack`` lives inside a
dedicated long-lived task. The task opens the contexts, signals the
caller they're ready, then awaits a shutdown event. ``close()`` sets the
event and the contexts exit in the same task that entered them. This
avoids the anyio "Attempted to exit cancel scope in a different task
than it was entered in" error that would otherwise fire on
:meth:`close_all` because the streamable-http client binds its cancel
scope to the opening task.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

import httpx
from mcp.client.session import ClientSession, MessageHandlerFnT
from mcp.client.streamable_http import streamable_http_client
from mcp.types import InitializeResult

from aios.config import get_settings
from aios.logging import get_logger
from aios.mcp._constants import _MCP_HTTPX_TIMEOUT as _MCP_HTTPX_TIMEOUT_SHARED
from aios.pinned_transport import PinnedTransport

log = get_logger("aios.mcp.pool")

# Consecutive acquire/discovery failures per ``_PoolKey`` before the circuit
# breaker opens (#1698). Bulkheads a dead/401'ing MCP server: after K straight
# failures the key is marked unhealthy and the next prelude/call skips connect
# fast (degraded, not muted) until the ``_DISCOVERY_UNHEALTHY_BACKOFF_S``
# cooldown auto-re-probes. A single healthy acquire resets the counter.
_BREAKER_FAILURE_THRESHOLD = 3

# Cooldown applied when the breaker opens (#1698). Matches the discovery backoff
# (``client._DISCOVERY_UNHEALTHY_BACKOFF_S`` = 60s) so discovery and call-time
# share one window; kept here to avoid a client→pool import cycle.
_BREAKER_UNHEALTHY_BACKOFF_S = 60.0


class MCPUnavailable(Exception):
    """A pooled MCP transport failed to connect/initialize (#1698).

    Raised by :meth:`McpSessionPool.acquire` (and the discovery/call paths that
    build on it) as the *single* typed failure the pool ever surfaces on a
    connect/init error. It carries the server ``url`` and chains the original
    cause via ``__cause__`` / ``__context__``.

    Motivation: anyio's ``streamable_http_client`` task group, cancelling its
    SSE-reader sibling while unwinding a failed ``initialize()``, can raise a
    bare ``BaseExceptionGroup("unhandled errors in a TaskGroup", [HTTPError,
    CancelledError])``. A ``BaseExceptionGroup`` whose leaves include a
    non-``Exception`` (``CancelledError``) is itself **not** an ``Exception``
    (``isinstance(grp, Exception) is False``), so it slips past every
    ``except Exception`` on the call path and propagates unhandled — muting the
    whole session. Converting it to this ``Exception`` subclass at the pool
    boundary means every existing upstream ``except Exception`` catches it: a
    failing server degrades *capability* (its tools vanish), never
    *availability* (the agent goes mute).
    """

    def __init__(self, url: str) -> None:
        super().__init__(f"MCP server unavailable: {url}")
        self.url = url


# httpx client bounds for MCP transport — single definition in
# :mod:`aios.mcp._constants`, shared with the per-call client transport
# (re-exported here so existing ``pool._MCP_HTTPX_TIMEOUT`` references and their
# test patches keep resolving).
_MCP_HTTPX_TIMEOUT = _MCP_HTTPX_TIMEOUT_SHARED

# Cap on live sessions per ``(url, vault_id, headers_key)``. Bounds session/connection growth
# under a model that fires many concurrent calls at one server; at the cap an
# ``acquire`` waits for a release rather than opening unboundedly. A session is
# created only when in-use < cap and none are idle, so the total live count never
# exceeds the peak concurrent in-use count, which is bounded by this value.
MAX_SESSIONS_PER_KEY = 8

type _PoolKey = tuple[str, str | None, str]  # (url, vault_id, headers_key)


@dataclass
class HttpErrorSink:
    """Captures the most recent 4xx/5xx HTTP response seen on a pooled MCP
    transport.

    The streamable-http client raises the HTTP error inside a long-lived anyio
    task group that stays suspended while the pooled session is parked, so a
    failed ``call_tool`` would otherwise hang until the tool-call timeout and
    surface a bare ``TimeoutError`` — losing the server's actual message. This
    sink lets the caller observe the error response immediately (via ``event``)
    and report it (``status`` + ``body``).
    """

    event: asyncio.Event = field(default_factory=asyncio.Event)
    status: int | None = None
    body: str = ""

    def reset(self) -> None:
        self.event.clear()
        self.status = None
        self.body = ""

    def record(self, status: int, body: str) -> None:
        self.status = status
        self.body = body
        self.event.set()


def _make_error_hook(sink: HttpErrorSink) -> Callable[[httpx.Response], Awaitable[None]]:
    """httpx response hook that records error responses into ``sink``.

    Only reads the body for 4xx/5xx (error JSON, never an SSE stream), so
    success-path streaming responses are untouched.
    """

    async def _hook(response: httpx.Response) -> None:
        if response.status_code < 400 or sink.event.is_set():
            return
        try:
            await response.aread()
            body = response.text[:800]
        except Exception:
            body = ""
        sink.record(response.status_code, body)

    return _hook


class _Entry:
    """A single pooled MCP session with its associated owner task.

    The session's contexts (httpx client, streamable-http client,
    ClientSession) all live inside ``_owner_task``. ``close()`` sets
    ``_shutdown`` so the task exits the contexts in the same task that
    entered them — see the module docstring for the anyio cancel-scope
    background.
    """

    def __init__(
        self,
        session: ClientSession,
        init_result: InitializeResult,
        shutdown: asyncio.Event,
        owner_task: asyncio.Task[None],
        last_used: float,
        error_sink: HttpErrorSink,
    ) -> None:
        self.session = session
        self.init_result = init_result
        self._shutdown = shutdown
        self._owner_task = owner_task
        # Records error responses on this session's transport so a failed
        # call_tool can fail fast with the server's message (see HttpErrorSink).
        self.error_sink = error_sink
        # monotonic() stamped each time this entry is released back to idle.
        # The reaper keys off this, NOT owner-task liveness: the task parks on
        # `await shutdown.wait()` for the entry's whole life, so liveness is a
        # useless staleness signal. Only idle entries are reaped (an in-use
        # entry is by definition active), so last_used is the time the entry
        # last finished a checkout.
        #
        # Post-#459 the pool keys on (url, vault_id, headers_key), stable
        # across OAuth refresh (headers_key hashes only the static spec
        # headers). The reaper's remaining job is the cold-entry
        # vector (a vault whose tenant never returns) — defense-in-depth
        # signed off in #459's planning round, not silent accretion.
        self.last_used = last_used
        # Set by :meth:`McpSessionPool.evict_by_vault` when an operator rotates
        # the vault credential while this entry is checked out. The entry's live
        # ClientSession baked in the OLD bearer at open time, so it must not
        # return to idle for reuse: :meth:`release` discards it instead (#1030).
        self.discard_on_release = False

    async def close(self) -> None:
        """Signal the owner task to exit its contexts and await its completion.

        Safe to call repeatedly; the owner task only sees the first set().

        The owner task's contexts may raise during exit (e.g. the remote
        MCP server already closed the socket). The caller —
        :meth:`McpSessionPool.close_all` — already swallows and logs such
        errors; we surface as if shutdown completed.
        """
        self._shutdown.set()
        with contextlib.suppress(Exception):
            await self._owner_task


class McpSessionPool:
    """Worker-scoped pool of MCP ``ClientSession`` instances, checked out per call.

    Single-event-loop — no thread-safety concerns. A per-key ``asyncio.Condition``
    guards the idle/in-use sets and blocks acquires at the per-key cap.
    """

    def __init__(self) -> None:
        # Idle (available) and in-use (checked out) entries per key. The cap is
        # on in-use count; idle entries are reused before a new one is opened.
        self._idle: dict[_PoolKey, list[_Entry]] = {}
        self._in_use: dict[_PoolKey, set[_Entry]] = {}
        self._conditions: dict[_PoolKey, asyncio.Condition] = {}
        self._closed = False
        self._reaper_task: asyncio.Task[None] | None = None
        # Strong refs for discard()'s fire-and-forget close tasks. asyncio
        # only weak-refs tasks, so without this the task can be GC'd
        # before close() unwinds the owner task's contexts — defeating
        # the leak fix.
        self._close_tasks: set[asyncio.Task[None]] = set()
        # Discovered-tool-list result cache (#1391). Keyed on
        # ``(_PoolKey, binding_id)`` where ``binding_id`` is the caller's
        # binding identity (agent id + version, or the frozen-surface hash) —
        # the precondition that makes the tool set static across steps. Caches
        # the FULL ``(tools, instructions)`` discovery result so the per-step
        # prelude no longer re-pays a ``list_tools()`` RPC every turn ("connection
        # reuse" was never "result reuse"). Invalidated on a binding-identity
        # change (a new key never hits) or a ``tools/list_changed`` notification
        # (which drops every binding's entry for the affected _PoolKey).
        self._tool_cache: dict[tuple[_PoolKey, str], tuple[list[Any], str | None]] = {}
        # Per-_PoolKey circuit breaker (#1391). When discovery times out or the
        # transport fails, the key is marked unhealthy until ``unhealthy_until``
        # (monotonic seconds) so the next prelude SKIPS it fast instead of
        # re-stalling for the per-server timeout every step. A successful
        # discovery clears the entry.
        self._unhealthy_until: dict[_PoolKey, float] = {}
        # Per-_PoolKey consecutive-failure counter feeding the breaker (#1698).
        # Every ``acquire``/discovery connect-or-init failure bumps the count;
        # the breaker opens once it reaches ``_BREAKER_FAILURE_THRESHOLD`` (K).
        # A successful ``acquire`` resets it to 0 at the pool boundary (the
        # fresh-open success path — call-time success is covered because every
        # ``call_mcp_tool`` acquires before use); ``mark_healthy`` (the
        # uncached-discovery success) and ``_clear_breaker_by_vault`` (credential
        # rotation) also reset it. So the count is truly *consecutive*: any
        # intervening success breaks the streak.
        self._failure_count: dict[_PoolKey, int] = {}
        # URLs whose breaker is currently OPEN, keyed by url. Backs
        # :meth:`degraded_servers` and dedupes the ``mcp_server_unavailable``
        # observability event to the DOWN transition (the breaker's open edge),
        # not every failure while it stays open (#1698).
        self._degraded_urls: set[str] = set()
        # DOWN-transition edges (URLs) recorded since the last drain. The
        # breaker arms in ``acquire``/discovery — code with no session context —
        # so it records the edge here; a caller WITH session context
        # (``discover_session_mcp_tools``) drains this via
        # :meth:`drain_degraded_events` and emits exactly one durable
        # ``mcp_server_unavailable`` session event per edge (#1698 (e)).
        self._pending_degraded_events: list[str] = []

    def _condition_for(self, key: _PoolKey) -> asyncio.Condition:
        cond = self._conditions.get(key)
        if cond is None:
            cond = asyncio.Condition()
            self._conditions[key] = cond
        return cond

    def _drop_in_use(self, key: _PoolKey, entry: _Entry) -> None:
        """Remove ``entry`` from the in-use set, deleting the set once it empties.

        Mirrors acquire's idle-list cleanup so a key whose last in-flight entry
        is released/discarded doesn't leave a ghost empty set behind. Caller
        holds the key's Condition.
        """
        in_use = self._in_use.get(key)
        if in_use is None:
            return
        in_use.discard(entry)
        if not in_use:
            del self._in_use[key]

    def _schedule_background_close(self, entry: _Entry, url: str) -> None:
        """Fire-and-forget ``entry.close()`` tracked in ``_close_tasks``.

        asyncio only weak-refs tasks, so the strong ref keeps the close from
        being GC'd before it unwinds the owner task's contexts (the leak the
        idle reaper exists to prevent). Shared by :meth:`discard` and the
        discard path of :meth:`release`.
        """
        task = asyncio.create_task(entry.close(), name=f"mcp-pool-discard:{url}")
        self._close_tasks.add(task)
        task.add_done_callback(self._close_tasks.discard)

    def _make_list_changed_handler(self, key: _PoolKey) -> MessageHandlerFnT:
        """Build a ``ClientSession`` message handler that invalidates the tool
        cache on a ``tools/list_changed`` notification (#1391).

        The MCP spec's ``notifications/tools/list_changed`` is the protocol's
        own cache-invalidation signal — previously unhandled, so aios ignored it
        and unconditionally re-polled. Registering this handler lets a server
        push an updated tool set: the cached ``(tools, instructions)`` for every
        binding on this transport key is dropped, and the next prelude
        re-discovers. Non-notification messages (server requests, exceptions)
        are passed through unchanged.
        """
        from mcp.types import ServerNotification, ToolListChangedNotification

        async def _handler(message: Any) -> None:
            if isinstance(message, ServerNotification) and isinstance(
                message.root, ToolListChangedNotification
            ):
                self._invalidate_tools_for_pool_key(key)

        return _handler

    async def _open_entry(
        self, url: str, headers: dict[str, str], key: _PoolKey | None = None
    ) -> _Entry:
        """Open a fresh session inside a dedicated long-lived owner task.

        The contexts (httpx client, streamable-http client, ClientSession)
        all enter and exit in the same task — see the module docstring on
        the anyio cancel-scope constraint. The task signals ``ready`` once
        the session is initialized, then awaits ``shutdown`` so the
        contexts stay open for the entry's lifetime.

        When ``key`` is supplied, a ``tools/list_changed`` message handler is
        registered on the session so the server's own invalidation signal drops
        the cached tool list for that transport key (#1391).
        """
        ready = asyncio.Event()
        shutdown = asyncio.Event()
        sink = HttpErrorSink()
        result: dict[str, Any] = {}
        message_handler = self._make_list_changed_handler(key) if key is not None else None

        async def _own() -> None:
            try:
                async with AsyncExitStack() as stack:
                    # PinnedTransport resolves-validates-pins the connect IP per
                    # request: a tenant-supplied MCP URL (or a later DNS rebind
                    # of it) that resolves to a private/internal address is
                    # refused at the transport. The operator dev bypass set is
                    # the same one the OAuth connect flow honors.
                    http_client: Any = await stack.enter_async_context(
                        httpx.AsyncClient(
                            headers=headers,
                            timeout=_MCP_HTTPX_TIMEOUT,
                            event_hooks={"response": [_make_error_hook(sink)]},
                            transport=PinnedTransport(
                                allow_hosts=get_settings().oauth_allow_insecure_host_set
                            ),
                        )
                    )
                    read_stream, write_stream, _ = await stack.enter_async_context(
                        streamable_http_client(url, http_client=http_client)
                    )
                    session = await stack.enter_async_context(
                        ClientSession(read_stream, write_stream, message_handler=message_handler)
                    )
                    result["session"] = session
                    result["init"] = await session.initialize()
                    ready.set()
                    await shutdown.wait()
            except BaseException as exc:
                result["error"] = exc
                # Unblock the caller waiting on ready so it doesn't hang
                # forever when the open path failed.
                ready.set()
                raise

        task = asyncio.create_task(_own(), name=f"mcp-pool:{url}")
        await ready.wait()
        if "error" in result:
            # The task already raised; await it to clear the unhandled-exception
            # slot on the owner task. We do NOT let that await re-raise the raw
            # failure here: anyio's ``streamable_http_client`` task group can
            # unwind a failed ``initialize()`` as a bare ``BaseExceptionGroup``
            # (leaves ``[HTTPError, CancelledError]``), which is not an
            # ``Exception`` and would escape every ``except Exception`` upstream
            # (the #1698 session-mute bug). Swallow the await's propagation and
            # raise the captured cause as :class:`MCPUnavailable` (an
            # ``Exception``) instead — ``acquire`` re-wraps into the same type,
            # so the pool NEVER re-raises a bare group. ``CancelledError`` is a
            # ``BaseException`` (not caught here), so a genuine cancellation of
            # the acquiring task still propagates for prompt teardown.
            with contextlib.suppress(BaseException):
                await task
            captured = result["error"]
            if isinstance(captured, asyncio.CancelledError):
                # A genuine cancellation of the owner task (shutdown/drain) is a
                # BaseException, not a transport failure — propagate it as-is so
                # cancellation semantics are preserved rather than laundered into
                # a retriable MCPUnavailable.
                raise captured
            raise MCPUnavailable(url) from captured
        return _Entry(result["session"], result["init"], shutdown, task, time.monotonic(), sink)

    async def acquire(
        self, url: str, vault_id: str | None, headers_key: str, headers: dict[str, str]
    ) -> _Entry:
        """Check out an entry for **exclusive** use; caller must release/discard it.

        Keyed on ``(url, vault_id, headers_key)``. ``vault_id`` keeps the key
        stable across OAuth token rotation because it is the row id of the
        ``vault_credentials`` entry that ``refresh_credential`` updates in
        place (see #459). ``headers_key`` hashes ONLY the spec's static
        config headers (never the merged auth headers), so token rotation
        does not change it either — preserving #459. ``vault_id`` is ``None``
        for the no-credential case (an unauthenticated MCP server); all such
        callers on the same URL with the same static headers share one key,
        which is safe because no auth identity is keyed on.

        Reuses an idle entry if one exists; otherwise opens a fresh one when
        under the per-key cap; otherwise waits on the per-key Condition for a
        release/discard to free a slot. The Condition is held across the open so
        the cap is strict (no thundering-herd over-open). An ``_open_entry``
        failure propagates out, releasing the Condition; no slot is consumed and
        no waiter is notified — the next waiter to wake simply retries.

        **Fault isolation (#1698).** A fresh-open failure is converted at this
        boundary into :class:`MCPUnavailable` — an ``Exception`` subclass — so a
        bare ``BaseExceptionGroup`` from anyio's transport task group can NEVER
        escape the pool. Every upstream ``except Exception`` therefore catches
        it, and the failing server degrades *capability* (its tools vanish), not
        *availability* (the agent goes mute). The conversion also arms the
        per-key circuit breaker (K consecutive failures → open).
        """
        if self._closed:
            raise RuntimeError("McpSessionPool is closed")
        key: _PoolKey = (url, vault_id, headers_key)
        cond = self._condition_for(key)
        async with cond:
            while True:
                idle = self._idle.get(key)
                if idle:
                    entry = idle.pop()
                    if not idle:
                        del self._idle[key]
                    self._in_use.setdefault(key, set()).add(entry)
                    return entry
                if len(self._in_use.get(key, set())) < MAX_SESSIONS_PER_KEY:
                    log.info("mcp_pool.connecting", url=url)
                    try:
                        entry = await self._open_entry(url, headers, key)
                    except MCPUnavailable:
                        # Already the typed boundary error (from _open_entry's
                        # own group-guard). Record the failure for the breaker
                        # and re-raise unchanged — no double-wrapping.
                        self._record_connect_failure(key, url)
                        raise
                    except (Exception, BaseExceptionGroup) as exc:
                        # Any residual connect/init failure — including a bare
                        # ``BaseExceptionGroup`` whose non-Exception leaf makes
                        # it slip past ``except Exception`` — is converted here
                        # so the pool's ONLY connect-failure surface is
                        # ``MCPUnavailable`` (an Exception). ``BaseException``
                        # (e.g. a genuine ``CancelledError`` cancelling this
                        # acquire) is deliberately NOT caught: it propagates for
                        # prompt teardown.
                        self._record_connect_failure(key, url)
                        raise MCPUnavailable(url) from exc
                    # This connect succeeded, so the consecutive-failure streak is
                    # broken — reset the per-key breaker counter (F1 / #1698 AC4:
                    # the count is *consecutive*, not cumulative). This is the
                    # single call-time reset site at the pool boundary; every
                    # ``call_mcp_tool`` acquires before use, so a successful call
                    # resets it here too (no separate call-path reset needed). An
                    # open breaker is skipped before ``acquire`` via
                    # ``is_unhealthy``, so a reset only follows a genuinely healthy
                    # connect.
                    self._failure_count.pop(key, None)
                    self._in_use.setdefault(key, set()).add(entry)
                    log.info("mcp_pool.connected", url=url)
                    return entry
                await cond.wait()

    async def release(
        self, url: str, vault_id: str | None, headers_key: str, entry: _Entry
    ) -> None:
        """Return a healthy entry to the idle pool for reuse.

        Called when the session is fine — a successful call, or a benign
        application-level HTTP error (the transport is unaffected). Stamps
        ``last_used`` (the reaper's staleness signal) and notifies one waiter
        that a slot is free.

        Exception: if :meth:`evict_by_vault` flagged this entry (an operator
        rotated the vault credential), the entry's live session still carries
        the OLD bearer, so it is DISCARDED here instead of being returned to
        idle — the next acquire then opens a fresh session on the rotated
        secret (#1030). The flag is read UNDER the key Condition because
        ``evict_by_vault`` sets it while holding that same lock; an unlocked
        check could pass and then race a concurrent eviction into the gap
        before the lock, returning the stale-token session to idle for reuse.
        """
        entry.last_used = time.monotonic()
        key: _PoolKey = (url, vault_id, headers_key)
        cond = self._condition_for(key)
        async with cond:
            self._drop_in_use(key, entry)
            # Re-read under the lock: serialize evict_by_vault's flag-set with
            # this idle-vs-discard decision so a flag flipped while release()
            # waited on the lock is observed (closes the #1030 reuse race).
            discard = entry.discard_on_release
            if not discard:
                self._idle.setdefault(key, []).append(entry)
            cond.notify()
        if discard:
            self._schedule_background_close(entry, url)
            log.info("mcp_pool.discarded", url=url)

    async def discard(
        self, url: str, vault_id: str | None, headers_key: str, entry: _Entry
    ) -> None:
        """Drop a broken entry and close it in the background.

        Called when the session may be broken (transport failure, cancellation).
        The caller can't block on close (the owner task may wait up to the httpx
        read timeout before unwinding), so the close is fire-and-forget. Skipping
        close entirely would strand the owner task + httpx client + SSE stream —
        the same leak the idle reaper exists to prevent. Frees the in-use slot
        and notifies one waiter.
        """
        key: _PoolKey = (url, vault_id, headers_key)
        cond = self._condition_for(key)
        async with cond:
            self._drop_in_use(key, entry)
            cond.notify()
        self._schedule_background_close(entry, url)
        log.info("mcp_pool.discarded", url=url)

    async def evict_by_vault(self, vault_id: str) -> None:
        """Discard every pooled session keyed on ``vault_id`` — an operator
        rotated (PUT/archive/delete) this vault's credential (#1030).

        The resolved bearer is baked into each live ``ClientSession`` at open
        time and the pool key is stable across token changes (by design, for
        OAuth-refresh reuse), so a rotated ``secret_value`` would otherwise keep
        serving the OLD token until the 900s idle reaper — which active polling
        defeats. This is the explicit eviction signal that closes that gap.

        Idle entries are closed and dropped immediately. In-use (checked-out)
        entries can't be torn down under their owner — closing mid-call would
        cancel the in-flight transport — so they are FLAGGED
        (``discard_on_release``); :meth:`release` then discards instead of
        returning them to idle. Either way no stale-token session survives to a
        later acquire.

        Fired cross-process: the credential mutation runs in the API process,
        the pool lives in the worker, so the worker's evict-listener invokes
        this on a NOTIFY (see ``MCP_EVICT_VAULT_CHANNEL`` /
        ``_run_mcp_evict_listener``). Unconditional — no config knob.
        """
        idle_closed = 0
        flagged = 0
        # Snapshot the matching idle entries under each key's Condition, then
        # close them outside the lock (close awaits the owner task, which the
        # Condition holder must not block on). Mirrors the reaper's TOCTOU
        # shape: re-check membership under the lock before removing.
        for key in [k for k in self._idle if k[1] == vault_id]:
            cond = self._condition_for(key)
            to_close: list[_Entry] = []
            async with cond:
                idle = self._idle.get(key)
                if idle is None:
                    continue
                to_close = list(idle)
                del self._idle[key]
            for entry in to_close:
                idle_closed += 1
                await entry.close()
        # Flag in-use entries so their release discards them.
        for key in [k for k in self._in_use if k[1] == vault_id]:
            cond = self._condition_for(key)
            async with cond:
                for entry in self._in_use.get(key, set()):
                    entry.discard_on_release = True
                    flagged += 1
        # Drop any cached tool lists discovered under the rotated identity so a
        # re-auth re-discovers rather than serving the stale-identity set (#1391).
        self.invalidate_tools_by_vault(vault_id)
        # Clear breaker state for every key bound to this vault so a fresh
        # credential re-probes immediately rather than sitting out the cooldown
        # from failures under the OLD (e.g. expired) secret (#1698 composes with
        # #1030). A stale-token 401 that opened the breaker must not keep the
        # newly-rotated secret's server dark.
        self._clear_breaker_by_vault(vault_id)
        if idle_closed or flagged:
            log.info(
                "mcp_pool.evict_by_vault",
                vault_id=vault_id,
                idle_closed=idle_closed,
                in_use_flagged=flagged,
            )

    # ── tool-list result cache (#1391) ────────────────────────────────────

    def get_cached_tools(
        self, url: str, vault_id: str | None, headers_key: str, binding_id: str
    ) -> tuple[list[Any], str | None] | None:
        """Return the cached ``(tools, instructions)`` for this binding, or ``None``.

        The cache key folds the transport identity (``_PoolKey``) together with
        the caller's ``binding_id`` (agent id + version, or frozen-surface hash).
        A binding-identity bump (agent-version change) therefore lands on a fresh
        key and misses — propagating the new tool set to the next step with no
        explicit invalidation (#1391 staleness guard).
        """
        key: _PoolKey = (url, vault_id, headers_key)
        return self._tool_cache.get((key, binding_id))

    def set_cached_tools(
        self,
        url: str,
        vault_id: str | None,
        headers_key: str,
        binding_id: str,
        tools: list[Any],
        instructions: str | None,
    ) -> None:
        """Cache a successful ``(tools, instructions)`` discovery for this binding."""
        key: _PoolKey = (url, vault_id, headers_key)
        self._tool_cache[(key, binding_id)] = (tools, instructions)

    def lookup_cached_tool_parameters(
        self, qualified_name: str, *, url: str | None = None
    ) -> dict[str, Any] | None:
        """Return sanitized ``function.parameters`` for ``qualified_name``, or ``None``.

        Walks the discovery result cache (#1391) — the same OpenAI-shaped tool
        dicts advertised to the model after ``make_function_tool`` /
        ``sanitize_mcp_schema``. ``url`` restricts the scan to that server's
        transport keys so two agents with similarly named servers don't
        cross-read schemas.

        If multiple matching cache entries advertise different schemas for the
        same qualified name, this is ambiguous by construction (same server URL,
        different auth/binding context). In that case the lookup fails open:
        warn and return ``None`` so callers treat it as a cache miss and skip
        local validation instead of picking an arbitrary schema.
        """
        candidates: list[dict[str, Any]] = []
        for (pool_key, _binding_id), (tools, _instructions) in self._tool_cache.items():
            if url is not None and pool_key[0] != url:
                continue
            for td in tools:
                if not isinstance(td, dict):
                    continue
                function = td.get("function")
                if not isinstance(function, dict):
                    continue
                if function.get("name") != qualified_name:
                    continue
                parameters = function.get("parameters")
                if isinstance(parameters, dict):
                    candidates.append(parameters)

        if not candidates:
            return None
        first = candidates[0]
        if any(parameters != first for parameters in candidates[1:]):
            log.warning(
                "mcp_pool.tool_parameters_ambiguous",
                qualified_name=qualified_name,
                url=url,
                candidate_count=len(candidates),
            )
            return None
        return first

    def _invalidate_tools_for_pool_key(self, key: _PoolKey) -> None:
        """Drop every binding's cached tool list for one transport key.

        Fired by the ``tools/list_changed`` notification handler: the server
        announced its tool set changed, so every binding that discovered against
        this ``(url, vault_id, headers_key)`` must re-discover on its next step.
        """
        stale = [ck for ck in self._tool_cache if ck[0] == key]
        for ck in stale:
            del self._tool_cache[ck]
        if stale:
            log.info("mcp_pool.tools_cache_invalidated", url=key[0], dropped=len(stale))

    def invalidate_tools_by_vault(self, vault_id: str) -> None:
        """Drop cached tool lists for every key bound to ``vault_id``.

        Called alongside :meth:`evict_by_vault` on a credential rotation so a
        re-auth can't keep serving a tool list discovered under the old identity.
        """
        stale = [ck for ck in self._tool_cache if ck[0][1] == vault_id]
        for ck in stale:
            del self._tool_cache[ck]

    # ── per-key discovery/connect circuit breaker (#1391 + #1698) ─────────

    def is_unhealthy(
        self, url: str, vault_id: str | None, headers_key: str, *, now: float | None = None
    ) -> bool:
        """True if this transport key is in a discovery/connect backoff window.

        The prelude AND ``call_mcp_tool`` consult this BEFORE attempting a
        connect (``list_tools()`` / ``acquire``) so a server that just failed is
        skipped fast (agent runs degraded on the other servers' tools) instead
        of re-stalling every step. On expiry the window is cleared here so the
        next probe re-attempts the connect (auto-re-probe); the DOWN marker in
        ``_degraded_urls`` is retained until an explicit heal/failure edge so a
        cooldown lapse alone doesn't spuriously fire an UP event.
        """
        key: _PoolKey = (url, vault_id, headers_key)
        until = self._unhealthy_until.get(key)
        if until is None:
            return False
        if (now if now is not None else time.monotonic()) >= until:
            del self._unhealthy_until[key]
            return False
        return True

    def _record_connect_failure(self, key: _PoolKey, url: str) -> bool:
        """Count one consecutive connect/init failure; open the breaker at K.

        Returns ``True`` on the DOWN transition (the breaker's open edge) — the
        failure that first opens the circuit — so a caller with session context
        can emit exactly one ``mcp_server_unavailable`` event (#1698 (e)).
        Failures while already open return ``False`` (deduped on the edge).

        Shared by ``acquire``'s new failure path and (via
        :meth:`note_discovery_failure`) the discovery caller, so discovery and
        call-time arm ONE breaker per key.
        """
        count = self._failure_count.get(key, 0) + 1
        self._failure_count[key] = count
        if count < _BREAKER_FAILURE_THRESHOLD:
            return False
        # At/over threshold: (re)open the circuit for the cooldown window.
        self._unhealthy_until[key] = time.monotonic() + _BREAKER_UNHEALTHY_BACKOFF_S
        transitioned_down = url not in self._degraded_urls
        self._degraded_urls.add(url)
        if transitioned_down:
            # Record the edge for a session-context caller to emit exactly one
            # durable ``mcp_server_unavailable`` event (#1698 (e)).
            self._pending_degraded_events.append(url)
            log.warning(
                "mcp_pool.breaker_opened",
                url=url,
                failures=count,
                backoff_s=_BREAKER_UNHEALTHY_BACKOFF_S,
            )
        return transitioned_down

    def drain_degraded_events(self) -> list[str]:
        """Pop and return the DOWN-transition URLs recorded since the last drain.

        A session-context caller (the per-turn discovery prelude) calls this and
        emits one durable ``mcp_server_unavailable`` session event per URL, so
        each breaker DOWN edge surfaces exactly once regardless of which path
        (connect vs discovery) armed it (#1698 (e)).
        """
        drained = self._pending_degraded_events
        self._pending_degraded_events = []
        return drained

    def note_discovery_failure(self, url: str, vault_id: str | None, headers_key: str) -> bool:
        """Record a discovery-path failure against the shared breaker (#1698).

        The discovery caller (``discover_mcp_tools``) sees ``list_tools()``
        failures that ``acquire`` itself can't (the connect succeeded but the RPC
        failed), so it feeds them into the SAME per-key counter. Returns the
        DOWN-transition flag like :meth:`_record_connect_failure`.
        """
        key: _PoolKey = (url, vault_id, headers_key)
        return self._record_connect_failure(key, url)

    def mark_unhealthy(
        self,
        url: str,
        vault_id: str | None,
        headers_key: str,
        *,
        backoff_s: float,
        now: float | None = None,
    ) -> None:
        """Open the circuit for this transport key for ``backoff_s`` seconds.

        Retained for the discovery retry path that forces the window open
        directly. Also marks the URL degraded so it surfaces in
        :meth:`degraded_servers`; the DOWN-transition event is emitted off the
        counter path (:meth:`note_discovery_failure`), so this returns nothing.
        """
        key: _PoolKey = (url, vault_id, headers_key)
        self._unhealthy_until[key] = (now if now is not None else time.monotonic()) + backoff_s
        self._degraded_urls.add(url)
        log.warning("mcp_pool.marked_unhealthy", url=url, backoff_s=backoff_s)

    def mark_healthy(self, url: str, vault_id: str | None, headers_key: str) -> None:
        """Clear backoff + failure counter for this key (a successful connect).

        Resets the consecutive-failure counter so the breaker re-arms from zero,
        clears any open window, and drops the URL from ``_degraded_urls`` unless
        another key on the same URL is still open — a successful re-probe
        re-closes the circuit (AC #4).
        """
        key: _PoolKey = (url, vault_id, headers_key)
        self._unhealthy_until.pop(key, None)
        self._failure_count.pop(key, None)
        if not self._url_has_open_breaker(url):
            self._degraded_urls.discard(url)

    def _url_has_open_breaker(self, url: str) -> bool:
        """True if any key on ``url`` still has an unexpired backoff window."""
        now = time.monotonic()
        return any(k[0] == url and until > now for k, until in self._unhealthy_until.items())

    def _clear_breaker_by_vault(self, vault_id: str) -> None:
        """Drop breaker state (window + counter) for every key on ``vault_id``.

        Called from :meth:`evict_by_vault` so a rotated credential re-probes
        immediately instead of serving out a cooldown opened under the old
        secret (#1698 composes with #1030).
        """
        for key in [k for k in self._unhealthy_until if k[1] == vault_id]:
            del self._unhealthy_until[key]
        for key in [k for k in self._failure_count if k[1] == vault_id]:
            del self._failure_count[key]
        # A URL with no remaining open key on any vault leaves the degraded set.
        self._degraded_urls = {u for u in self._degraded_urls if self._url_has_open_breaker(u)}

    def degraded_servers(self) -> list[str]:
        """Return the URLs whose breaker is currently OPEN (#1698 (e)).

        This is an in-memory, per-worker accessor (reflecting only THIS worker's
        breaker state, not persisted) for tests and in-process introspection; the
        wired external artifact the ops-agent consumes is the durable
        ``mcp_server_unavailable`` session event emitted on the breaker's DOWN
        edge (see :meth:`drain_degraded_events`), not this accessor.
        """
        return sorted(self._degraded_urls)

    async def _reap_idle_once(self, *, idle_timeout: float, now: float) -> None:
        """Close idle entries unused longer than ``idle_timeout`` seconds.

        Idle entries only — an in-use (checked-out) entry is by definition
        active. Keyed on ``_Entry.last_used`` (stamped at :meth:`release`).
        Reclamation goes through ``_Entry.close()`` (same path as
        :meth:`close_all`), NOT a bare ``remove``: dropping the reference alone
        strands the owner task, httpx client and SSE stream — exactly the leak
        this reaps. No notify after close: reaping an idle entry doesn't free an
        in-use slot, so no capped waiter is unblocked by it.
        """
        if self._closed:
            return
        stale = [
            (key, entry)
            for key, entries in self._idle.items()
            for entry in entries
            if now - entry.last_used > idle_timeout
        ]
        for key, entry in stale:
            async with self._condition_for(key):
                # Re-check under the lock: acquire() may have popped this entry
                # (warm reuse) between the scan and here, or release() may have
                # refreshed its last_used — same TOCTOU shape as the
                # SandboxRegistry reaper fixed in #654.
                idle = self._idle.get(key)
                if idle is None or entry not in idle or now - entry.last_used <= idle_timeout:
                    continue
                idle.remove(entry)
                if not idle:
                    del self._idle[key]
            log.info("mcp_pool.idle_close", url=key[0])
            await entry.close()

    async def _reap_idle_loop(self, idle_timeout: float, interval: float = 60.0) -> None:
        """Background loop: close pooled sessions idle > idle_timeout.

        The try/except is nested INSIDE ``while True`` (mirroring
        :meth:`aios.sandbox.registry.SandboxRegistry._reap_idle_loop`
        and the PR #443 fix for ``_run_interrupt_listener``): a teardown
        error from one entry's ``close()`` must not silently disable the
        reaper for the worker's lifetime — that would leak every
        subsequently-superseded entry until process exit.
        ``CancelledError`` is not an :class:`Exception` subclass, so
        :meth:`stop_reaper`'s cancel still exits the loop cleanly.
        """
        while True:
            try:
                await asyncio.sleep(interval)
                await self._reap_idle_once(idle_timeout=idle_timeout, now=time.monotonic())
            except Exception:
                log.exception("mcp_pool.reap_idle_loop_failed")

    def start_reaper(self, idle_timeout: float) -> asyncio.Task[None]:
        """Start the idle-TTL reaper background task and return its handle."""
        if self._reaper_task is not None:
            return self._reaper_task
        self._reaper_task = asyncio.create_task(
            self._reap_idle_loop(idle_timeout),
            name="mcp-pool-idle-reaper",
        )
        return self._reaper_task

    def stop_reaper(self) -> None:
        """Cancel the idle-TTL reaper."""
        if self._reaper_task is not None:
            self._reaper_task.cancel()
            self._reaper_task = None

    async def close_all(self) -> None:
        """Tear down all sessions — idle AND in-use. Called at worker shutdown.

        In-use entries must be closed too: their owner tasks are parked on the
        shutdown event and would leak otherwise (setting it unwinds the
        AsyncExitStack, cancelling any in-flight transport task). ``_closed`` is
        set first so no new session is acquired during or after teardown.
        """
        self._closed = True
        entries = [e for entries in self._idle.values() for e in entries]
        entries += [e for entries in self._in_use.values() for e in entries]
        self._idle.clear()
        self._in_use.clear()
        self._conditions.clear()
        # Snapshot the in-flight background-close tasks (discard/evict path,
        # ``_schedule_background_close``) BEFORE awaiting: the done-callback
        # mutates ``_close_tasks`` on completion, so iterating it directly
        # would race. These and the idle/in-use snapshot are the complete set
        # of sessions that must be unwound before the worker exits — without
        # draining them a background close racing shutdown is abandoned with a
        # half-unwound owner task / httpx client / SSE stream.
        pending_closes = list(self._close_tasks)
        if not entries and not pending_closes:
            return
        log.info(
            "mcp_pool.close_all",
            count=len(entries),
            pending_closes=len(pending_closes),
        )
        for entry in entries:
            try:
                await entry.close()
            except Exception:
                log.warning("mcp_pool.close_entry_failed", exc_info=True)
        if pending_closes:
            # Let in-flight background closes finish unwinding (errors swallowed
            # — a failed background close must not abort shutdown).
            await asyncio.gather(*pending_closes, return_exceptions=True)
