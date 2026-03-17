"""APScheduler integration and SSE event broadcasting for Agent Agora.

This module is responsible for:

- Maintaining a registry of Server-Sent Events (SSE) subscriber queues so
  that any part of the application can push an event and all connected
  browsers receive it.
- Creating and managing an :class:`~apscheduler.schedulers.asyncio.AsyncIOScheduler`
  that ticks each active agent on a configurable interval.
- Exposing lifecycle helpers (``start_scheduler``, ``stop_scheduler``) that
  are called by the FastAPI app on startup/shutdown.
- Providing a ``broadcast`` coroutine and a ``subscribe`` / ``unsubscribe``
  API for the SSE endpoint.

The scheduler queries the database on every master tick to discover which
agents are currently active, then fires an individual async job for each one.
This means agents spawned or paused mid-simulation are picked up within one
tick interval without restarting the scheduler.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import AsyncIterator, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from agent_agora import database as db
from agent_agora.agent_runner import run_agent_tick
from agent_agora.models import AgentStatus, SSEEvent

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: How often (in seconds) the master scheduler tick fires.
DEFAULT_TICK_INTERVAL: int = int(os.getenv("AGENT_TICK_INTERVAL_SECONDS", "15"))

# ---------------------------------------------------------------------------
# SSE subscriber registry
# ---------------------------------------------------------------------------

# Each connected browser gets its own asyncio Queue placed in this set.
_subscribers: set[asyncio.Queue] = set()


def subscribe() -> asyncio.Queue:
    """Register a new SSE subscriber and return its event queue.

    The caller (SSE endpoint) should call :func:`unsubscribe` when the
    client disconnects to avoid memory leaks.

    Returns:
        A fresh :class:`asyncio.Queue` that will receive
        :class:`~agent_agora.models.SSEEvent` instances.
    """
    queue: asyncio.Queue = asyncio.Queue(maxsize=200)
    _subscribers.add(queue)
    log.debug("SSE subscriber added — total: %d", len(_subscribers))
    return queue


def unsubscribe(queue: asyncio.Queue) -> None:
    """Remove a subscriber queue from the registry.

    Args:
        queue: The queue returned by a previous :func:`subscribe` call.
    """
    _subscribers.discard(queue)
    log.debug("SSE subscriber removed — total: %d", len(_subscribers))


async def broadcast(event: SSEEvent) -> None:
    """Push *event* to every connected SSE subscriber.

    Subscribers whose queues are full (``maxsize`` reached) are silently
    skipped to avoid blocking the scheduler — a slow browser is not
    allowed to stall the simulation.

    Args:
        event: The :class:`~agent_agora.models.SSEEvent` to broadcast.
    """
    if not _subscribers:
        return
    dead_queues: list[asyncio.Queue] = []
    for queue in list(_subscribers):
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            log.debug("SSE queue full — dropping event for one subscriber.")
        except Exception as exc:  # noqa: BLE001
            log.warning("Unexpected error broadcasting SSE event: %s", exc)
            dead_queues.append(queue)
    for queue in dead_queues:
        unsubscribe(queue)


async def event_stream(queue: asyncio.Queue) -> AsyncIterator[str]:
    """Async generator that yields SSE-formatted strings from *queue*.

    Each event is serialised as a ``data:`` line followed by a blank line,
    as required by the SSE protocol.

    The caller is responsible for unsubscribing when the client disconnects::

        queue = subscribe()
        try:
            async for chunk in event_stream(queue):
                yield chunk
        finally:
            unsubscribe(queue)

    Args:
        queue: A subscriber queue obtained from :func:`subscribe`.

    Yields:
        SSE-formatted byte strings ready to send to the browser.
    """
    while True:
        try:
            event: SSEEvent = await asyncio.wait_for(queue.get(), timeout=30.0)
            payload = event.model_dump_json()
            yield f"event: agent_action\ndata: {payload}\n\n"
        except asyncio.TimeoutError:
            # Send a keep-alive comment so the connection stays open
            yield ": keep-alive\n\n"
        except asyncio.CancelledError:
            break
        except Exception as exc:  # noqa: BLE001
            log.warning("Error in event_stream generator: %s", exc)
            break


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

#: Singleton scheduler instance managed by this module.
_scheduler: Optional[AsyncIOScheduler] = None


async def _master_tick() -> None:
    """Master scheduler callback: tick every active agent.

    Queries the database for all active agents, then runs
    :func:`~agent_agora.agent_runner.run_agent_tick` for each one
    concurrently using :func:`asyncio.gather`.  Any SSE events returned
    are broadcast to subscribers.

    Errors in individual agent ticks are caught inside
    :func:`~agent_agora.agent_runner.run_agent_tick` and do not propagate
    here, so one failing agent cannot block others.
    """
    try:
        active_agents = db.list_agents(status=AgentStatus.ACTIVE)
    except Exception as exc:  # noqa: BLE001
        log.error("Master tick: failed to load active agents: %s", exc)
        return

    if not active_agents:
        log.debug("Master tick: no active agents.")
        return

    log.debug("Master tick: ticking %d active agent(s).", len(active_agents))

    async def _tick_and_broadcast(agent_id: int) -> None:
        """Tick one agent and broadcast its SSE event if produced."""
        try:
            event = await run_agent_tick(agent_id)
            if event is not None:
                await broadcast(event)
        except Exception as exc:  # noqa: BLE001
            log.exception("Unhandled error ticking agent %d: %s", agent_id, exc)

    agent_ids = [a.id for a in active_agents]
    await asyncio.gather(*[_tick_and_broadcast(aid) for aid in agent_ids])


def get_scheduler() -> Optional[AsyncIOScheduler]:
    """Return the current scheduler instance, or *None* if not yet created.

    Returns:
        The singleton :class:`~apscheduler.schedulers.asyncio.AsyncIOScheduler`
        instance, or *None*.
    """
    return _scheduler


def create_scheduler(
    tick_interval_seconds: int = DEFAULT_TICK_INTERVAL,
) -> AsyncIOScheduler:
    """Create (but do not start) the APScheduler instance.

    The scheduler is stored as a module-level singleton so it can be
    retrieved via :func:`get_scheduler` from anywhere in the application.

    Args:
        tick_interval_seconds: How often the master tick fires, in seconds.

    Returns:
        The configured :class:`~apscheduler.schedulers.asyncio.AsyncIOScheduler`.
    """
    global _scheduler

    if _scheduler is not None:
        log.debug("create_scheduler: scheduler already exists, returning existing instance.")
        return _scheduler

    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        _master_tick,
        trigger=IntervalTrigger(seconds=tick_interval_seconds),
        id="master_agent_tick",
        name="Master agent tick",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )

    _scheduler = scheduler
    log.info(
        "Scheduler created with tick interval %ds.", tick_interval_seconds
    )
    return scheduler


def start_scheduler(
    tick_interval_seconds: int = DEFAULT_TICK_INTERVAL,
) -> AsyncIOScheduler:
    """Create and start the APScheduler.

    Safe to call multiple times — if the scheduler is already running this
    is a no-op.

    Args:
        tick_interval_seconds: Tick interval in seconds.

    Returns:
        The running :class:`~apscheduler.schedulers.asyncio.AsyncIOScheduler`.
    """
    scheduler = create_scheduler(tick_interval_seconds)
    if not scheduler.running:
        scheduler.start()
        log.info("Scheduler started (tick every %ds).", tick_interval_seconds)
    else:
        log.debug("start_scheduler: scheduler is already running.")
    return scheduler


async def stop_scheduler() -> None:
    """Gracefully shut down the scheduler.

    Waits for any currently executing jobs to finish before returning.
    Clears the module-level singleton so :func:`create_scheduler` will
    build a fresh instance if called again.
    """
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        _scheduler.shutdown(wait=True)
        log.info("Scheduler stopped.")
    _scheduler = None


async def trigger_agent_tick(agent_id: int) -> Optional[SSEEvent]:
    """Manually trigger a single agent tick outside the normal schedule.

    Useful for the dashboard "run now" button or for tests that need
    deterministic control over when agents fire.

    Args:
        agent_id: Primary key of the agent to tick.

    Returns:
        The :class:`~agent_agora.models.SSEEvent` produced (if any), after
        broadcasting it to all subscribers.
    """
    event = await run_agent_tick(agent_id)
    if event is not None:
        await broadcast(event)
    return event


def subscriber_count() -> int:
    """Return the current number of connected SSE subscribers.

    Returns:
        Integer count of active subscriber queues.
    """
    return len(_subscribers)
