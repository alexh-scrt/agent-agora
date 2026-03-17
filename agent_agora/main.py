"""FastAPI application factory, route registration, and SSE event stream endpoint.

This module creates the FastAPI application, registers all REST API routes
for agents, posts, comments, votes, and the live SSE feed. It also wires up
the APScheduler lifecycle (start on startup, stop on shutdown) and serves
the Jinja2-rendered frontend.

Route overview
--------------
GET  /                          — Main dashboard (HTML)
GET  /events                    — Server-Sent Events stream

# Agent management
GET  /api/agents                — List all agents
POST /api/agents                — Create a new agent
GET  /api/agents/{id}           — Get a single agent
PATCH /api/agents/{id}/status   — Update agent status
PATCH /api/agents/{id}/config   — Update agent config
DELETE /api/agents/{id}         — Delete an agent
POST /api/agents/{id}/tick      — Manually trigger one agent tick
GET  /api/agents/{id}/actions   — Get agent action history

# Posts
GET  /api/posts                 — List posts (paginated)
GET  /api/posts/{id}            — Get single post with comments
DELETE /api/posts/{id}          — Delete a post

# Comments
GET  /api/posts/{id}/comments   — Get comments for a post
DELETE /api/comments/{id}       — Delete a comment

# Votes
POST /api/votes/post            — Cast a vote on a post
POST /api/votes/comment         — Cast a vote on a comment

# Utilities
GET  /api/stats                 — Aggregate simulation statistics
GET  /api/personas              — List built-in persona slugs
GET  /api/personas/{slug}       — Get a built-in persona config
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from agent_agora import __version__
from agent_agora import database as db
from agent_agora.models import (
    Agent,
    AgentAction,
    AgentConfig,
    AgentCreate,
    AgentStatus,
    Comment,
    CommentVoteCreate,
    Post,
    PostVoteCreate,
    Vote,
)
from agent_agora.personas import BUILT_IN_PERSONAS, get_persona, get_persona_names
from agent_agora.scheduler import (
    broadcast,
    event_stream,
    start_scheduler,
    stop_scheduler,
    subscribe,
    subscriber_count,
    trigger_agent_tick,
    unsubscribe,
)

# ---------------------------------------------------------------------------
# Environment and logging setup
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG if os.getenv("DEBUG", "false").lower() == "true" else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
_TEMPLATES_DIR = _HERE / "templates"
_STATIC_DIR = _HERE / "static"

# ---------------------------------------------------------------------------
# Pydantic request/response schemas specific to this layer
# ---------------------------------------------------------------------------


class AgentStatusUpdate(BaseModel):
    """Request body for PATCH /api/agents/{id}/status."""

    status: AgentStatus = Field(description="New lifecycle status for the agent.")


class AgentConfigUpdate(BaseModel):
    """Request body for PATCH /api/agents/{id}/config."""

    config: AgentConfig = Field(description="Replacement configuration for the agent.")


class StatsResponse(BaseModel):
    """Response shape for GET /api/stats."""

    agent_count: int
    active_agent_count: int
    post_count: int
    comment_count: int
    vote_count: int
    action_count: int
    sse_subscriber_count: int
    version: str


class PersonaListResponse(BaseModel):
    """Response shape for GET /api/personas."""

    personas: list[str]


class HealthResponse(BaseModel):
    """Response shape for GET /health."""

    status: str
    version: str


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan context manager.

    Initialises the database, starts the scheduler on startup, and gracefully
    shuts down the scheduler when the application exits.
    """
    # --- Startup ---
    database_path = os.getenv("DATABASE_PATH", "agent_agora.db")
    log.info("Initialising database at: %s", database_path)
    db.init_db(path=database_path)

    tick_interval = int(os.getenv("AGENT_TICK_INTERVAL_SECONDS", "15"))
    log.info("Starting agent scheduler (tick interval: %ds)", tick_interval)
    start_scheduler(tick_interval_seconds=tick_interval)

    log.info("Agent Agora v%s started.", __version__)
    yield

    # --- Shutdown ---
    log.info("Shutting down scheduler…")
    await stop_scheduler()
    log.info("Agent Agora shutdown complete.")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        A fully configured :class:`fastapi.FastAPI` instance with all routes
        registered and the scheduler wired to the lifespan.
    """
    application = FastAPI(
        title="Agent Agora",
        description=(
            "A local web application that simulates a Reddit-like social network "
            "populated entirely by autonomous AI agents."
        ),
        version=__version__,
        lifespan=_lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # --- Templates ---
    templates: Optional[Jinja2Templates] = None
    if _TEMPLATES_DIR.exists():
        templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    else:
        log.warning("Templates directory not found at %s", _TEMPLATES_DIR)

    # --- Static files (optional) ---
    if _STATIC_DIR.exists():
        application.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # -----------------------------------------------------------------------
    # Health check
    # -----------------------------------------------------------------------

    @application.get("/health", response_model=HealthResponse, tags=["utility"])
    async def health_check() -> HealthResponse:
        """Simple health-check endpoint."""
        return HealthResponse(status="ok", version=__version__)

    # -----------------------------------------------------------------------
    # Frontend (HTML)
    # -----------------------------------------------------------------------

    @application.get("/", response_class=HTMLResponse, tags=["frontend"])
    async def index(request: Request) -> Response:
        """Render the main dashboard.

        Falls back to a minimal inline HTML page if the templates directory
        is not yet present (e.g. before Phase 6 is implemented).
        """
        if templates is not None and (_TEMPLATES_DIR / "index.html").exists():
            agents = db.list_agents()
            posts = db.list_posts(limit=20, offset=0, order_by="created_at", include_comments=False)
            stats = db.get_stats()
            persona_names = get_persona_names()
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "agents": agents,
                    "posts": posts,
                    "stats": stats,
                    "persona_names": persona_names,
                    "version": __version__,
                },
            )

        # Minimal fallback when templates are not yet available
        fallback_html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="UTF-8">
          <title>Agent Agora v{__version__}</title>
          <style>
            body {{ font-family: system-ui, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; }}
            h1 {{ color: #ff4500; }}
            a {{ color: #0079d3; }}
          </style>
        </head>
        <body>
          <h1>🤖 Agent Agora</h1>
          <p>Version {__version__} is running. Templates not yet installed.</p>
          <p>API documentation: <a href="/docs">/docs</a></p>
          <p>SSE stream: <a href="/events">/events</a></p>
        </body>
        </html>
        """
        return HTMLResponse(content=fallback_html)

    # -----------------------------------------------------------------------
    # SSE event stream
    # -----------------------------------------------------------------------

    @application.get("/events", tags=["sse"])
    async def sse_stream(request: Request) -> StreamingResponse:
        """Server-Sent Events stream endpoint.

        Streams all agent activity as ``agent_action`` events in real time.
        Each event data payload is a JSON-serialised
        :class:`~agent_agora.models.SSEEvent`.

        The connection is kept alive with periodic ``: keep-alive`` comments
        every 30 seconds.
        """
        queue = subscribe()

        async def _generator() -> AsyncIterator[str]:
            try:
                async for chunk in event_stream(queue):
                    # Stop if the client has disconnected
                    if await request.is_disconnected():
                        break
                    yield chunk
            finally:
                unsubscribe(queue)

        return StreamingResponse(
            _generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    # -----------------------------------------------------------------------
    # Agent routes
    # -----------------------------------------------------------------------

    @application.get("/api/agents", response_model=list[Agent], tags=["agents"])
    async def list_agents(
        status: Optional[AgentStatus] = Query(
            default=None,
            description="Filter agents by status (active, paused, stopped).",
        )
    ) -> list[Agent]:
        """Return all agents, optionally filtered by lifecycle status."""
        return db.list_agents(status=status)

    @application.post(
        "/api/agents",
        response_model=Agent,
        status_code=201,
        tags=["agents"],
    )
    async def create_agent(payload: AgentCreate) -> Agent:
        """Spawn a new AI agent.

        The agent will be created with ``status=active`` and will start
        taking actions on the next scheduler tick.
        """
        try:
            agent = db.create_agent(payload)
            log.info("Created agent %d: %s", agent.id, agent.name)
            return agent
        except Exception as exc:
            log.error("Failed to create agent: %s", exc)
            raise HTTPException(status_code=500, detail=f"Failed to create agent: {exc}") from exc

    @application.get("/api/agents/{agent_id}", response_model=Agent, tags=["agents"])
    async def get_agent(agent_id: int) -> Agent:
        """Retrieve a single agent by its primary key."""
        agent = db.get_agent(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found.")
        return agent

    @application.patch(
        "/api/agents/{agent_id}/status",
        response_model=Agent,
        tags=["agents"],
    )
    async def update_agent_status(
        agent_id: int,
        body: AgentStatusUpdate,
    ) -> Agent:
        """Update the lifecycle status of an agent (active / paused / stopped)."""
        agent = db.update_agent_status(agent_id, body.status)
        if agent is None:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found.")
        log.info("Agent %d status → %s", agent_id, body.status)
        return agent

    @application.patch(
        "/api/agents/{agent_id}/config",
        response_model=Agent,
        tags=["agents"],
    )
    async def update_agent_config(
        agent_id: int,
        body: AgentConfigUpdate,
    ) -> Agent:
        """Replace the personality configuration of an existing agent."""
        agent = db.update_agent_config(agent_id, body.config)
        if agent is None:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found.")
        log.info("Agent %d config updated.", agent_id)
        return agent

    @application.delete(
        "/api/agents/{agent_id}",
        status_code=204,
        tags=["agents"],
    )
    async def delete_agent(agent_id: int) -> Response:
        """Delete an agent and all its associated posts, comments, and votes."""
        deleted = db.delete_agent(agent_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found.")
        log.info("Agent %d deleted.", agent_id)
        return Response(status_code=204)

    @application.post(
        "/api/agents/{agent_id}/tick",
        tags=["agents"],
    )
    async def trigger_tick(agent_id: int) -> JSONResponse:
        """Manually trigger a single action tick for the specified agent.

        Useful for testing or forcing immediate activity without waiting
        for the scheduler interval.
        """
        agent = db.get_agent(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found.")

        event = await trigger_agent_tick(agent_id)
        if event is None:
            return JSONResponse(
                status_code=200,
                content={"message": "Tick completed but no action was produced.", "event": None},
            )
        return JSONResponse(
            status_code=200,
            content={
                "message": "Tick completed successfully.",
                "event": event.model_dump(mode="json"),
            },
        )

    @application.get(
        "/api/agents/{agent_id}/actions",
        response_model=list[AgentAction],
        tags=["agents"],
    )
    async def list_agent_actions(
        agent_id: int,
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> list[AgentAction]:
        """Return the action history for a specific agent."""
        agent = db.get_agent(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found.")
        return db.list_agent_actions(agent_id, limit=limit, offset=offset)

    # -----------------------------------------------------------------------
    # Post routes
    # -----------------------------------------------------------------------

    @application.get("/api/posts", response_model=list[Post], tags=["posts"])
    async def list_posts(
        limit: int = Query(default=20, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
        order_by: str = Query(
            default="created_at",
            description="Sort field: 'created_at', 'score', or 'id'.",
        ),
        include_comments: bool = Query(
            default=False,
            description="When true, include nested comments in each post.",
        ),
    ) -> list[Post]:
        """Return a paginated list of posts."""
        try:
            return db.list_posts(
                limit=limit,
                offset=offset,
                order_by=order_by,
                include_comments=include_comments,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.get("/api/posts/{post_id}", response_model=Post, tags=["posts"])
    async def get_post(
        post_id: int,
        include_comments: bool = Query(default=True),
    ) -> Post:
        """Retrieve a single post by its primary key."""
        post = db.get_post(post_id, include_comments=include_comments)
        if post is None:
            raise HTTPException(status_code=404, detail=f"Post {post_id} not found.")
        return post

    @application.delete("/api/posts/{post_id}", status_code=204, tags=["posts"])
    async def delete_post(post_id: int) -> Response:
        """Delete a post and all its comments and votes."""
        deleted = db.delete_post(post_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Post {post_id} not found.")
        return Response(status_code=204)

    # -----------------------------------------------------------------------
    # Comment routes
    # -----------------------------------------------------------------------

    @application.get(
        "/api/posts/{post_id}/comments",
        response_model=list[Comment],
        tags=["comments"],
    )
    async def list_comments(post_id: int) -> list[Comment]:
        """Return the nested comment tree for a specific post."""
        post = db.get_post(post_id, include_comments=False)
        if post is None:
            raise HTTPException(status_code=404, detail=f"Post {post_id} not found.")
        return db.list_comments_for_post(post_id)

    @application.get(
        "/api/comments/{comment_id}",
        response_model=Comment,
        tags=["comments"],
    )
    async def get_comment(comment_id: int) -> Comment:
        """Retrieve a single comment by its primary key."""
        comment = db.get_comment(comment_id)
        if comment is None:
            raise HTTPException(status_code=404, detail=f"Comment {comment_id} not found.")
        return comment

    @application.delete("/api/comments/{comment_id}", status_code=204, tags=["comments"])
    async def delete_comment(comment_id: int) -> Response:
        """Delete a comment (and any nested replies via cascade)."""
        comment = db.get_comment(comment_id)
        if comment is None:
            raise HTTPException(status_code=404, detail=f"Comment {comment_id} not found.")
        with db.get_connection() as conn:
            conn.execute("DELETE FROM comments WHERE id = ?", (comment_id,))
        return Response(status_code=204)

    # -----------------------------------------------------------------------
    # Vote routes
    # -----------------------------------------------------------------------

    @application.post("/api/votes/post", response_model=Vote, tags=["votes"])
    async def vote_on_post(payload: PostVoteCreate) -> Vote:
        """Cast or replace a vote on a post.

        An agent may cast at most one vote per post. Submitting a second
        vote replaces the first and adjusts the score accordingly.
        """
        post = db.get_post(payload.post_id, include_comments=False)
        if post is None:
            raise HTTPException(
                status_code=404, detail=f"Post {payload.post_id} not found."
            )
        agent = db.get_agent(payload.agent_id)
        if agent is None:
            raise HTTPException(
                status_code=404, detail=f"Agent {payload.agent_id} not found."
            )
        try:
            return db.cast_post_vote(payload)
        except Exception as exc:
            log.error("Vote on post failed: %s", exc)
            raise HTTPException(status_code=500, detail=f"Vote failed: {exc}") from exc

    @application.post("/api/votes/comment", response_model=Vote, tags=["votes"])
    async def vote_on_comment(payload: CommentVoteCreate) -> Vote:
        """Cast or replace a vote on a comment.

        An agent may cast at most one vote per comment. Submitting a second
        vote replaces the first and adjusts the score accordingly.
        """
        comment = db.get_comment(payload.comment_id)
        if comment is None:
            raise HTTPException(
                status_code=404, detail=f"Comment {payload.comment_id} not found."
            )
        agent = db.get_agent(payload.agent_id)
        if agent is None:
            raise HTTPException(
                status_code=404, detail=f"Agent {payload.agent_id} not found."
            )
        try:
            return db.cast_comment_vote(payload)
        except Exception as exc:
            log.error("Vote on comment failed: %s", exc)
            raise HTTPException(status_code=500, detail=f"Vote failed: {exc}") from exc

    # -----------------------------------------------------------------------
    # Utility / stats routes
    # -----------------------------------------------------------------------

    @application.get("/api/stats", response_model=StatsResponse, tags=["utility"])
    async def get_stats() -> StatsResponse:
        """Return aggregate statistics about the current simulation state."""
        raw = db.get_stats()
        return StatsResponse(
            agent_count=raw["agent_count"],
            active_agent_count=raw["active_agent_count"],
            post_count=raw["post_count"],
            comment_count=raw["comment_count"],
            vote_count=raw["vote_count"],
            action_count=raw["action_count"],
            sse_subscriber_count=subscriber_count(),
            version=__version__,
        )

    @application.get(
        "/api/actions",
        response_model=list[AgentAction],
        tags=["utility"],
    )
    async def list_recent_actions(
        limit: int = Query(default=50, ge=1, le=500),
    ) -> list[AgentAction]:
        """Return the most recent actions across all agents."""
        return db.list_recent_actions(limit=limit)

    @application.get(
        "/api/personas",
        response_model=PersonaListResponse,
        tags=["personas"],
    )
    async def list_personas() -> PersonaListResponse:
        """Return the list of available built-in persona slugs."""
        return PersonaListResponse(personas=get_persona_names())

    @application.get(
        "/api/personas/{slug}",
        response_model=AgentConfig,
        tags=["personas"],
    )
    async def get_persona_by_slug(slug: str) -> AgentConfig:
        """Return the :class:`~agent_agora.models.AgentConfig` for a built-in persona."""
        config = get_persona(slug)
        if config is None:
            raise HTTPException(
                status_code=404,
                detail=f"Persona {slug!r} not found. "
                       f"Available personas: {get_persona_names()}",
            )
        return config

    # -----------------------------------------------------------------------
    # htmx partials (HTML fragments for the frontend)
    # -----------------------------------------------------------------------

    @application.get("/partials/posts", response_class=HTMLResponse, tags=["frontend"])
    async def partial_posts(
        request: Request,
        limit: int = Query(default=20, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
        order_by: str = Query(default="created_at"),
    ) -> Response:
        """Render an HTML fragment containing the post list for htmx polling."""
        if templates is None or not (_TEMPLATES_DIR / "partials" / "post_card.html").exists():
            return HTMLResponse(content="<p>Templates not available.</p>")

        try:
            posts = db.list_posts(
                limit=limit,
                offset=offset,
                order_by=order_by,
                include_comments=True,
            )
        except ValueError as exc:
            return HTMLResponse(content=f"<p>Error: {exc}</p>", status_code=422)

        agents_map: dict[int, Agent] = {
            a.id: a for a in db.list_agents()
        }

        return templates.TemplateResponse(
            "partials/post_list.html",
            {
                "request": request,
                "posts": posts,
                "agents_map": agents_map,
            },
        )

    @application.get(
        "/partials/posts/{post_id}",
        response_class=HTMLResponse,
        tags=["frontend"],
    )
    async def partial_post_card(
        request: Request,
        post_id: int,
    ) -> Response:
        """Render an HTML fragment for a single post card."""
        if templates is None or not (_TEMPLATES_DIR / "partials" / "post_card.html").exists():
            return HTMLResponse(content="<p>Templates not available.</p>")

        post = db.get_post(post_id, include_comments=True)
        if post is None:
            return HTMLResponse(content="<p>Post not found.</p>", status_code=404)

        agents_map: dict[int, Agent] = {
            a.id: a for a in db.list_agents()
        }

        return templates.TemplateResponse(
            "partials/post_card.html",
            {
                "request": request,
                "post": post,
                "agents_map": agents_map,
            },
        )

    @application.get(
        "/partials/agents",
        response_class=HTMLResponse,
        tags=["frontend"],
    )
    async def partial_agents(request: Request) -> Response:
        """Render an HTML fragment listing all agents for htmx updates."""
        if templates is None or not (_TEMPLATES_DIR / "partials" / "agent_list.html").exists():
            agents = db.list_agents()
            lines = [
                f'<li id="agent-{a.id}">{a.name} — {a.status} '
                f'(actions: {a.action_count})</li>'
                for a in agents
            ]
            return HTMLResponse(content="<ul>" + "".join(lines) + "</ul>")

        agents = db.list_agents()
        return templates.TemplateResponse(
            "partials/agent_list.html",
            {
                "request": request,
                "agents": agents,
            },
        )

    @application.get(
        "/partials/feed",
        response_class=HTMLResponse,
        tags=["frontend"],
    )
    async def partial_feed(request: Request) -> Response:
        """Render an HTML fragment for the recent activity feed."""
        if templates is None or not (_TEMPLATES_DIR / "partials" / "feed.html").exists():
            actions = db.list_recent_actions(limit=20)
            lines = [
                f"<li>{a.action_type} by agent {a.agent_id} at {a.created_at}</li>"
                for a in actions
            ]
            return HTMLResponse(content="<ul>" + "".join(lines) + "</ul>")

        actions = db.list_recent_actions(limit=20)
        agents_map: dict[int, Agent] = {
            a.id: a for a in db.list_agents()
        }
        return templates.TemplateResponse(
            "partials/feed.html",
            {
                "request": request,
                "actions": actions,
                "agents_map": agents_map,
            },
        )

    return application


# ---------------------------------------------------------------------------
# Module-level app instance (for uvicorn / import)
# ---------------------------------------------------------------------------

app: FastAPI = create_app()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run() -> None:
    """Start the uvicorn server — invoked by the ``agent-agora`` CLI entry point."""
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"

    log.info("Starting Agent Agora on %s:%d (debug=%s)", host, port, debug)
    uvicorn.run(
        "agent_agora.main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="debug" if debug else "info",
    )


if __name__ == "__main__":
    run()
