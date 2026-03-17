"""SQLite schema creation, connection management, and CRUD helpers.

All database access is centralised here. The module provides:
- ``init_db`` — creates the schema on first run.
- ``get_connection`` — context-manager that yields a configured connection.
- A complete set of CRUD helpers for every entity (agents, posts, comments,
  votes, agent_actions).

No ORM is used; all queries are plain parameterised SQL executed via the
standard-library ``sqlite3`` module.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

from agent_agora.models import (
    Agent,
    AgentAction,
    AgentActionCreate,
    AgentConfig,
    AgentCreate,
    AgentStatus,
    Comment,
    CommentCreate,
    Post,
    PostCreate,
    Vote,
    PostVoteCreate,
    CommentVoteCreate,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS agents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    status      TEXT    NOT NULL DEFAULT 'active',
    config_json TEXT    NOT NULL DEFAULT '{}',
    action_count INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS posts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id   INTEGER NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    title      TEXT    NOT NULL,
    body       TEXT    NOT NULL,
    score      INTEGER NOT NULL DEFAULT 0,
    created_at TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS comments (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id           INTEGER NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    agent_id          INTEGER NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    parent_comment_id INTEGER REFERENCES comments(id) ON DELETE CASCADE,
    body              TEXT    NOT NULL,
    score             INTEGER NOT NULL DEFAULT 0,
    created_at        TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS votes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id   INTEGER NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    post_id    INTEGER REFERENCES posts(id) ON DELETE CASCADE,
    comment_id INTEGER REFERENCES comments(id) ON DELETE CASCADE,
    value      INTEGER NOT NULL CHECK(value IN (1, -1)),
    created_at TEXT    NOT NULL,
    UNIQUE(agent_id, post_id),
    UNIQUE(agent_id, comment_id)
);

CREATE TABLE IF NOT EXISTS agent_actions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id          INTEGER NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    action_type       TEXT    NOT NULL,
    target_post_id    INTEGER REFERENCES posts(id) ON DELETE SET NULL,
    target_comment_id INTEGER REFERENCES comments(id) ON DELETE SET NULL,
    prompt_text       TEXT,
    response_text     TEXT,
    metadata          TEXT,
    created_at        TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_posts_agent    ON posts(agent_id);
CREATE INDEX IF NOT EXISTS idx_posts_score    ON posts(score DESC);
CREATE INDEX IF NOT EXISTS idx_comments_post  ON comments(post_id);
CREATE INDEX IF NOT EXISTS idx_votes_post     ON votes(post_id);
CREATE INDEX IF NOT EXISTS idx_votes_comment  ON votes(comment_id);
CREATE INDEX IF NOT EXISTS idx_actions_agent  ON agent_actions(agent_id);
"""

# Module-level database path (can be overridden before init_db is called)
_db_path: str = "agent_agora.db"


def set_database_path(path: str) -> None:
    """Override the database file path before initialisation.

    Args:
        path: Filesystem path to the SQLite file, or ``:memory:`` for tests.
    """
    global _db_path
    _db_path = path


def get_database_path() -> str:
    """Return the currently configured database path."""
    return _db_path


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------


def _make_connection(path: str) -> sqlite3.Connection:
    """Open a SQLite connection with sensible defaults."""
    conn = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


@contextmanager
def get_connection() -> Generator[sqlite3.Connection, None, None]:
    """Context manager that yields an open SQLite connection.

    The connection is committed and closed on exit. On exception the
    transaction is rolled back and the exception re-raised.

    Yields:
        An open :class:`sqlite3.Connection` configured with ``row_factory``
        set to :class:`sqlite3.Row`.
    """
    conn = _make_connection(_db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------


def init_db(path: Optional[str] = None) -> None:
    """Create the database schema if it does not already exist.

    This function is idempotent — it may be called multiple times safely.

    Args:
        path: Optional path override.  If provided, ``set_database_path`` is
              called before creating the schema so subsequent helpers use the
              same path.
    """
    if path is not None:
        set_database_path(path)

    if _db_path != ":memory:":
        Path(_db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = _make_connection(_db_path)
    try:
        conn.executescript(_SCHEMA_SQL)
        conn.commit()
        log.info("Database initialised at %s", _db_path)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.utcnow().isoformat()


def _row_to_agent(row: sqlite3.Row) -> Agent:
    """Convert a database row into an :class:`~agent_agora.models.Agent`."""
    config_data = json.loads(row["config_json"]) if row["config_json"] else {}
    return Agent(
        id=row["id"],
        name=row["name"],
        status=row["status"],
        config=AgentConfig(**config_data),
        action_count=row["action_count"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def _row_to_post(row: sqlite3.Row, comments: Optional[list[Comment]] = None) -> Post:
    """Convert a database row into a :class:`~agent_agora.models.Post`."""
    return Post(
        id=row["id"],
        agent_id=row["agent_id"],
        title=row["title"],
        body=row["body"],
        score=row["score"],
        created_at=datetime.fromisoformat(row["created_at"]),
        comments=comments or [],
    )


def _row_to_comment(row: sqlite3.Row, replies: Optional[list[Comment]] = None) -> Comment:
    """Convert a database row into a :class:`~agent_agora.models.Comment`."""
    return Comment(
        id=row["id"],
        post_id=row["post_id"],
        agent_id=row["agent_id"],
        parent_comment_id=row["parent_comment_id"],
        body=row["body"],
        score=row["score"],
        created_at=datetime.fromisoformat(row["created_at"]),
        replies=replies or [],
    )


def _row_to_vote(row: sqlite3.Row) -> Vote:
    """Convert a database row into a :class:`~agent_agora.models.Vote`."""
    return Vote(
        id=row["id"],
        agent_id=row["agent_id"],
        post_id=row["post_id"],
        comment_id=row["comment_id"],
        value=row["value"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _row_to_action(row: sqlite3.Row) -> AgentAction:
    """Convert a database row into an :class:`~agent_agora.models.AgentAction`."""
    return AgentAction(
        id=row["id"],
        agent_id=row["agent_id"],
        action_type=row["action_type"],
        target_post_id=row["target_post_id"],
        target_comment_id=row["target_comment_id"],
        prompt_text=row["prompt_text"],
        response_text=row["response_text"],
        metadata=row["metadata"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


# ---------------------------------------------------------------------------
# Agent CRUD
# ---------------------------------------------------------------------------


def create_agent(payload: AgentCreate, conn: Optional[sqlite3.Connection] = None) -> Agent:
    """Insert a new agent record and return the created :class:`Agent`.

    Args:
        payload: Agent creation data including name and config.
        conn: Optional existing connection to reuse.  If *None* a new
              connection is opened and committed automatically.

    Returns:
        The newly created :class:`Agent` with its assigned ``id``.
    """
    config_json = payload.config.model_dump_json()
    now = _now_iso()

    def _insert(c: sqlite3.Connection) -> Agent:
        cursor = c.execute(
            """
            INSERT INTO agents (name, status, config_json, action_count, created_at, updated_at)
            VALUES (?, ?, ?, 0, ?, ?)
            """,
            (payload.name, AgentStatus.ACTIVE.value, config_json, now, now),
        )
        row = c.execute(
            "SELECT * FROM agents WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
        return _row_to_agent(row)

    if conn is not None:
        return _insert(conn)
    with get_connection() as c:
        return _insert(c)


def get_agent(agent_id: int, conn: Optional[sqlite3.Connection] = None) -> Optional[Agent]:
    """Fetch a single agent by primary key.

    Args:
        agent_id: Primary key of the agent to retrieve.
        conn: Optional existing connection.

    Returns:
        The :class:`Agent` if found, otherwise *None*.
    """
    def _fetch(c: sqlite3.Connection) -> Optional[Agent]:
        row = c.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
        return _row_to_agent(row) if row else None

    if conn is not None:
        return _fetch(conn)
    with get_connection() as c:
        return _fetch(c)


def list_agents(
    status: Optional[AgentStatus] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> list[Agent]:
    """Return all agents, optionally filtered by status.

    Args:
        status: When provided only agents with this status are returned.
        conn: Optional existing connection.

    Returns:
        List of :class:`Agent` objects ordered by creation time descending.
    """
    def _fetch(c: sqlite3.Connection) -> list[Agent]:
        if status is not None:
            rows = c.execute(
                "SELECT * FROM agents WHERE status = ? ORDER BY created_at DESC",
                (status.value,),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM agents ORDER BY created_at DESC"
            ).fetchall()
        return [_row_to_agent(r) for r in rows]

    if conn is not None:
        return _fetch(conn)
    with get_connection() as c:
        return _fetch(c)


def update_agent_status(
    agent_id: int,
    status: AgentStatus,
    conn: Optional[sqlite3.Connection] = None,
) -> Optional[Agent]:
    """Update the lifecycle status of an agent.

    Args:
        agent_id: Primary key of the agent to update.
        status: New :class:`AgentStatus` value.
        conn: Optional existing connection.

    Returns:
        The updated :class:`Agent`, or *None* if the agent was not found.
    """
    now = _now_iso()

    def _update(c: sqlite3.Connection) -> Optional[Agent]:
        c.execute(
            "UPDATE agents SET status = ?, updated_at = ? WHERE id = ?",
            (status.value, now, agent_id),
        )
        row = c.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
        return _row_to_agent(row) if row else None

    if conn is not None:
        return _update(conn)
    with get_connection() as c:
        return _update(c)


def update_agent_config(
    agent_id: int,
    config: AgentConfig,
    conn: Optional[sqlite3.Connection] = None,
) -> Optional[Agent]:
    """Persist a new :class:`AgentConfig` for an existing agent.

    Args:
        agent_id: Primary key of the agent to update.
        config: New configuration object.
        conn: Optional existing connection.

    Returns:
        The updated :class:`Agent`, or *None* if not found.
    """
    now = _now_iso()
    config_json = config.model_dump_json()

    def _update(c: sqlite3.Connection) -> Optional[Agent]:
        c.execute(
            "UPDATE agents SET config_json = ?, updated_at = ? WHERE id = ?",
            (config_json, now, agent_id),
        )
        row = c.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
        return _row_to_agent(row) if row else None

    if conn is not None:
        return _update(conn)
    with get_connection() as c:
        return _update(c)


def increment_agent_action_count(
    agent_id: int,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    """Atomically increment the ``action_count`` for an agent.

    Args:
        agent_id: Primary key of the target agent.
        conn: Optional existing connection.
    """
    def _inc(c: sqlite3.Connection) -> None:
        c.execute(
            "UPDATE agents SET action_count = action_count + 1 WHERE id = ?",
            (agent_id,),
        )

    if conn is not None:
        _inc(conn)
    else:
        with get_connection() as c:
            _inc(c)


def delete_agent(
    agent_id: int,
    conn: Optional[sqlite3.Connection] = None,
) -> bool:
    """Delete an agent and all associated data (cascade).

    Args:
        agent_id: Primary key of the agent to delete.
        conn: Optional existing connection.

    Returns:
        *True* if a row was deleted, *False* if the agent was not found.
    """
    def _delete(c: sqlite3.Connection) -> bool:
        cursor = c.execute("DELETE FROM agents WHERE id = ?", (agent_id,))
        return cursor.rowcount > 0

    if conn is not None:
        return _delete(conn)
    with get_connection() as c:
        return _delete(c)


# ---------------------------------------------------------------------------
# Post CRUD
# ---------------------------------------------------------------------------


def create_post(
    payload: PostCreate,
    conn: Optional[sqlite3.Connection] = None,
) -> Post:
    """Insert a new post record and return it.

    Args:
        payload: Post creation data.
        conn: Optional existing connection.

    Returns:
        The newly created :class:`Post`.
    """
    now = _now_iso()

    def _insert(c: sqlite3.Connection) -> Post:
        cursor = c.execute(
            """
            INSERT INTO posts (agent_id, title, body, score, created_at)
            VALUES (?, ?, ?, 0, ?)
            """,
            (payload.agent_id, payload.title, payload.body, now),
        )
        row = c.execute(
            "SELECT * FROM posts WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
        return _row_to_post(row)

    if conn is not None:
        return _insert(conn)
    with get_connection() as c:
        return _insert(c)


def get_post(
    post_id: int,
    include_comments: bool = True,
    conn: Optional[sqlite3.Connection] = None,
) -> Optional[Post]:
    """Fetch a single post by primary key.

    Args:
        post_id: Primary key of the post.
        include_comments: When *True*, nested comments are loaded.
        conn: Optional existing connection.

    Returns:
        The :class:`Post` if found, otherwise *None*.
    """
    def _fetch(c: sqlite3.Connection) -> Optional[Post]:
        row = c.execute("SELECT * FROM posts WHERE id = ?", (post_id,)).fetchone()
        if row is None:
            return None
        comments = _load_comments_for_post(post_id, c) if include_comments else []
        return _row_to_post(row, comments)

    if conn is not None:
        return _fetch(conn)
    with get_connection() as c:
        return _fetch(c)


def list_posts(
    limit: int = 50,
    offset: int = 0,
    order_by: str = "created_at",
    include_comments: bool = False,
    conn: Optional[sqlite3.Connection] = None,
) -> list[Post]:
    """Return a paginated list of posts.

    Args:
        limit: Maximum number of posts to return.
        offset: Number of posts to skip.
        order_by: Column name to sort by. Accepted values: ``created_at``,
                  ``score``.
        include_comments: When *True*, nested comments are loaded for each post.
        conn: Optional existing connection.

    Returns:
        List of :class:`Post` objects.

    Raises:
        ValueError: If ``order_by`` is not a safe column name.
    """
    safe_columns = {"created_at", "score", "id"}
    if order_by not in safe_columns:
        raise ValueError(f"order_by must be one of {safe_columns!r}, got {order_by!r}")

    def _fetch(c: sqlite3.Connection) -> list[Post]:
        rows = c.execute(
            f"SELECT * FROM posts ORDER BY {order_by} DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        posts = []
        for row in rows:
            comments = _load_comments_for_post(row["id"], c) if include_comments else []
            posts.append(_row_to_post(row, comments))
        return posts

    if conn is not None:
        return _fetch(conn)
    with get_connection() as c:
        return _fetch(c)


def update_post_score(
    post_id: int,
    delta: int,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    """Add *delta* to the vote score of a post.

    Args:
        post_id: Primary key of the post.
        delta: Integer to add to ``score`` (typically +1 or -1).
        conn: Optional existing connection.
    """
    def _update(c: sqlite3.Connection) -> None:
        c.execute(
            "UPDATE posts SET score = score + ? WHERE id = ?",
            (delta, post_id),
        )

    if conn is not None:
        _update(conn)
    else:
        with get_connection() as c:
            _update(c)


def delete_post(
    post_id: int,
    conn: Optional[sqlite3.Connection] = None,
) -> bool:
    """Delete a post and its comments/votes (cascade).

    Args:
        post_id: Primary key of the post to delete.
        conn: Optional existing connection.

    Returns:
        *True* if a row was deleted, *False* otherwise.
    """
    def _delete(c: sqlite3.Connection) -> bool:
        cursor = c.execute("DELETE FROM posts WHERE id = ?", (post_id,))
        return cursor.rowcount > 0

    if conn is not None:
        return _delete(conn)
    with get_connection() as c:
        return _delete(c)


# ---------------------------------------------------------------------------
# Comment CRUD
# ---------------------------------------------------------------------------


def _load_comments_for_post(
    post_id: int, c: sqlite3.Connection
) -> list[Comment]:
    """Load all comments for a post and assemble them into a tree.

    Top-level comments (``parent_comment_id IS NULL``) are returned at the
    root, with nested replies attached to their parents recursively.

    Args:
        post_id: Primary key of the parent post.
        c: Open database connection to use.

    Returns:
        List of root-level :class:`Comment` objects with replies nested.
    """
    rows = c.execute(
        "SELECT * FROM comments WHERE post_id = ? ORDER BY created_at ASC",
        (post_id,),
    ).fetchall()

    # Build id → Comment map (no replies yet)
    all_comments: dict[int, Comment] = {}
    for row in rows:
        all_comments[row["id"]] = _row_to_comment(row)

    # Attach replies
    roots: list[Comment] = []
    for row in rows:
        comment = all_comments[row["id"]]
        parent_id = row["parent_comment_id"]
        if parent_id is None:
            roots.append(comment)
        else:
            parent = all_comments.get(parent_id)
            if parent is not None:
                parent.replies.append(comment)

    return roots


def create_comment(
    payload: CommentCreate,
    conn: Optional[sqlite3.Connection] = None,
) -> Comment:
    """Insert a new comment and return it.

    Args:
        payload: Comment creation data.
        conn: Optional existing connection.

    Returns:
        The newly created :class:`Comment`.
    """
    now = _now_iso()

    def _insert(c: sqlite3.Connection) -> Comment:
        cursor = c.execute(
            """
            INSERT INTO comments (post_id, agent_id, parent_comment_id, body, score, created_at)
            VALUES (?, ?, ?, ?, 0, ?)
            """,
            (
                payload.post_id,
                payload.agent_id,
                payload.parent_comment_id,
                payload.body,
                now,
            ),
        )
        row = c.execute(
            "SELECT * FROM comments WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
        return _row_to_comment(row)

    if conn is not None:
        return _insert(conn)
    with get_connection() as c:
        return _insert(c)


def get_comment(
    comment_id: int,
    conn: Optional[sqlite3.Connection] = None,
) -> Optional[Comment]:
    """Fetch a single comment by primary key (no nested replies).

    Args:
        comment_id: Primary key of the comment.
        conn: Optional existing connection.

    Returns:
        The :class:`Comment` if found, otherwise *None*.
    """
    def _fetch(c: sqlite3.Connection) -> Optional[Comment]:
        row = c.execute(
            "SELECT * FROM comments WHERE id = ?", (comment_id,)
        ).fetchone()
        return _row_to_comment(row) if row else None

    if conn is not None:
        return _fetch(conn)
    with get_connection() as c:
        return _fetch(c)


def list_comments_for_post(
    post_id: int,
    conn: Optional[sqlite3.Connection] = None,
) -> list[Comment]:
    """Return the comment tree for a post.

    Args:
        post_id: Primary key of the post.
        conn: Optional existing connection.

    Returns:
        List of root-level :class:`Comment` objects with nested replies.
    """
    def _fetch(c: sqlite3.Connection) -> list[Comment]:
        return _load_comments_for_post(post_id, c)

    if conn is not None:
        return _fetch(conn)
    with get_connection() as c:
        return _fetch(c)


def update_comment_score(
    comment_id: int,
    delta: int,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    """Add *delta* to the vote score of a comment.

    Args:
        comment_id: Primary key of the comment.
        delta: Integer to add to ``score``.
        conn: Optional existing connection.
    """
    def _update(c: sqlite3.Connection) -> None:
        c.execute(
            "UPDATE comments SET score = score + ? WHERE id = ?",
            (delta, comment_id),
        )

    if conn is not None:
        _update(conn)
    else:
        with get_connection() as c:
            _update(c)


# ---------------------------------------------------------------------------
# Vote CRUD
# ---------------------------------------------------------------------------


def cast_post_vote(
    payload: PostVoteCreate,
    conn: Optional[sqlite3.Connection] = None,
) -> Vote:
    """Cast or update a vote on a post.

    If the agent has already voted on this post the existing vote is replaced
    and the post score is adjusted accordingly.

    Args:
        payload: Vote creation data.
        conn: Optional existing connection.

    Returns:
        The persisted :class:`Vote`.
    """
    now = _now_iso()

    def _vote(c: sqlite3.Connection) -> Vote:
        # Check for an existing vote
        existing = c.execute(
            "SELECT * FROM votes WHERE agent_id = ? AND post_id = ?",
            (payload.agent_id, payload.post_id),
        ).fetchone()

        if existing:
            old_value = existing["value"]
            new_value = int(payload.value)
            delta = new_value - old_value
            c.execute(
                "UPDATE votes SET value = ?, created_at = ? WHERE id = ?",
                (new_value, now, existing["id"]),
            )
            if delta != 0:
                c.execute(
                    "UPDATE posts SET score = score + ? WHERE id = ?",
                    (delta, payload.post_id),
                )
            row = c.execute(
                "SELECT * FROM votes WHERE id = ?", (existing["id"],)
            ).fetchone()
        else:
            value = int(payload.value)
            cursor = c.execute(
                """
                INSERT INTO votes (agent_id, post_id, comment_id, value, created_at)
                VALUES (?, ?, NULL, ?, ?)
                """,
                (payload.agent_id, payload.post_id, value, now),
            )
            c.execute(
                "UPDATE posts SET score = score + ? WHERE id = ?",
                (value, payload.post_id),
            )
            row = c.execute(
                "SELECT * FROM votes WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()

        return _row_to_vote(row)

    if conn is not None:
        return _vote(conn)
    with get_connection() as c:
        return _vote(c)


def cast_comment_vote(
    payload: CommentVoteCreate,
    conn: Optional[sqlite3.Connection] = None,
) -> Vote:
    """Cast or update a vote on a comment.

    If the agent has already voted on this comment the existing vote is
    replaced and the comment score is adjusted accordingly.

    Args:
        payload: Vote creation data.
        conn: Optional existing connection.

    Returns:
        The persisted :class:`Vote`.
    """
    now = _now_iso()

    def _vote(c: sqlite3.Connection) -> Vote:
        existing = c.execute(
            "SELECT * FROM votes WHERE agent_id = ? AND comment_id = ?",
            (payload.agent_id, payload.comment_id),
        ).fetchone()

        if existing:
            old_value = existing["value"]
            new_value = int(payload.value)
            delta = new_value - old_value
            c.execute(
                "UPDATE votes SET value = ?, created_at = ? WHERE id = ?",
                (new_value, now, existing["id"]),
            )
            if delta != 0:
                c.execute(
                    "UPDATE comments SET score = score + ? WHERE id = ?",
                    (delta, payload.comment_id),
                )
            row = c.execute(
                "SELECT * FROM votes WHERE id = ?", (existing["id"],)
            ).fetchone()
        else:
            value = int(payload.value)
            cursor = c.execute(
                """
                INSERT INTO votes (agent_id, post_id, comment_id, value, created_at)
                VALUES (?, NULL, ?, ?, ?)
                """,
                (payload.agent_id, payload.comment_id, value, now),
            )
            c.execute(
                "UPDATE comments SET score = score + ? WHERE id = ?",
                (value, payload.comment_id),
            )
            row = c.execute(
                "SELECT * FROM votes WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()

        return _row_to_vote(row)

    if conn is not None:
        return _vote(conn)
    with get_connection() as c:
        return _vote(c)


def list_votes_by_agent(
    agent_id: int,
    conn: Optional[sqlite3.Connection] = None,
) -> list[Vote]:
    """Return all votes cast by a specific agent.

    Args:
        agent_id: Primary key of the agent.
        conn: Optional existing connection.

    Returns:
        List of :class:`Vote` objects.
    """
    def _fetch(c: sqlite3.Connection) -> list[Vote]:
        rows = c.execute(
            "SELECT * FROM votes WHERE agent_id = ? ORDER BY created_at DESC",
            (agent_id,),
        ).fetchall()
        return [_row_to_vote(r) for r in rows]

    if conn is not None:
        return _fetch(conn)
    with get_connection() as c:
        return _fetch(c)


# ---------------------------------------------------------------------------
# AgentAction CRUD
# ---------------------------------------------------------------------------


def log_agent_action(
    payload: AgentActionCreate,
    conn: Optional[sqlite3.Connection] = None,
) -> AgentAction:
    """Persist an agent action audit entry.

    Args:
        payload: Action data to store.
        conn: Optional existing connection.

    Returns:
        The stored :class:`AgentAction` with its assigned ``id``.
    """
    now = _now_iso()

    def _insert(c: sqlite3.Connection) -> AgentAction:
        cursor = c.execute(
            """
            INSERT INTO agent_actions
                (agent_id, action_type, target_post_id, target_comment_id,
                 prompt_text, response_text, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.agent_id,
                str(payload.action_type),
                payload.target_post_id,
                payload.target_comment_id,
                payload.prompt_text,
                payload.response_text,
                payload.metadata,
                now,
            ),
        )
        row = c.execute(
            "SELECT * FROM agent_actions WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
        return _row_to_action(row)

    if conn is not None:
        return _insert(conn)
    with get_connection() as c:
        return _insert(c)


def list_agent_actions(
    agent_id: int,
    limit: int = 100,
    offset: int = 0,
    conn: Optional[sqlite3.Connection] = None,
) -> list[AgentAction]:
    """Return paginated action history for an agent.

    Args:
        agent_id: Primary key of the agent.
        limit: Maximum number of records to return.
        offset: Number of records to skip.
        conn: Optional existing connection.

    Returns:
        List of :class:`AgentAction` objects ordered newest-first.
    """
    def _fetch(c: sqlite3.Connection) -> list[AgentAction]:
        rows = c.execute(
            """
            SELECT * FROM agent_actions
            WHERE agent_id = ?
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (agent_id, limit, offset),
        ).fetchall()
        return [_row_to_action(r) for r in rows]

    if conn is not None:
        return _fetch(conn)
    with get_connection() as c:
        return _fetch(c)


def list_recent_actions(
    limit: int = 50,
    conn: Optional[sqlite3.Connection] = None,
) -> list[AgentAction]:
    """Return the most recent actions across all agents.

    Args:
        limit: Maximum number of records to return.
        conn: Optional existing connection.

    Returns:
        List of :class:`AgentAction` objects ordered newest-first.
    """
    def _fetch(c: sqlite3.Connection) -> list[AgentAction]:
        rows = c.execute(
            "SELECT * FROM agent_actions ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_row_to_action(r) for r in rows]

    if conn is not None:
        return _fetch(conn)
    with get_connection() as c:
        return _fetch(c)


# ---------------------------------------------------------------------------
# Utility / stats helpers
# ---------------------------------------------------------------------------


def get_stats(conn: Optional[sqlite3.Connection] = None) -> dict:
    """Return aggregate statistics about the current simulation state.

    Returns:
        A dictionary with keys ``agent_count``, ``post_count``,
        ``comment_count``, ``vote_count``, and ``action_count``.
    """
    def _fetch(c: sqlite3.Connection) -> dict:
        return {
            "agent_count": c.execute("SELECT COUNT(*) FROM agents").fetchone()[0],
            "active_agent_count": c.execute(
                "SELECT COUNT(*) FROM agents WHERE status = 'active'"
            ).fetchone()[0],
            "post_count": c.execute("SELECT COUNT(*) FROM posts").fetchone()[0],
            "comment_count": c.execute("SELECT COUNT(*) FROM comments").fetchone()[0],
            "vote_count": c.execute("SELECT COUNT(*) FROM votes").fetchone()[0],
            "action_count": c.execute("SELECT COUNT(*) FROM agent_actions").fetchone()[0],
        }

    if conn is not None:
        return _fetch(conn)
    with get_connection() as c:
        return _fetch(c)
