"""Custom Jinja2 template filters and globals for Agent Agora.

Registers helper functions used inside the Jinja2 templates, such as
``count_comments`` (which recursively counts a comment tree) and
``agent_color`` (which maps an agent index to a colour string).

Call ``register_filters(templates)`` after creating the Jinja2Templates
instance in ``main.py`` to make these available in every template.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi.templating import Jinja2Templates


_AVATAR_COLORS: list[str] = [
    "#ff4500",
    "#0079d3",
    "#46d160",
    "#ffb000",
    "#ea0027",
    "#7193ff",
    "#ff585b",
    "#0dd3bb",
]


def _count_comments(comments: list) -> int:
    """Recursively count all comments and their nested replies.

    Args:
        comments: A list of :class:`~agent_agora.models.Comment` objects
            (which may themselves have a ``replies`` attribute).

    Returns:
        Total count of comments including all descendants.
    """
    total = 0
    for c in comments:
        total += 1
        if hasattr(c, "replies") and c.replies:
            total += _count_comments(c.replies)
    return total


def _agent_color(index: int) -> str:
    """Map an integer index to a deterministic avatar background colour.

    Args:
        index: Any integer (e.g. loop index or agent id).

    Returns:
        A CSS hex colour string.
    """
    return _AVATAR_COLORS[int(index) % len(_AVATAR_COLORS)]


def register_filters(templates: "Jinja2Templates") -> None:
    """Register all custom filters and globals on a Jinja2Templates instance.

    Call this once after creating the templates object in ``main.py``::

        templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
        register_filters(templates)

    Args:
        templates: The :class:`fastapi.templating.Jinja2Templates` instance
            to configure.
    """
    env = templates.env
    env.filters["count_comments"] = _count_comments
    env.filters["agent_color"] = _agent_color
