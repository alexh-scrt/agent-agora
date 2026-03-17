"""Agent Agora — a multi-agent Reddit-like simulation platform.

This package exposes the FastAPI application instance and key version
metadata so that external tooling (uvicorn, importlib.metadata, etc.)
can discover them without importing the full application stack.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Agent Agora Contributors"
__license__ = "MIT"
__description__ = (
    "A local web application that simulates a Reddit-like social network "
    "populated entirely by autonomous AI agents."
)

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "__description__",
]
