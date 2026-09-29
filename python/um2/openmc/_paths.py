"""Path helpers (subset of openmc.utility_funcs)."""

from __future__ import annotations

from pathlib import Path


def input_path(filename) -> Path:
    """Return a path object for an input file, resolved to an absolute path
    unless config['resolve_paths'] is False."""
    from .config import config
    p = Path(filename)
    return p.resolve() if config['resolve_paths'] else p
