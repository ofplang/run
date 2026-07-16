"""Load and serialize execution documents (spec §6).

The runner reads an execution plan (a §6 document) and, after driving it, writes
an execution status (the same shape, §6/§7). These helpers are the thin YAML
boundary; all structure handling lives in `runner.py`.
"""

from __future__ import annotations

import yaml


def load_document(path) -> dict:
    """Read a §6 execution document (plan or status) from a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def serialize_document(document: dict) -> str:
    """Render a §6 execution document to YAML text.

    Keys are emitted in insertion order (not sorted) so the runner controls a
    readable top-level layout (time, now, interface, activities, meta).
    """
    return yaml.safe_dump(
        document,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )
