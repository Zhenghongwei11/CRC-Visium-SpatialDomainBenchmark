"""Shared mapping for supplementary figure output identifiers."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NUMBERING = json.loads((ROOT / "config/supplementary_numbering.json").read_text())
FIGURE_OLD_TO_NEW = {
    int(old): int(new)
    for old, new in NUMBERING["supplementary_figures_old_to_new"].items()
}


def figure_name(value: str) -> str:
    """Map a legacy supplementary figure identifier, preserving its suffix."""
    match = re.match(r"^(figureS)(\d+)(.*)$", value)
    if not match:
        return value
    old = int(match.group(2))
    if old not in FIGURE_OLD_TO_NEW:
        raise ValueError(f"No supplementary figure mapping for {value}")
    return f"{match.group(1)}{FIGURE_OLD_TO_NEW[old]}{match.group(3)}"
