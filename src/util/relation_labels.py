"""Verbal spatial relation parsing for R-channel hints (8 directions)."""

from __future__ import annotations

RELATION_CLASSES = (
    "above",
    "below",
    "behind",
    "in front of",
    "leftof",
    "rightof",
    "on",
    "under",
)
NUM_RELATION_CLASSES = len(RELATION_CLASSES)

# Backward-compatible aliases (R-channel code)
L_REL_CLASSES = RELATION_CLASSES
NUM_L_REL_CLASSES = NUM_RELATION_CLASSES


def _detect_verbal_key(text: str) -> str:
    """Map a caption to one of 8 verbal relation keys.

    WhatsUp one-object prompts use absolute phrases (``on the left/right/top/bottom``)
    that must be parsed *before* the generic `` on `` check, otherwise they collapse
    to relation ``on`` and poison the R channel.
    """
    t = text.lower()
    # Absolute / one-object WhatsUp phrasing (Kamath et al.)
    if "on the left" in t or "to the left" in t or "left of" in t:
        return "leftof"
    if "on the right" in t or "to the right" in t or "right of" in t:
        return "rightof"
    if "on the top" in t or "on top" in t:
        return "above"
    if "on the bottom" in t:
        return "below"
    if "in front of" in t:
        return "in front of"
    if "behind" in t:
        return "behind"
    if "below" in t:
        return "below"
    if "above" in t:
        return "above"
    if " under " in f" {t} " or t.endswith(" under a"):
        return "under"
    if " on " in f" {t} " or " on a " in f" {t} ":
        return "on"
    raise ValueError(f"No spatial relation found in caption: {text!r}")


def verbal_key_to_class(key: str) -> int:
    return RELATION_CLASSES.index(key)
