"""Conservative (subject, relation, object) parse for the 8-direction inventory.

Returns None for unary location phrases (on the left/top/bottom) and for
captions that are not a binary spatial template. Aliases map WhatsUp / VG
wording onto the notebook 8-word keys.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

# Longest phrase first.
_PHRASE_TO_KEY = (
    ("in front of", "in front of"),
    ("to the front of", "in front of"),
    ("to the behind of", "behind"),
    ("to the left of", "leftof"),
    ("to the right of", "rightof"),
    ("left of", "leftof"),
    ("right of", "rightof"),
    ("on top of", "on"),
    ("in front", "in front of"),
    ("behind", "behind"),
    ("above", "above"),
    ("below", "below"),
    ("under", "under"),
    ("on", "on"),
)

_UNARY = re.compile(
    r"\bon the (left|right|top|bottom|front|back)\b",
    flags=re.IGNORECASE,
)

_PREFIX = re.compile(
    r"^(?:an?\s+)?(?:photo|picture|image)\s+of\s+",
    flags=re.IGNORECASE,
)
_DET = re.compile(r"^(?:the|an|a)\s+", flags=re.IGNORECASE)


def _strip_left(span: str) -> str:
    s = span.strip(" .,:;")
    s = _PREFIX.sub("", s)
    s = _DET.sub("", s)
    s = re.sub(r"^(?:the|an|a)\s+", "", s, flags=re.IGNORECASE)
    return s.strip(" .,:;")


def _strip_right(span: str) -> str:
    s = span.strip(" .,:;")
    s = _DET.sub("", s)
    s = s.split(".")[0].split(",")[0].strip()
    return s.strip(" .,:;")


def parse_spatial_triplet(caption: str) -> Optional[Tuple[str, str, str]]:
    if not caption or not str(caption).strip():
        return None
    raw = " ".join(str(caption).strip().split())
    low = raw.lower()
    if _UNARY.search(low):
        return None

    for phrase, key in _PHRASE_TO_KEY:
        pat = re.compile(
            rf"^(?P<left>.+?)\s+{re.escape(phrase)}\s+(?P<right>.+)$",
            flags=re.IGNORECASE,
        )
        m = pat.search(low)
        if not m:
            continue
        subj = _strip_left(m.group("left"))
        obj = _strip_right(m.group("right"))
        if not subj or not obj:
            continue
        if subj == obj:
            continue
        if len(subj.split()) > 8 or len(obj.split()) > 8:
            continue
        return subj, key, obj
    return None
