from __future__ import annotations

import re
from typing import Optional

VALID_CANONICAL_CLASSES = [
    "Covered",
    "Not Covered",
    "Coverage with Conditions",
    "Coverage with Conditions(PA Required)",
    "Coverage with Conditions(ST Required)",
]

RE_PA = re.compile(r"(?:^|(?<=[^a-zA-Z0-9]))pa(?=(?:[^a-zA-Z0-9]|$))(?!\d)", re.IGNORECASE)
RE_ST = re.compile(r"(?:^|(?<=[^a-zA-Z0-9]))(?:st|dst)(?=(?:[^a-zA-Z0-9]|$))(?!\d)", re.IGNORECASE)


def standardize_base_status(status: str | None) -> Optional[str]:
    if status is None:
        return None
    s = str(status).strip()
    if not s:
        return None
    sl = s.lower()

    if "not covered" in sl or sl in {"no", "n", "not applicable", "n/a", "non-covered"}:
        return "Not Covered"
    if "covered with condition" in sl or "coverage with condition" in sl:
        return "Coverage with Conditions"
    if sl in {"covered", "part b covered", "part b\ncovered", "yes", "y"}:
        return "Covered"
    if sl.startswith("covered") and "condition" not in sl and "not" not in sl:
        return "Covered"
    if "condition" in sl:
        return "Coverage with Conditions"
    return None


def canonicalize_coverage_status(raw_status: str | None, acronym: str | None) -> Optional[str]:
    base = standardize_base_status(raw_status)
    if base is None:
        return None

    acr = (acronym or "").strip()
    if RE_PA.search(acr):
        return "Coverage with Conditions(PA Required)"
    if RE_ST.search(acr):
        return "Coverage with Conditions(ST Required)"
    return base
