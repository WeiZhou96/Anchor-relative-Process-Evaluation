"""Validate independent MM-AU class decisions without filling human labels."""

from __future__ import annotations

from typing import Any

CLASSES = ("head-on", "rear-end", "t-bone", "sideswipe", "single")
DECISIONS = {*CLASSES, "ambiguous", "out_of_scope"}


def merge_decision(first: str, second: str) -> tuple[str, str | None]:
    """Apply the recorded two-person rule; incomplete input is an error."""
    if first not in DECISIONS or second not in DECISIONS:
        raise ValueError("Both human decisions must be completed with an allowed value")
    if first == second and first in CLASSES:
        return "unique", first
    if first == second == "out_of_scope":
        return "out_of_scope", None
    return "ambiguous", None


def validate_form(rows: list[dict[str, Any]], definitions: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Require complete, unmodified native codes and source definitions."""
    canonical = {int(r["code"]): r["definition"] for r in definitions}
    if len(canonical) != 58 or len(definitions) != 58:
        raise ValueError("Expected the 58 release definitions")
    indexed = {int(r["code"]): r for r in rows}
    if len(indexed) != len(rows) or set(indexed) != set(canonical):
        raise ValueError("Duplicate, missing, or unexpected native class codes")
    for code, row in indexed.items():
        if row["name_from_definition_text"] != canonical[code]:
            raise ValueError(f"Definition changed for code {code}")
        if row["decision"] not in DECISIONS or not str(row.get("note") or "").strip():
            raise ValueError(f"Missing or invalid human decision/reason for code {code}")
    return indexed


def merge_forms(
    first: list[dict[str, Any]], second: list[dict[str, Any]], definitions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Merge by native code, independent of row order; preserve both judgments."""
    a, b = validate_form(first, definitions), validate_form(second, definitions)
    merged = []
    for code in sorted(a):
        status, mapped = merge_decision(a[code]["decision"], b[code]["decision"])
        merged.append(
            dict(
                code=code,
                name_from_definition_text=a[code]["name_from_definition_text"],
                mapper1=a[code]["decision"],
                mapper2=b[code]["decision"],
                note1=a[code]["note"],
                note2=b[code]["note"],
                status=status,
                mapped_class=mapped,
            )
        )
    return merged
