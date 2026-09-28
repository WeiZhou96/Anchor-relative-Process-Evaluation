"""Immutable, explicit label vocabularies for independent frame-axis tasks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .classes import CLASS_NAMES


@dataclass(frozen=True)
class TaskVocabulary:
    """Ordered native-code to class-index mapping, including task identity."""

    task_id: str
    version: str
    native_codes: tuple[int, ...]
    class_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("task_id must be nonempty")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("version must be nonempty")
        if not isinstance(self.native_codes, tuple) or not isinstance(self.class_names, tuple):
            raise ValueError("vocabulary sequences must be immutable tuples")
        if len(self.native_codes) < 2 or len(self.native_codes) != len(self.class_names):
            raise ValueError("vocabulary requires at least two paired codes and names")
        if any(type(c) is not int or c < 0 for c in self.native_codes):
            raise ValueError("native codes must be nonnegative integers")
        if any(not isinstance(n, str) or not n.strip() for n in self.class_names):
            raise ValueError("class names must be nonempty")
        if len(set(self.native_codes)) != self.n_classes or len(set(self.class_names)) != self.n_classes:
            raise ValueError("duplicate native codes or class names")

    @property
    def n_classes(self) -> int:
        return len(self.native_codes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "version": self.version,
            "native_codes": list(self.native_codes),
            "class_names": list(self.class_names),
        }

    @property
    def vocabulary_hash(self) -> str:
        payload = json.dumps(self.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def encode(self, native_code: int) -> int:
        if type(native_code) is not int or native_code not in self.native_codes:
            raise ValueError(f"native code outside vocabulary: {native_code!r}")
        return self.native_codes.index(native_code)

    def decode(self, class_index: int) -> int:
        if type(class_index) is not int or not 0 <= class_index < self.n_classes:
            raise ValueError(f"class index outside vocabulary: {class_index!r}")
        return self.native_codes[class_index]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TaskVocabulary:
        vocabulary = cls(value["task_id"], value["version"], tuple(value["native_codes"]), tuple(value["class_names"]))
        if "vocabulary_hash" in value and value["vocabulary_hash"] != vocabulary.vocabulary_hash:
            raise ValueError("vocabulary hash mismatch")
        return vocabulary


ACCIDENT_VOCABULARY = TaskVocabulary("accident-five-geometry", "legacy-v1", tuple(range(5)), tuple(CLASS_NAMES))
