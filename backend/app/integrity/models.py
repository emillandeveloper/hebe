from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Finding:
    check_id: str
    category: str
    severity: str
    message: str
    count: int = 1
    records: tuple[str, ...] = ()
    blocking: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SEVERITIES = ("ERROR", "WARNING", "NEEDS_REVIEW", "INFO")
CLASSIFICATIONS = ("KEEP", "MIGRATE", "MERGE", "INVALIDATE", "ARCHIVE", "DELETE", "NEEDS_REVIEW")
