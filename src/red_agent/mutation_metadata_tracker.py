"""Metadata lineage tracking for mutated adversarial samples."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class MutatedSampleMetadata:
    """Mandatory reproducibility metadata for each mutated sample."""

    sample_id: str
    original_sample_id: str
    original_label: int
    mutation_type: str
    mutation_parameters: Dict[str, Any]
    mutation_severity: float
    seed: int
    timestamp_utc: str
    mutation_history: List[Dict[str, Any]] = field(default_factory=list)
    evaluation_outcome: Dict[str, Any] = field(default_factory=dict)


class MutationMetadataTracker:
    """Track and persist sample-level mutation lineage in JSONL format."""

    def __init__(self) -> None:
        self._records: List[MutatedSampleMetadata] = []

    @staticmethod
    def utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def add(self, record: MutatedSampleMetadata) -> None:
        self._records.append(record)

    def as_dicts(self) -> List[Dict[str, Any]]:
        return [asdict(r) for r in self._records]

    def save_jsonl(self, output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in self.as_dicts():
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    def save_json(self, output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.as_dicts(), handle, indent=2, sort_keys=True)
