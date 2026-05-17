"""Build scenario-specific CTU-13 NPZ files for the multifamily experiment.

Pipeline per scenario:
1) Parse raw CTU .binetflow into baseline 12-feature session NPZ.
2) Convert baseline NPZ into 10-feature experiment schema NPZ.

The module is manifest-driven so missing scenarios can be tracked explicitly,
while still allowing partial execution when only some raw captures are present.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data_loader.ctu13_processor import execute_pipeline
from src.pipelines.convert_npz_to_experiment_schema import convert_npz_to_experiment_schema


@dataclass(frozen=True)
class ScenarioRawSource:
    raw_path: str
    family: str
    scenario: str
    output_npz: str


def _load_sources(manifest_path: str) -> tuple[str, list[ScenarioRawSource], str | None]:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    experiment_name = payload.get("experiment_name", "ctu13_multifamily_generalization")
    benign_npz = payload.get("benign_npz")
    sources = [
        ScenarioRawSource(
            raw_path=item["raw_path"],
            family=str(item["family"]).strip().lower(),
            scenario=str(item["scenario"]).strip(),
            output_npz=item["output_npz"],
        )
        for item in payload.get("sources", [])
    ]
    if not sources:
        raise ValueError("Manifest must include at least one source")
    return experiment_name, sources, benign_npz


def _count_labels(npz_path: Path) -> tuple[int, int, int]:
    data = np.load(npz_path, allow_pickle=True)
    y = data["y"] if "y" in data else data["labels"]
    total = int(len(y))
    c2 = int((y == 1).sum())
    benign = int((y == 0).sum())
    return total, c2, benign


def _build_single_source(source: ScenarioRawSource, keep_intermediate: bool) -> dict:
    raw_path = Path(source.raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw CTU file not found: {raw_path}")

    output_npz = Path(source.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    intermediate_npz = output_npz.with_name(f"{output_npz.stem}_12f.npz")

    execute_pipeline(str(raw_path), str(intermediate_npz))
    convert_npz_to_experiment_schema(str(intermediate_npz), str(output_npz))

    if not keep_intermediate and intermediate_npz.exists():
        intermediate_npz.unlink()

    total, c2, benign = _count_labels(output_npz)
    return {
        "path": str(output_npz).replace("\\", "/"),
        "family": source.family,
        "label_type": "c2",
        "scenario": source.scenario,
        "n_sessions": total,
        "n_c2": c2,
        "n_benign": benign,
    }


def build_ctu13_scenario_npzs(
    manifest_path: str,
    out_sources_manifest: str,
    allow_partial: bool = False,
    keep_intermediate: bool = False,
) -> dict:
    experiment_name, sources, benign_npz = _load_sources(manifest_path)

    built_sources: list[dict] = []
    missing_sources: list[dict] = []

    for source in sources:
        try:
            built = _build_single_source(source, keep_intermediate=keep_intermediate)
            built_sources.append(built)
        except FileNotFoundError as exc:
            missing_sources.append(
                {
                    "raw_path": source.raw_path,
                    "family": source.family,
                    "scenario": source.scenario,
                    "error": str(exc),
                }
            )
            if not allow_partial:
                raise

    sources_payload = [
        {
            "path": item["path"],
            "family": item["family"],
            "label_type": "c2",
            "scenario": item["scenario"],
        }
        for item in built_sources
    ]

    if benign_npz:
        benign_path = Path(benign_npz)
        if benign_path.exists():
            sources_payload.append(
                {
                    "path": str(benign_path).replace("\\", "/"),
                    "family": "ctu13_benign",
                    "label_type": "benign",
                    "scenario": "ctu13_benign",
                }
            )

    out_path = Path(out_sources_manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_payload = {
        "experiment_name": experiment_name,
        "sources": sources_payload,
    }
    out_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")

    return {
        "sources_manifest": str(out_path).replace("\\", "/"),
        "built_count": len(built_sources),
        "missing_count": len(missing_sources),
        "built_sources": built_sources,
        "missing_sources": missing_sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build scenario-specific CTU-13 NPZs and a sources manifest")
    parser.add_argument(
        "--manifest",
        default="experiments/multifamily_generalization/ctu13_raw_sources.json",
        help="Raw-source manifest describing CTU scenario files",
    )
    parser.add_argument(
        "--out_sources_manifest",
        default="experiments/multifamily_generalization/ctu13_sources_generated.json",
        help="Output sources manifest usable by multifamily loaders/splitters",
    )
    parser.add_argument(
        "--allow_partial",
        action="store_true",
        help="Continue building available scenarios when some raw files are missing",
    )
    parser.add_argument(
        "--keep_intermediate",
        action="store_true",
        help="Keep temporary 12-feature NPZ files",
    )
    args = parser.parse_args()

    summary = build_ctu13_scenario_npzs(
        manifest_path=args.manifest,
        out_sources_manifest=args.out_sources_manifest,
        allow_partial=args.allow_partial,
        keep_intermediate=args.keep_intermediate,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
