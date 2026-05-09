"""CTU-13 multifamily dataset build entrypoint.

This is a scaffolded loader path for the controlled experiment. It keeps the
scenario/family mapping explicit via a small manifest and can be extended to
scenario-specific raw parsing once the raw CTU-13 file layout is finalized.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data_loader.ctu13_multifamily_loader import Ctu13MultifamilyLoader, Ctu13SourceSpec


DEFAULT_SOURCES = [
    Ctu13SourceSpec(path="data/processed/ctu13_c2_sessions.npz", family="ctu13_mixed", label_type="c2", scenario="ctu13_c2"),
    Ctu13SourceSpec(path="data/processed/ctu13_external_unseen_sample.npz", family="ctu13_mixed", label_type="c2", scenario="ctu13_external_unseen"),
    Ctu13SourceSpec(path="data/processed/ctu13_external_unseen_sample_30k.npz", family="ctu13_mixed", label_type="c2", scenario="ctu13_external_unseen_30k"),
    Ctu13SourceSpec(path="data/processed/benign_mix/ctu13_benign_sample_sessions.npz", family="ctu13_benign", label_type="benign", scenario="ctu13_benign"),
]


def _load_source_specs_from_file(path: str | None) -> list[Ctu13SourceSpec]:
    if not path:
        return DEFAULT_SOURCES

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    sources = []
    for item in payload.get("sources", []):
        sources.append(
            Ctu13SourceSpec(
                path=item["path"],
                family=item["family"],
                label_type=item.get("label_type", "c2"),
                scenario=item.get("scenario", ""),
            )
        )
    return sources or DEFAULT_SOURCES


def run_ctu13_multifamily_build(manifest_path: str, out_manifest_path: str) -> str:
    loader = Ctu13MultifamilyLoader(_load_source_specs_from_file(manifest_path))
    _, _, _, manifest = loader.load()
    manifest.save(out_manifest_path)
    return out_manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a CTU-13 multifamily manifest")
    parser.add_argument(
        "--manifest",
        default="experiments/multifamily_generalization/ctu13_sources.json",
        help="JSON file describing CTU-13 sources; defaults to the repo-provided source manifest",
    )
    parser.add_argument(
        "--out_manifest",
        default="experiments/multifamily_generalization/ctu13_multifamily_manifest.json",
        help="Where to save the generated manifest",
    )
    args = parser.parse_args()

    output_path = run_ctu13_multifamily_build(args.manifest, args.out_manifest)
    print(f"[COMPLETE] Wrote multifamily manifest to {output_path}")


if __name__ == "__main__":
    main()
