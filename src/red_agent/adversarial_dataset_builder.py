"""Build adversarially mutated datasets from baseline session NPZ files."""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Iterable, List, Sequence

import numpy as np

# Import modules for side-effect registration in mutation registry.
import src.red_agent.flow_mutations as _flow_mutations  # noqa: F401
import src.red_agent.persistence_mutations as _persistence_mutations  # noqa: F401
import src.red_agent.timing_mutations as _timing_mutations  # noqa: F401
import src.red_agent.tls_mutations as _tls_mutations  # noqa: F401
from src.data_loader.npz_utils import load_session_npz
from src.features.feature_config_experiment import FEATURE_NAMES
from src.red_agent.mutation_engine import MutationEngine, MutationSpec
from src.red_agent.mutation_metadata_tracker import MutatedSampleMetadata, MutationMetadataTracker


def _seed_for_sample(base_seed: int, sample_index: int, mutation_index: int) -> int:
    return int(base_seed + (sample_index * 1009) + mutation_index)


def build_adversarial_dataset(
    input_npz: str,
    output_npz: str,
    output_metadata_jsonl: str,
    mutation_plan: Sequence[Dict[str, object]],
    base_seed: int = 1337,
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> Dict[str, object]:
    """Apply deterministic mutation plan to each sample in NPZ and save results."""

    sequences, labels, masks = load_session_npz(input_npz, expected_feature_names=feature_names)
    feature_index = {name: idx for idx, name in enumerate(feature_names)}

    engine = MutationEngine()
    tracker = MutationMetadataTracker()

    mutated_sequences = np.empty_like(sequences, dtype=np.float32)
    mutated_labels = labels.copy().astype(np.int64)
    mutated_masks = masks.copy().astype(bool)

    for i in range(sequences.shape[0]):
        specs: List[MutationSpec] = []
        for m_idx, plan in enumerate(mutation_plan):
            params = dict(plan.get("params", {}))
            params["feature_index"] = feature_index
            specs.append(
                MutationSpec(
                    mutation_type=str(plan["mutation_type"]),
                    mutation_params=params,
                    mutation_severity=float(plan.get("severity", 0.5)),
                    seed=_seed_for_sample(base_seed, i, m_idx),
                )
            )

        result = engine.apply_mutations(sequences[i], specs)
        mutated_sequences[i] = result.mutated_sequence

        history_dict = [asdict(item) for item in result.mutation_history]
        final = result.mutation_history[-1] if result.mutation_history else None
        tracker.add(
            MutatedSampleMetadata(
                sample_id=f"mut_{i}",
                original_sample_id=f"orig_{i}",
                original_label=int(labels[i]),
                mutation_type=final.mutation_type if final else "none",
                mutation_parameters=final.mutation_params if final else {},
                mutation_severity=float(final.mutation_severity) if final else 0.0,
                seed=int(final.seed) if final else base_seed,
                timestamp_utc=tracker.utc_now_iso(),
                mutation_history=history_dict,
                evaluation_outcome={},
            )
        )

    np.savez_compressed(
        output_npz,
        X=mutated_sequences.astype(np.float32),
        y=mutated_labels,
        masks=mutated_masks,
        feature_names=np.array(feature_names),
    )
    tracker.save_jsonl(output_metadata_jsonl)

    return {
        "input_npz": input_npz,
        "output_npz": output_npz,
        "output_metadata_jsonl": output_metadata_jsonl,
        "n_samples": int(mutated_sequences.shape[0]),
        "n_mutations_per_sample": int(len(mutation_plan)),
        "base_seed": int(base_seed),
    }
