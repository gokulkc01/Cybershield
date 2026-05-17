"""Mutation-based adversarial robustness toolkit for CyberShield."""

from src.red_agent.adversarial_dataset_builder import build_adversarial_dataset
from src.red_agent.behavioral_failure_analyzer import analyze_failures
from src.red_agent.evaluation_runner import run_mutation_evaluation
from src.red_agent.mutation_engine import MutationEngine, MutationRecord, MutationResult, MutationSpec
from src.red_agent.mutation_metadata_tracker import MutatedSampleMetadata, MutationMetadataTracker
from src.red_agent.mutation_registry import get_mutation, list_mutations, register_mutation

# Import mutation modules to ensure registry population via decorators.
import src.red_agent.flow_mutations as _flow_mutations  # noqa: F401
import src.red_agent.persistence_mutations as _persistence_mutations  # noqa: F401
import src.red_agent.timing_mutations as _timing_mutations  # noqa: F401
import src.red_agent.tls_mutations as _tls_mutations  # noqa: F401

__all__ = [
	"MutationEngine",
	"MutationSpec",
	"MutationResult",
	"MutationRecord",
	"MutatedSampleMetadata",
	"MutationMetadataTracker",
	"register_mutation",
	"get_mutation",
	"list_mutations",
	"build_adversarial_dataset",
	"run_mutation_evaluation",
	"analyze_failures",
]
