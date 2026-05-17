"""
Request/Response schemas for Mutation APIs.

Adversarial mutation and robustness testing schemas.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class MutationOperator(BaseModel):
    """Definition of a mutation operator to apply."""
    
    mutation_type: str = Field(..., description="Type of mutation")
    severity: float = Field(0.5, ge=0, le=1, description="Severity [0, 1]")
    params: Dict[str, Any] = Field(default_factory=dict, description="Operator-specific parameters")
    

class MutationRequest(BaseModel):
    """Request to apply mutations to a session or dataset."""
    
    source_npz_file_id: str = Field(..., description="Source dataset file ID")
    mutations: List[MutationOperator] = Field(..., description="Mutations to apply")
    base_seed: int = Field(42, description="Base seed for determinism")
    sample_limit: Optional[int] = Field(None, description="Limit samples (None=all)")
    batch_size: int = Field(256, description="Processing batch size")
    

class MutationResult(BaseModel):
    """Result of mutation operation."""
    
    mutation_id: str = Field(..., description="Unique mutation ID")
    source_file_id: str = Field(..., description="Source file ID")
    mutations_applied: List[MutationOperator] = Field(..., description="Applied mutations")
    samples_mutated: int = Field(..., description="Number of samples mutated")
    output_npz_file_id: str = Field(..., description="Output file ID")
    metadata_file_id: str = Field(..., description="Mutation metadata file ID (JSONL)")
    created_at: str = Field(..., description="Creation timestamp")


class MutationEvaluationRequest(BaseModel):
    """Request to evaluate robustness against mutations."""
    
    baseline_npz_file_id: str = Field(..., description="Baseline dataset file ID")
    mutated_npz_file_id: str = Field(..., description="Mutated dataset file ID")
    checkpoint_name: Optional[str] = Field("default", description="Checkpoint identifier")
    batch_size: int = Field(256, description="Evaluation batch size")
    

class MutationEvaluationResult(BaseModel):
    """Results of mutation robustness evaluation."""
    
    evaluation_id: str = Field(..., description="Unique evaluation ID")
    baseline_recall: float = Field(..., description="Baseline detection recall")
    mutated_recall: float = Field(..., description="Detection recall on mutations")
    recall_degradation: float = Field(..., description="Difference (negative = worse)")
    fpr_shift: float = Field(..., description="False positive rate shift")
    behavioral_invariance: float = Field(..., description="Behavioral robustness invariance")
    
    # Per-mutation metrics
    per_mutation_metrics: Dict[str, dict] = Field(..., description="Metrics per mutation type")
    
    # Fragile mutations
    fragile_mutations: Dict[str, int] = Field(..., description="Mutations causing most failures")
    
    created_at: str = Field(..., description="Creation timestamp")


class AvailableMutationsResponse(BaseModel):
    """List of available mutation operators."""
    
    mutations: List[Dict[str, Any]] = Field(..., description="Available mutations with descriptions")
    total_count: int = Field(..., description="Total number of operators")
