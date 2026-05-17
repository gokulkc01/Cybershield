"""
Request/Response schemas for Robustness Analytics APIs.

Robustness metrics and failure analysis schemas.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class RobustnessReport(BaseModel):
    """Comprehensive robustness evaluation report."""
    
    report_id: str = Field(..., description="Unique report ID")
    baseline_recall: float = Field(..., description="Baseline detection recall")
    mutated_recall: float = Field(..., description="Detection recall under mutations")
    recall_degradation: float = Field(..., description="Percentage degradation")
    
    # Robustness metrics
    behavioral_invariance: float = Field(..., description="Behavioral invariance score")
    fpr_shift: float = Field(..., description="False positive rate shift")
    
    # Per-mutation breakdown
    per_mutation_analysis: Dict[str, dict] = Field(..., description="Analysis per mutation type")
    
    # Mutation ranking by impact
    mutation_impact_ranking: List[Dict[str, Any]] = Field(..., description="Mutations ranked by impact")
    
    # Fragile samples
    fragile_samples: int = Field(..., description="Samples vulnerable to mutation")
    
    # Feature analysis
    feature_drift: Dict[str, dict] = Field(..., description="Feature shifts per mutation")
    
    created_at: str = Field(..., description="Creation timestamp")


class RobustnessMetrics(BaseModel):
    """Key robustness metrics."""
    
    metric_id: str = Field(..., description="Metric set ID")
    baseline_recall: float = Field(..., description="Baseline recall")
    mutated_recall: float = Field(..., description="Mutated recall")
    recall_degradation: float = Field(..., description="Degradation %")
    behavioral_invariance: float = Field(..., description="Invariance score")
    fpr_baseline: float = Field(..., description="Baseline FPR")
    fpr_mutated: float = Field(..., description="Mutated FPR")


class FailureAnalysisResult(BaseModel):
    """Detailed behavioral failure analysis."""
    
    analysis_id: str = Field(..., description="Unique analysis ID")
    total_failures: int = Field(..., description="Total failure samples")
    failure_rate: float = Field(..., description="Percentage of samples failing")
    
    # Fragile mutations
    fragile_mutations: Dict[str, int] = Field(..., description="Mutation → failure count")
    
    # Failed invariants
    failed_invariants: List[Dict[str, Any]] = Field(..., description="Broken behavioral properties")
    
    # Feature shifts
    feature_shift_analysis: Dict[str, dict] = Field(..., description="Per-feature analysis")
    
    # Behavioral patterns
    behavioral_patterns: List[Dict[str, Any]] = Field(..., description="Identified patterns in failures")
    
    created_at: str = Field(..., description="Creation timestamp")


class RobustnessComparisonRequest(BaseModel):
    """Request to compare robustness across mutations."""
    
    baseline_npz_file_id: str = Field(..., description="Baseline dataset")
    mutation_groups: List[Dict[str, Any]] = Field(..., description="Groups of mutations to compare")
    checkpoint_name: Optional[str] = Field("default", description="Checkpoint to use")


class RedAgentMutationRequest(BaseModel):
    """Request to evaluate Red-Agent mutations remotely."""

    sessions: List[List[List[float]]] = Field(..., description="Batch of session tensors")
    masks: List[List[bool]] = Field(..., description="Batch of real-flow masks")
    mutation_types: Optional[List[str]] = Field(None, description="Optional mutation types per sample")
    severities: Optional[List[float]] = Field(None, description="Optional mutation severities per sample")
    labels: Optional[List[int]] = Field(None, description="Optional ground-truth labels (1=malicious,0=benign)")
    threshold: Optional[float] = Field(0.5, description="Decision threshold for detector confidence to declare positive")


class RedAgentMutationItem(BaseModel):
    """Single Red-Agent evaluation result."""

    sample_index: int = Field(..., description="Index within the batch")
    mutation_type: str = Field(..., description="Mutation type used")
    severity: float = Field(..., description="Mutation severity")
    realism_score: float = Field(..., description="Realism score")
    is_realistic: bool = Field(..., description="Whether the mutation passed realism checks")
    constraint_violations: List[str] = Field(..., description="Constraint violations")
    detector_confidence_original: float = Field(..., description="Detector confidence on original session")
    detector_confidence_mutated: float = Field(..., description="Detector confidence on mutated session")
    reward: float = Field(..., description="Total reward")
    reward_components: Dict[str, float] = Field(..., description="Reward component breakdown")


class RedAgentMutationResponse(BaseModel):
    """Response for remote Red-Agent evaluation."""

    evaluation_id: str = Field(..., description="Unique evaluation ID")
    n_samples: int = Field(..., description="Number of evaluated samples")
    mean_reward: float = Field(..., description="Average reward")
    mean_realism: float = Field(..., description="Average realism score")
    evasion_rate: float = Field(..., description="Fraction of samples with reduced detector confidence")
    constraint_violation_rate: float = Field(..., description="Average violations per sample")
    baseline_recall: Optional[float] = Field(None, description="Baseline detection recall (if labels provided)")
    mutated_recall: Optional[float] = Field(None, description="Mutated detection recall (if labels provided)")
    baseline_true_positives: Optional[int] = Field(None, description="True positives before mutation")
    mutated_true_positives: Optional[int] = Field(None, description="True positives after mutation")
    positives_count: Optional[int] = Field(None, description="Total positive samples in provided labels")
    results: List[RedAgentMutationItem] = Field(..., description="Per-sample evaluation results")
