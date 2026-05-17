"""
Request/Response schemas for Detection APIs.

Behavioral detection and inference schemas.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class SessionDetectionSummary(BaseModel):
    """Per-session detection summary used in dataset evaluation results."""

    session_id: str = Field(..., description="Synthetic or source session identifier")
    host_key: str = Field(..., description="Host or behavioral cohort key")
    risk_score: float = Field(..., description="Risk score [0, 1]")
    is_suspicious: bool = Field(..., description="Suspicious classification")
    confidence: float = Field(..., description="Model confidence")
    duration: float = Field(..., description="Session duration")
    bytes_in: float = Field(..., description="Inbound bytes")
    bytes_out: float = Field(..., description="Outbound bytes")
    src_port: int = Field(..., description="Source port")
    dst_port: int = Field(..., description="Destination port")


class HostBehaviorSummary(BaseModel):
    """Aggregated host-centric behavioral summary."""

    host_key: str = Field(..., description="Host or cohort key")
    session_count: int = Field(..., description="Number of sessions in the cohort")
    suspicious_count: int = Field(..., description="Suspicious sessions in the cohort")
    suspicious_rate: float = Field(..., description="Suspicious session ratio")
    avg_risk_score: float = Field(..., description="Average risk score")
    max_risk_score: float = Field(..., description="Maximum risk score")
    avg_duration: float = Field(..., description="Average session duration")
    dominant_src_port: int = Field(..., description="Most common source port")
    dominant_dst_port: int = Field(..., description="Most common destination port")
    dominant_protocol: str = Field(..., description="Dominant protocol")


class HostCentricSummary(BaseModel):
    """Host-centric summary for the uploaded dataset."""

    total_hosts: int = Field(..., description="Number of host/cohort groups")
    suspicious_hosts: int = Field(..., description="Number of hosts with suspicious traffic")
    avg_sessions_per_host: float = Field(..., description="Average sessions per host")
    top_hosts: List[HostBehaviorSummary] = Field(..., description="Top host/cohort summaries")
    high_risk_hosts: List[HostBehaviorSummary] = Field(..., description="Hosts/cohorts with suspicious sessions")


class SessionData(BaseModel):
    """Single network session with 10 behavioral features."""
    
    session_id: str = Field(..., description="Unique session identifier")
    duration: float = Field(..., description="Session duration in seconds")
    bytes_in: float = Field(..., description="Inbound bytes")
    bytes_out: float = Field(..., description="Outbound bytes")
    packets_in: int = Field(..., description="Inbound packet count")
    packets_out: int = Field(..., description="Outbound packet count")
    protocol: str = Field(..., description="Protocol (TCP/UDP)")
    src_port: int = Field(..., description="Source port")
    dst_port: int = Field(..., description="Destination port")
    timestamp: float = Field(..., description="Session start timestamp")
    

class SessionBatch(BaseModel):
    """Batch of sessions for inference."""
    
    sessions: List[SessionData] = Field(..., description="List of sessions")
    batch_id: Optional[str] = Field(None, description="Batch identifier")
    

class DetectionResult(BaseModel):
    """Detection result for a single session."""
    
    session_id: str = Field(..., description="Session ID")
    risk_score: float = Field(..., description="Risk score [0, 1]")
    is_suspicious: bool = Field(..., description="Suspicious classification")
    confidence: float = Field(..., description="Model confidence")
    behavioral_features: dict = Field(..., description="Extracted behavioral features")
    timestamp: float = Field(..., description="Detection timestamp")


class BatchDetectionResult(BaseModel):
    """Detection results for a batch."""
    
    batch_id: str = Field(..., description="Batch identifier")
    results: List[DetectionResult] = Field(..., description="Per-session results")
    summary: dict = Field(..., description="Batch summary statistics")
    

class DatasetEvaluationRequest(BaseModel):
    """Request to evaluate a dataset NPZ file."""
    
    npz_file_id: str = Field(..., description="File ID from artifact storage")
    sample_limit: Optional[int] = Field(None, description="Limit evaluations (None=all)")
    batch_size: int = Field(256, description="Batch size for inference")


class DatasetEvaluationResult(BaseModel):
    """Results from dataset evaluation."""
    
    evaluation_id: str = Field(..., description="Unique evaluation ID")
    total_sessions: int = Field(..., description="Sessions evaluated")
    suspicious_count: int = Field(..., description="Suspicious sessions detected")
    detection_rate: float = Field(..., description="Proportion detected")
    avg_risk_score: float = Field(..., description="Mean risk score")
    std_risk_score: float = Field(..., description="Std deviation of risk scores")
    risk_distribution: dict = Field(..., description="Histogram of risk scores")
    session_results: List[SessionDetectionSummary] = Field(..., description="Per-session detection results")
    host_summary: HostCentricSummary = Field(..., description="Host-centric behavioral summary")
    # Optional raw session tensors and masks (SESSION_LEN x FEATURE_DIM per session)
    session_tensors: Optional[List[List[List[float]]]] = Field(None, description="Optional raw session tensors for each evaluated session")
    session_masks: Optional[List[List[bool]]] = Field(None, description="Optional masks indicating active timesteps per session")
    created_at: str = Field(..., description="Creation timestamp")
