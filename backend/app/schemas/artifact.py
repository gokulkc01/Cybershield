"""
Request/Response schemas for Artifact Management APIs.

File upload, storage, and retrieval schemas.
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class FileUploadResponse(BaseModel):
    """Response after file upload."""
    
    file_id: str = Field(..., description="Unique file identifier")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="Type: npz, jsonl, json, csv")
    size_bytes: int = Field(..., description="File size in bytes")
    upload_timestamp: str = Field(..., description="Upload timestamp")
    
    # Metadata for NPZ files
    sample_count: Optional[int] = Field(None, description="Number of samples in dataset")
    feature_count: Optional[int] = Field(None, description="Number of features")


class ArtifactMetadata(BaseModel):
    """Metadata for stored artifacts."""
    
    artifact_id: str = Field(..., description="Artifact ID")
    artifact_type: str = Field(..., description="Type: dataset, result, report, metadata")
    filename: str = Field(..., description="Filename")
    size_bytes: int = Field(..., description="Size in bytes")
    created_at: str = Field(..., description="Creation timestamp")
    
    # Related items
    source_file_id: Optional[str] = Field(None, description="Source file if derived")
    related_artifacts: List[str] = Field(default_factory=list, description="Related artifact IDs")
    
    # Content metadata
    metadata: dict = Field(default_factory=dict, description="Type-specific metadata")


class ArtifactListResponse(BaseModel):
    """List of artifacts."""
    
    artifacts: List[ArtifactMetadata] = Field(..., description="List of artifacts")
    total_count: int = Field(..., description="Total count")
    page: int = Field(..., description="Page number")
    page_size: int = Field(..., description="Items per page")


class ReportMetadata(BaseModel):
    """Metadata for stored reports."""
    
    report_id: str = Field(..., description="Report ID")
    report_type: str = Field(..., description="Type: robustness, evaluation, failure, mutation")
    title: str = Field(..., description="Report title")
    description: Optional[str] = Field(None, description="Report description")
    created_at: str = Field(..., description="Creation timestamp")
    
    # Related artifacts
    related_artifacts: List[str] = Field(default_factory=list, description="Related file IDs")
    
    # Summary
    summary: dict = Field(..., description="Report summary")
