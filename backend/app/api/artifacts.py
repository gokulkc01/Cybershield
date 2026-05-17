"""
Artifacts API Endpoints.

File upload, retrieval, management, and listing.
"""

import logging
import uuid
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import StreamingResponse

from app.schemas.artifact import (
    FileUploadResponse,
    ArtifactMetadata,
    ArtifactListResponse,
    ReportMetadata
)
from app.services import artifact_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/upload", response_model=FileUploadResponse)
async def upload_artifact(
    file: UploadFile = File(...),
    file_type: str = Form(..., description="Type: npz, jsonl, json, csv")
):
    """
    Upload a file artifact.
    
    Supports NPZ datasets, JSON reports, JSONL metadata, CSV files.
    """
    try:
        # Read file
        content = await file.read()
        
        # Upload to artifact storage
        metadata = artifact_service.upload_file(
            file_content=content,
            filename=file.filename or "unknown",
            file_type=file_type
        )
        
        response = FileUploadResponse(
            file_id=metadata['file_id'],
            filename=metadata['filename'],
            file_type=metadata['file_type'],
            size_bytes=metadata['size_bytes'],
            upload_timestamp=metadata['upload_timestamp'],
            sample_count=metadata.get('sample_count'),
            feature_count=metadata.get('feature_count')
        )
        
        return response
        
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/file/{file_id}")
async def download_file(file_id: str):
    """
    Download a file artifact.
    
    Returns file as binary stream.
    """
    try:
        file_content = artifact_service.get_file(file_id)
        if not file_content:
            raise HTTPException(status_code=404, detail="File not found")
        
        metadata = artifact_service.get_metadata(file_id)
        
        return StreamingResponse(
            iter([file_content]),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={metadata['filename']}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File download failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metadata/{file_id}", response_model=ArtifactMetadata)
async def get_artifact_metadata(file_id: str):
    """
    Get metadata for an artifact.
    """
    try:
        metadata = artifact_service.get_metadata(file_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="Artifact not found")
        
        response = ArtifactMetadata(
            artifact_id=metadata['file_id'],
            artifact_type=metadata['file_type'],
            filename=metadata['filename'],
            size_bytes=metadata['size_bytes'],
            created_at=metadata['upload_timestamp'],
            source_file_id=None,
            related_artifacts=[],
            metadata=metadata.get('custom_metadata', {})
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Metadata retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list", response_model=ArtifactListResponse)
async def list_artifacts(
    file_type: str = Query(None, description="Filter by type"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """
    List artifacts with optional filtering.
    
    Returns paginated list of artifacts.
    """
    try:
        artifacts = artifact_service.list_files(
            file_type=file_type,
            limit=limit,
            offset=offset
        )
        
        artifacts_list = [
            ArtifactMetadata(
                artifact_id=a['file_id'],
                artifact_type=a['file_type'],
                filename=a['filename'],
                size_bytes=a['size_bytes'],
                created_at=a['upload_timestamp'],
                source_file_id=None,
                related_artifacts=[],
                metadata=a.get('custom_metadata', {})
            )
            for a in artifacts
        ]
        
        return ArtifactListResponse(
            artifacts=artifacts_list,
            total_count=len(artifact_service.metadata),
            page=offset // limit + 1,
            page_size=limit
        )
        
    except Exception as e:
        logger.error(f"Artifact listing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/file/{file_id}")
async def delete_artifact(file_id: str):
    """
    Delete an artifact.
    """
    try:
        success = artifact_service.delete_file(file_id)
        if not success:
            raise HTTPException(status_code=404, detail="File not found or already deleted")
        
        return {"status": "deleted", "file_id": file_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File deletion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/report")
async def save_report(
    report_id: str = Form(None),
    report_type: str = Form(..., description="Type: robustness, evaluation, failure, mutation"),
    title: str = Form(...),
    description: str = Form(None),
    file: UploadFile = File(...)
):
    """
    Save a report artifact.
    
    Associates report with type and metadata.
    """
    try:
        if not report_id:
            report_id = str(uuid.uuid4())
        
        content = await file.read()
        
        metadata = artifact_service.upload_file(
            file_content=content,
            filename=file.filename or f"{report_id}.json",
            file_type="json",
            metadata={
                "report_id": report_id,
                "report_type": report_type,
                "title": title,
                "description": description
            }
        )
        
        response = ReportMetadata(
            report_id=report_id,
            report_type=report_type,
            title=title,
            description=description,
            created_at=metadata['upload_timestamp'],
            related_artifacts=[],
            summary={
                "file_id": metadata['file_id'],
                "size_bytes": metadata['size_bytes']
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Report save failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
