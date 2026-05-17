"""
Artifact Service - File upload, storage, and retrieval.

Manages persistent artifacts (datasets, reports, etc.).
"""

import os
import uuid
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
from datetime import datetime

from app.core.config import settings

logger = logging.getLogger(__name__)


class ArtifactService:
    """Service for managing artifacts."""
    
    def __init__(self):
        """Initialize artifact service."""
        self.artifacts_dir = Path(settings.ARTIFACTS_DIR)
        self.artifacts_dir.mkdir(exist_ok=True)
        self.metadata_file = self.artifacts_dir / ".metadata.json"
        self._load_metadata()
    
    def _load_metadata(self):
        """Load artifacts metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
                self.metadata = {}
        else:
            self.metadata = {}
    
    def _save_metadata(self):
        """Save artifacts metadata to disk."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def upload_file(
        self,
        file_content: bytes,
        filename: str,
        file_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Upload a file to artifact storage.
        
        Args:
            file_content: File bytes
            filename: Original filename
            file_type: Type (npz, json, jsonl, csv)
            metadata: Additional metadata
            
        Returns:
            File metadata
        """
        try:
            # Generate file ID
            file_id = str(uuid.uuid4())
            file_path = self.artifacts_dir / f"{file_id}_{filename}"
            
            # Save file
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            file_size = len(file_content)
            
            # Extract metadata for NPZ files
            sample_count = None
            feature_count = None
            if file_type == "npz":
                try:
                    with np.load(file_path, allow_pickle=True) as data:
                        if 'X' in data:
                            sample_count = int(data['X'].shape[0])
                            feature_count = int(data['X'].shape[-1]) if data['X'].ndim >= 3 else None
                        elif 'sessions' in data:
                            sample_count = int(data['sessions'].shape[0])
                            feature_count = int(data['sessions'].shape[1]) if data['sessions'].ndim >= 3 else None
                except Exception as e:
                    logger.warning(f"Failed to extract NPZ metadata: {e}")
            
            # Store metadata
            artifact_meta = {
                "file_id": file_id,
                "filename": filename,
                "file_type": file_type,
                "file_path": str(file_path),
                "size_bytes": file_size,
                "upload_timestamp": datetime.utcnow().isoformat(),
                "sample_count": sample_count,
                "feature_count": feature_count,
                "custom_metadata": metadata or {}
            }
            
            self.metadata[file_id] = artifact_meta
            self._save_metadata()
            
            logger.info(f"Uploaded file {file_id}: {filename} ({file_size} bytes)")
            
            return artifact_meta
            
        except Exception as e:
            logger.error(f"File upload failed: {e}")
            raise
    
    def get_file(self, file_id: str) -> Optional[bytes]:
        """Retrieve file by ID."""
        try:
            if file_id not in self.metadata:
                logger.warning(f"File not found: {file_id}")
                return None
            
            file_path = Path(self.metadata[file_id]['file_path'])
            
            if not file_path.exists():
                logger.warning(f"File path does not exist: {file_path}")
                return None
            
            with open(file_path, 'rb') as f:
                return f.read()
                
        except Exception as e:
            logger.error(f"File retrieval failed: {e}")
            return None
    
    def get_file_path(self, file_id: str) -> Optional[Path]:
        """Get file path by ID."""
        # Reload metadata to ensure fresh data
        self._load_metadata()
        
        logger.info(f"Looking for file_id: {file_id}")
        logger.info(f"Available file IDs in metadata: {list(self.metadata.keys())}")
        
        if file_id not in self.metadata:
            logger.error(f"File ID not found in metadata: {file_id}")
            return None
        
        file_meta = self.metadata[file_id]
        file_path = Path(file_meta['file_path'])
        
        logger.info(f"File path for {file_id}: {file_path}")
        logger.info(f"File path exists: {file_path.exists()}")
        
        if not file_path.exists():
            logger.error(f"File path does not exist on disk: {file_path}")
            return None
            
        return file_path
    
    def get_metadata(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get file metadata."""
        return self.metadata.get(file_id)
    
    def list_files(
        self,
        file_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List artifacts with optional filtering."""
        files = list(self.metadata.values())
        
        if file_type:
            files = [f for f in files if f['file_type'] == file_type]
        
        # Sort by upload time (newest first)
        files.sort(key=lambda x: x['upload_timestamp'], reverse=True)
        
        # Paginate
        return files[offset:offset + limit]
    
    def delete_file(self, file_id: str) -> bool:
        """Delete a file."""
        try:
            if file_id not in self.metadata:
                logger.warning(f"File not found for deletion: {file_id}")
                return False
            
            file_path = Path(self.metadata[file_id]['file_path'])
            
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted file: {file_path}")
            
            del self.metadata[file_id]
            self._save_metadata()
            
            return True
            
        except Exception as e:
            logger.error(f"File deletion failed: {e}")
            return False
    
    def save_report(
        self,
        report_id: str,
        report_type: str,
        title: str,
        content: Dict[str, Any],
        description: Optional[str] = None,
        related_artifacts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Save a report as a JSON artifact.
        
        Returns:
            Report metadata
        """
        try:
            # Create report JSON
            report_data = {
                "report_id": report_id,
                "report_type": report_type,
                "title": title,
                "description": description,
                "created_at": datetime.utcnow().isoformat(),
                "content": content
            }
            
            filename = f"{report_id}.json"
            file_content = json.dumps(report_data, indent=2).encode('utf-8')
            
            artifact_meta = self.upload_file(
                file_content=file_content,
                filename=filename,
                file_type="json",
                metadata={
                    "report_id": report_id,
                    "report_type": report_type,
                    "related_artifacts": related_artifacts or []
                }
            )
            
            logger.info(f"Saved report {report_id}: {title}")
            return artifact_meta
            
        except Exception as e:
            logger.error(f"Report save failed: {e}")
            raise
