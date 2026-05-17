"""Services for CyberShield backend."""

from app.services.artifact import ArtifactService
from app.services.detection import DetectionService

# Shared service instances (singletons across all endpoints)
artifact_service = ArtifactService()
detection_service = DetectionService()

__all__ = ["artifact_service", "detection_service"]
