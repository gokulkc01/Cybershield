"""
Mutation API Endpoints.

Adversarial mutation and robustness evaluation endpoints.
"""

import logging
import uuid
from fastapi import APIRouter, HTTPException
from typing import List

from app.schemas.mutation import (
    MutationOperator,
    MutationRequest,
    MutationResult,
    MutationEvaluationRequest,
    MutationEvaluationResult,
    AvailableMutationsResponse
)
from app.services.mutation import MutationService
from app.services.detection import DetectionService
from app.services.robustness import RobustnessService
from app.services.artifact import ArtifactService
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Service instances
mutation_service = MutationService()
detection_service = DetectionService()
robustness_service = RobustnessService(detection_service)
artifact_service = ArtifactService()


@router.get("/available", response_model=AvailableMutationsResponse)
async def list_available_mutations():
    """
    List available mutation operators.
    
    Returns all registered mutations with descriptions.
    """
    try:
        mutation_service.initialize()
        mutations_info = mutation_service.list_mutations()
        
        mutations_list = list(mutations_info.values())
        
        return AvailableMutationsResponse(
            mutations=mutations_list,
            total_count=len(mutations_list)
        )
        
    except Exception as e:
        logger.error(f"Failed to list mutations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/apply", response_model=MutationResult)
async def apply_mutations(request: MutationRequest):
    """
    Apply mutations to a dataset.
    
    Generates mutated session dataset with metadata tracking.
    """
    try:
        mutation_service.initialize()
        
        # Get source file
        source_file_path = artifact_service.get_file_path(request.source_npz_file_id)
        if not source_file_path:
            raise HTTPException(status_code=404, detail="Source NPZ file not found")
        
        mutation_id = str(uuid.uuid4())
        
        logger.info(f"Applying mutations to {request.source_npz_file_id}")
        logger.info(f"Mutations: {[m.mutation_type for m in request.mutations]}")
        
        # Convert request mutations to dicts
        mutations_list = [
            {
                "mutation_type": m.mutation_type,
                "severity": m.severity,
                "params": m.params
            }
            for m in request.mutations
        ]
        
        # Apply mutations
        mutated_npz_path, metadata_path = mutation_service.batch_mutate(
            dataset_path=source_file_path,
            mutations=mutations_list,
            base_seed=request.base_seed,
            sample_limit=request.sample_limit,
            batch_size=request.batch_size
        )
        
        # Upload artifacts
        with open(mutated_npz_path, 'rb') as f:
            mutated_artifact = artifact_service.upload_file(
                file_content=f.read(),
                filename=f"mutated_{mutation_id}.npz",
                file_type="npz"
            )
        
        with open(metadata_path, 'rb') as f:
            metadata_artifact = artifact_service.upload_file(
                file_content=f.read(),
                filename=f"metadata_{mutation_id}.jsonl",
                file_type="jsonl"
            )
        
        result = MutationResult(
            mutation_id=mutation_id,
            source_file_id=request.source_npz_file_id,
            mutations_applied=request.mutations,
            samples_mutated=mutated_artifact.get('sample_count', 0),
            output_npz_file_id=mutated_artifact['file_id'],
            metadata_file_id=metadata_artifact['file_id'],
            created_at=mutated_artifact['upload_timestamp']
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Mutation application failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate", response_model=MutationEvaluationResult)
async def evaluate_mutation_robustness(request: MutationEvaluationRequest):
    """
    Evaluate robustness against mutations.
    
    Compares baseline vs mutated detection performance.
    """
    try:
        detection_service.initialize()
        robustness_service = RobustnessService(detection_service)
        
        # Get file paths
        baseline_path = artifact_service.get_file_path(request.baseline_npz_file_id)
        mutated_path = artifact_service.get_file_path(request.mutated_npz_file_id)
        
        if not baseline_path or not mutated_path:
            raise HTTPException(status_code=404, detail="Dataset files not found")
        
        logger.info(f"Evaluating robustness: baseline={request.baseline_npz_file_id}, mutated={request.mutated_npz_file_id}")
        
        # Run robustness evaluation
        report = robustness_service.evaluate_robustness(
            baseline_npz_path=baseline_path,
            mutated_npz_path=mutated_path,
            batch_size=request.batch_size
        )
        
        evaluation_id = str(uuid.uuid4())
        
        result = MutationEvaluationResult(
            evaluation_id=evaluation_id,
            baseline_recall=report['baseline_recall'],
            mutated_recall=report['mutated_recall'],
            recall_degradation=report['recall_degradation'],
            fpr_shift=report['fpr_shift'],
            behavioral_invariance=report['behavioral_invariance'],
            per_mutation_metrics={},  # Could be expanded
            fragile_mutations={},  # Could be expanded with metadata
            created_at=""
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Robustness evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
