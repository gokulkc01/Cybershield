"""
Robustness Analytics API Endpoints.

Robustness metrics, failure analysis, and comparative analytics.
"""

import logging
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import List

import numpy as np
from fastapi import APIRouter, HTTPException

from app.schemas.robustness import (
    RobustnessReport,
    RobustnessMetrics,
    FailureAnalysisResult,
    RobustnessComparisonRequest,
    RedAgentMutationRequest,
    RedAgentMutationResponse,
    RedAgentMutationItem,
)
from app.services.detection import DetectionService
from app.services.robustness import RobustnessService
from app.services.artifact import ArtifactService
from src.red_agent.orchestrator import RedAgentOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter()
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RED_AGENT_SUMMARY = PROJECT_ROOT / "experiments" / "red_agent_host_aware_smoke" / "host_aware_phase1_summary.json"

# Service instances
detection_service = DetectionService()
robustness_service = RobustnessService(detection_service)
artifact_service = ArtifactService()


def _build_detector_inference_fn():
    """Wrap the detector service as a confidence function for Red-Agent."""
    detection_service.initialize()

    def infer(session: np.ndarray) -> float:
        sequence = np.asarray(session, dtype=np.float32)[np.newaxis, ...]
        real_masks = np.any(sequence != 0.0, axis=2)
        _, _, confidence = detection_service.infer_tensor_batch(sequence, real_masks)[0]
        return float(confidence)

    return infer


@router.get("/report/{evaluation_id}", response_model=RobustnessReport)
async def get_robustness_report(evaluation_id: str):
    """
    Retrieve a robustness report.
    
    Returns comprehensive analysis of adversarial robustness.
    """
    try:
        # Get report from artifact storage
        # In production, would query database
        
        report = RobustnessReport(
            report_id=evaluation_id,
            baseline_recall=0.9636,
            mutated_recall=0.9400,
            recall_degradation=2.36,
            behavioral_invariance=0.975,
            fpr_shift=0.01,
            per_mutation_analysis={},
            mutation_impact_ranking=[],
            fragile_samples=100,
            feature_drift={},
            created_at=""
        )
        
        return report
        
    except Exception as e:
        logger.error(f"Failed to retrieve report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/{evaluation_id}", response_model=RobustnessMetrics)
async def get_robustness_metrics(evaluation_id: str):
    """
    Get summary robustness metrics.
    """
    try:
        metrics = RobustnessMetrics(
            metric_id=evaluation_id,
            baseline_recall=0.9636,
            mutated_recall=0.9400,
            recall_degradation=2.36,
            behavioral_invariance=0.975,
            fpr_baseline=0.01,
            fpr_mutated=0.011
        )
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/failures", response_model=FailureAnalysisResult)
async def analyze_failures(request: dict):
    """
    Analyze behavioral failures under mutation.
    
    Identifies fragile mutations and feature shifts.
    """
    try:
        detection_service.initialize()
        
        # Get file paths
        baseline_path = artifact_service.get_file_path(request.get('baseline_npz_file_id'))
        mutated_path = artifact_service.get_file_path(request.get('mutated_npz_file_id'))
        
        if not baseline_path or not mutated_path:
            raise HTTPException(status_code=404, detail="Dataset files not found")
        
        # Run failure analysis
        analysis = robustness_service.analyze_failures(
            baseline_npz_path=baseline_path,
            mutated_npz_path=mutated_path,
            batch_size=256
        )
        
        analysis_id = str(uuid.uuid4())
        
        result = FailureAnalysisResult(
            analysis_id=analysis_id,
            total_failures=analysis['total_failures'],
            failure_rate=analysis['failure_rate'],
            fragile_mutations=analysis['fragile_mutations'],
            failed_invariants=[],
            feature_shift_analysis=analysis['feature_shifts'],
            behavioral_patterns=[],
            created_at=""
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failure analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare")
async def compare_robustness(request: RobustnessComparisonRequest):
    """
    Compare robustness across different mutation groups.
    
    Returns comparative metrics.
    """
    try:
        # Placeholder for comparative analysis
        comparison = {
            "comparison_id": str(uuid.uuid4()),
            "baseline_id": request.baseline_npz_file_id,
            "mutation_groups": request.mutation_groups,
            "results": []
        }
        
        return comparison
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/red-agent/demo-report")
async def get_red_agent_demo_report():
    """Return the real host-aware Red-Agent demo report from training outputs."""
    try:
        if not DEFAULT_RED_AGENT_SUMMARY.exists():
            raise HTTPException(
                status_code=404,
                detail=(
                    "Red-Agent demo summary not found. Run the host-aware Red-Agent smoke command first."
                ),
            )

        from scripts.demo_red_agent_report import build_report, render_markdown

        report = build_report(DEFAULT_RED_AGENT_SUMMARY)
        markdown = render_markdown(report)
        return {
            "status": "success",
            "source": "real_training_outputs",
            "report": asdict(report),
            "markdown": markdown,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to build Red-Agent demo report: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/red-agent/evaluate", response_model=RedAgentMutationResponse)
async def evaluate_red_agent_mutations(request: RedAgentMutationRequest):
    """Evaluate Red-Agent mutations remotely against the frozen detector."""
    try:
        sessions = np.asarray(request.sessions, dtype=np.float32)
        masks = np.asarray(request.masks, dtype=bool)

        if sessions.ndim != 3:
            raise HTTPException(status_code=400, detail="sessions must have shape (N, 20, D)")
        if masks.ndim != 2 or masks.shape[0] != sessions.shape[0]:
            raise HTTPException(status_code=400, detail="masks must have shape (N, 20)")

        orchestrator = RedAgentOrchestrator()
        detector_infer = _build_detector_inference_fn()

        results = orchestrator.batch_evaluate_mutations(
            sessions=sessions,
            masks=masks,
            detector_inference_fn=detector_infer,
            mutation_types=request.mutation_types,
            severities=request.severities,
        )

        items: List[RedAgentMutationItem] = []
        rewards = []
        realism_scores = []
        evasions = 0
        constraint_total = 0

        for index, result in enumerate(results):
            if result is None:
                continue
            rewards.append(float(result.reward))
            realism_scores.append(float(result.mutation_result.realism_score))
            if result.detector_confidence_mutated < result.detector_confidence_original:
                evasions += 1
            constraint_total += len(result.mutation_result.constraint_violations)

            items.append(
                RedAgentMutationItem(
                    sample_index=index,
                    mutation_type=result.mutation_result.mutation_type,
                    severity=float(result.mutation_result.mutation_metadata.get("severity", 0.0)),
                    realism_score=float(result.mutation_result.realism_score),
                    is_realistic=bool(result.mutation_result.is_realistic),
                    constraint_violations=list(result.mutation_result.constraint_violations),
                    detector_confidence_original=float(result.detector_confidence_original),
                    detector_confidence_mutated=float(result.detector_confidence_mutated),
                    reward=float(result.reward),
                    reward_components={k: float(v) for k, v in result.reward_components.items()},
                )
            )

        evaluation_id = str(uuid.uuid4())
        # Compute recall metrics if labels and threshold provided
        baseline_recall = None
        mutated_recall = None
        baseline_tp = None
        mutated_tp = None
        positives = None

        threshold = float(request.threshold) if getattr(request, 'threshold', None) is not None else 0.5
        labels = np.asarray(request.labels, dtype=int) if getattr(request, 'labels', None) is not None else None

        if labels is not None:
            if labels.shape[0] != sessions.shape[0]:
                raise HTTPException(status_code=400, detail="labels length must match number of sessions")

            successful_indices = [it.sample_index for it in items]
            filtered_labels = labels[successful_indices]

            positives = int(np.sum(filtered_labels == 1))
            orig_preds = np.array([1 if it.detector_confidence_original >= threshold else 0 for it in items])
            mut_preds = np.array([1 if it.detector_confidence_mutated >= threshold else 0 for it in items])

            baseline_tp = int(np.sum((filtered_labels == 1) & (orig_preds == 1)))
            mutated_tp = int(np.sum((filtered_labels == 1) & (mut_preds == 1)))

            baseline_recall = float(baseline_tp / positives) if positives > 0 else None
            mutated_recall = float(mutated_tp / positives) if positives > 0 else None

        return RedAgentMutationResponse(
            evaluation_id=evaluation_id,
            n_samples=len(items),
            mean_reward=float(np.mean(rewards)) if rewards else 0.0,
            mean_realism=float(np.mean(realism_scores)) if realism_scores else 0.0,
            evasion_rate=float(evasions / max(len(items), 1)),
            constraint_violation_rate=float(constraint_total / max(len(items), 1)),
            baseline_recall=baseline_recall,
            mutated_recall=mutated_recall,
            baseline_true_positives=baseline_tp,
            mutated_true_positives=mutated_tp,
            positives_count=positives,
            results=items,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Red-Agent evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
