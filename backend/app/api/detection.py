"""
Detection API Endpoints.

Session inference and dataset analysis endpoints.
"""

import logging
import uuid
from collections import defaultdict, Counter
from datetime import datetime
from fastapi import APIRouter, HTTPException
from typing import List
import numpy as np

from app.schemas.detection import (
    SessionData,
    SessionBatch,
    DetectionResult,
    BatchDetectionResult,
    DatasetEvaluationRequest,
    DatasetEvaluationResult,
    SessionDetectionSummary,
    HostBehaviorSummary,
    HostCentricSummary,
)
from app.services import detection_service, artifact_service
from src.features.feature_config import FEATURE_INDEX
from src.features.session_builder import load_sessions

logger = logging.getLogger(__name__)

router = APIRouter()


def _session_tensor_to_features(session: np.ndarray, mask: np.ndarray) -> dict:
    """Convert one session tensor into scalar features for summaries and host grouping.
    
    Handles both 10-feature and 12-feature formats.
    Format (10 features): orig_bytes, resp_bytes, orig_pkts, resp_pkts, bytes_per_pkt, packet_ratio, byte_ratio, is_outbound, duration, src_port
    Format (12 features): ...above + dst_port, iat
    """
    real = session[mask] if mask.any() else session[:1]
    num_features = session.shape[1]
    
    if len(real) > 0:
        duration = float(real[:, 8].sum()) if num_features > 8 else 0.0  # duration at index 8
        bytes_in = float(real[:, 0].sum())  # orig_bytes
        bytes_out = float(real[:, 1].sum())  # resp_bytes
        packets_in = float(real[:, 2].sum())  # orig_pkts
        packets_out = float(real[:, 3].sum())  # resp_pkts
        is_outbound = float(real[:, 7].mean()) if num_features > 7 else 0.0  # is_outbound
        src_port = int(np.median(real[:, 9])) if num_features > 9 else 0  # src_port
        dst_port = int(np.median(real[:, 10])) if num_features > 10 else 0  # dst_port
    else:
        duration = bytes_in = bytes_out = packets_in = packets_out = is_outbound = 0.0
        src_port = dst_port = 0

    protocol = "TCP" if is_outbound >= 0.5 else "UDP"
    host_key = f"src_{(src_port // 1024) * 1024}_dst_{(dst_port // 1024) * 1024}_{protocol.lower()}"

    return {
        "duration": duration,
        "bytes_in": bytes_in,
        "bytes_out": bytes_out,
        "packets_in": packets_in,
        "packets_out": packets_out,
        "src_port": src_port,
        "dst_port": dst_port,
        "protocol": protocol,
        "host_key": host_key,
    }


def _build_host_summary(session_rows: List[dict]) -> HostCentricSummary:
    host_groups: dict[str, list[dict]] = defaultdict(list)
    for row in session_rows:
        host_groups[row["host_key"]].append(row)

    summaries: list[HostBehaviorSummary] = []
    for host_key, rows in host_groups.items():
        risk_scores = [float(row["risk_score"]) for row in rows]
        suspicious_count = sum(1 for row in rows if row["is_suspicious"])
        src_port_counts = Counter(int(row["src_port"]) for row in rows)
        dst_port_counts = Counter(int(row["dst_port"]) for row in rows)
        proto_counts = Counter(str(row["protocol"]) for row in rows)

        summaries.append(
            HostBehaviorSummary(
                host_key=host_key,
                session_count=len(rows),
                suspicious_count=suspicious_count,
                suspicious_rate=float(suspicious_count / len(rows)) if rows else 0.0,
                avg_risk_score=float(np.mean(risk_scores)) if risk_scores else 0.0,
                max_risk_score=float(np.max(risk_scores)) if risk_scores else 0.0,
                avg_duration=float(np.mean([float(row["duration"]) for row in rows])) if rows else 0.0,
                dominant_src_port=int(src_port_counts.most_common(1)[0][0]) if src_port_counts else 0,
                dominant_dst_port=int(dst_port_counts.most_common(1)[0][0]) if dst_port_counts else 0,
                dominant_protocol=str(proto_counts.most_common(1)[0][0]).upper() if proto_counts else "TCP",
            )
        )

    summaries.sort(key=lambda item: (item.suspicious_rate, item.avg_risk_score, item.session_count), reverse=True)
    high_risk_hosts = [summary for summary in summaries if summary.suspicious_count > 0]

    return HostCentricSummary(
        total_hosts=len(summaries),
        suspicious_hosts=len(high_risk_hosts),
        avg_sessions_per_host=float(len(session_rows) / len(summaries)) if summaries else 0.0,
        top_hosts=summaries[:10],
        high_risk_hosts=high_risk_hosts[:10],
    )


@router.post("/session", response_model=DetectionResult)
async def analyze_session(session: SessionData):
    """
    Analyze a single session for behavioral risk.
    
    Returns detection result with risk score and confidence.
    """
    try:
        detection_service.initialize()
        
        # Convert session to dict
        session_dict = session.dict()
        
        # Run inference
        risk_score, is_suspicious, confidence = detection_service.infer_single(session_dict)
        
        # Extract features for explanation
        features = detection_service.extract_features(session_dict)
        
        result = DetectionResult(
            session_id=session.session_id,
            risk_score=risk_score,
            is_suspicious=is_suspicious,
            confidence=confidence,
            behavioral_features={
                "duration": float(session.duration),
                "bytes_in": float(session.bytes_in),
                "bytes_out": float(session.bytes_out),
                "packets_in": int(session.packets_in),
                "packets_out": int(session.packets_out),
            },
            timestamp=session.timestamp
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Session analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=BatchDetectionResult)
async def analyze_batch(batch: SessionBatch):
    """
    Analyze a batch of sessions.
    
    Returns per-session results and batch summary.
    """
    try:
        detection_service.initialize()
        
        batch_id = batch.batch_id or str(uuid.uuid4())
        
        # Run inference
        results_list = []
        all_risk_scores = []
        suspicious_count = 0
        
        for session in batch.sessions:
            session_dict = session.dict()
            risk_score, is_suspicious, confidence = detection_service.infer_single(session_dict)
            
            result = DetectionResult(
                session_id=session.session_id,
                risk_score=risk_score,
                is_suspicious=is_suspicious,
                confidence=confidence,
                behavioral_features={
                    "duration": float(session.duration),
                    "bytes_in": float(session.bytes_in),
                    "bytes_out": float(session.bytes_out),
                    "packets_in": int(session.packets_in),
                    "packets_out": int(session.packets_out),
                },
                timestamp=session.timestamp
            )
            
            results_list.append(result)
            all_risk_scores.append(risk_score)
            if is_suspicious:
                suspicious_count += 1
        
        # Compute summary
        import numpy as np
        risk_array = np.array(all_risk_scores)
        
        summary = {
            "batch_id": batch_id,
            "total_sessions": len(batch.sessions),
            "suspicious_count": suspicious_count,
            "detection_rate": float(suspicious_count / len(batch.sessions)),
            "avg_risk_score": float(np.mean(risk_array)),
            "std_risk_score": float(np.std(risk_array)),
            "min_risk_score": float(np.min(risk_array)),
            "max_risk_score": float(np.max(risk_array)),
        }
        
        return BatchDetectionResult(
            batch_id=batch_id,
            results=results_list,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dataset", response_model=DatasetEvaluationResult)
async def evaluate_dataset(request: DatasetEvaluationRequest):
    """
    Evaluate a full dataset NPZ file.
    
    Runs inference on all samples and returns aggregate statistics.
    """
    try:
        logger.info(f"Starting dataset evaluation: npz_file_id={request.npz_file_id}, sample_limit={request.sample_limit}")
        detection_service.initialize()
        logger.info("Detection service initialized")
        
        # Get file from artifact storage
        file_path = artifact_service.get_file_path(request.npz_file_id)
        if not file_path:
            raise HTTPException(status_code=404, detail="NPZ file not found")
        
        logger.info(f"Loading sessions from {file_path}")
        sessions_data, labels, masks = load_sessions(str(file_path))
        logger.info(f"Loaded {len(sessions_data)} sessions")
        
        if request.sample_limit:
            sessions_data = sessions_data[:request.sample_limit]
            labels = labels[:request.sample_limit]
            masks = masks[:request.sample_limit]
            logger.info(f"Limited to {len(sessions_data)} samples")

        session_rows: list[dict] = []
        all_risk_scores: list[float] = []
        suspicious_count = 0

        logger.info(f"Starting batch inference with batch_size={request.batch_size}")
        for i in range(0, len(sessions_data), request.batch_size):
            batch_sessions = sessions_data[i:i + request.batch_size]
            batch_masks = masks[i:i + request.batch_size]
            logger.debug(f"Processing batch {i//request.batch_size + 1} ({i}-{min(i+request.batch_size, len(sessions_data))})")
            
            batch_results = detection_service.infer_tensor_batch(batch_sessions, batch_masks)

            for offset, (risk_score, is_suspicious, confidence) in enumerate(batch_results):
                row = _session_tensor_to_features(batch_sessions[offset], batch_masks[offset])
                row.update({
                    "session_id": f"session_{i + offset + 1}",
                    "risk_score": risk_score,
                    "is_suspicious": is_suspicious,
                    "confidence": confidence,
                })
                session_rows.append(row)
                all_risk_scores.append(risk_score)
                if is_suspicious:
                    suspicious_count += 1

        logger.info(f"Inference complete: {len(session_rows)} sessions processed, {suspicious_count} suspicious")
        
        risk_array = np.array(all_risk_scores, dtype=np.float32)
        risk_bins = np.histogram(risk_array, bins=10, range=(0, 1)) if len(risk_array) else (np.array([]), np.array([]))
        risk_distribution = {
            "bins": [float(x) for x in (risk_bins[1] if len(risk_bins[1]) else np.linspace(0.0, 1.0, 11))],
            "counts": [int(x) for x in (risk_bins[0] if len(risk_bins[0]) else np.zeros(10, dtype=int))]
        }

        logger.info("Building host summary")
        host_summary = _build_host_summary(session_rows)

        evaluation_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()
        
        logger.info(f"Creating result object for evaluation_id={evaluation_id}")
        # Convert raw session tensors and masks to Python-native lists for JSON
        session_tensors_list = [session.tolist() for session in sessions_data]
        session_masks_list = [[bool(x) for x in mask] for mask in masks]

        result = DatasetEvaluationResult(
            evaluation_id=evaluation_id,
            total_sessions=len(sessions_data),
            suspicious_count=suspicious_count,
            detection_rate=float(suspicious_count / len(sessions_data)) if len(sessions_data) else 0.0,
            avg_risk_score=float(np.mean(risk_array)) if len(risk_array) else 0.0,
            std_risk_score=float(np.std(risk_array)) if len(risk_array) else 0.0,
            risk_distribution=risk_distribution,
            session_results=[
                SessionDetectionSummary(
                    session_id=row["session_id"],
                    host_key=row["host_key"],
                    risk_score=row["risk_score"],
                    is_suspicious=row["is_suspicious"],
                    confidence=row["confidence"],
                    duration=row["duration"],
                    bytes_in=row["bytes_in"],
                    bytes_out=row["bytes_out"],
                    src_port=row["src_port"],
                    dst_port=row["dst_port"],
                )
                for row in session_rows
            ],
            host_summary=host_summary,
            session_tensors=session_tensors_list,
            session_masks=session_masks_list,
            created_at=created_at,
        )
        
        logger.info(f"Dataset evaluation completed successfully: {result.evaluation_id}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dataset evaluation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
