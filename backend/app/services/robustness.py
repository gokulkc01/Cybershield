"""
Robustness Service - Adversarial robustness evaluation and analytics.

Computes robustness metrics and failure analysis.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
from collections import defaultdict

from app.services.detection import DetectionService

logger = logging.getLogger(__name__)


class RobustnessService:
    """Service for robustness evaluation and analytics."""
    
    def __init__(self, detection_service: DetectionService = None):
        """Initialize robustness service."""
        self.detection_service = detection_service or DetectionService()
        if not self.detection_service._is_initialized:
            self.detection_service.initialize()
    
    def evaluate_robustness(
        self,
        baseline_npz_path: Path,
        mutated_npz_path: Path,
        batch_size: int = 256
    ) -> Dict[str, Any]:
        """
        Evaluate robustness by comparing baseline vs mutated detection.
        
        Returns comprehensive robustness report.
        """
        try:
            # Load datasets
            baseline_data = np.load(baseline_npz_path)
            mutated_data = np.load(mutated_npz_path)
            
            baseline_sessions = baseline_data['sessions']
            mutated_sessions = mutated_data['sessions']
            baseline_labels = baseline_data.get('labels', np.ones(len(baseline_sessions)))
            
            logger.info(f"Evaluating {len(baseline_sessions)} baseline samples")
            logger.info(f"Evaluating {len(mutated_sessions)} mutated samples")
            
            # Run baseline inference
            baseline_predictions = self._batch_predict(baseline_sessions, batch_size)
            baseline_recall = self._compute_recall(baseline_predictions, baseline_labels)
            baseline_risk_scores = baseline_predictions[:, 0]
            
            # Run mutated inference
            mutated_predictions = self._batch_predict(mutated_sessions, batch_size)
            mutated_recall = self._compute_recall(mutated_predictions, baseline_labels)
            mutated_risk_scores = mutated_predictions[:, 0]
            
            # Compute robustness metrics
            recall_degradation = baseline_recall - mutated_recall
            behavioral_invariance = mutated_recall / max(baseline_recall, 1e-6)
            
            # FPR shift (estimated from risk score changes)
            fpr_shift = np.mean(np.abs(mutated_risk_scores - baseline_risk_scores))
            
            # Identify fragile samples
            fragile_mask = (baseline_predictions[:, 1] == 1) & (mutated_predictions[:, 1] == 0)
            fragile_count = np.sum(fragile_mask)
            
            report = {
                "baseline_recall": float(baseline_recall),
                "mutated_recall": float(mutated_recall),
                "recall_degradation": float(recall_degradation),
                "recall_degradation_pct": float(recall_degradation * 100),
                "behavioral_invariance": float(behavioral_invariance),
                "fpr_shift": float(fpr_shift),
                "fragile_samples": int(fragile_count),
                "fragile_rate": float(fragile_count / len(baseline_sessions)),
                "mean_baseline_score": float(np.mean(baseline_risk_scores)),
                "mean_mutated_score": float(np.mean(mutated_risk_scores)),
                "std_baseline_score": float(np.std(baseline_risk_scores)),
                "std_mutated_score": float(np.std(mutated_risk_scores)),
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Robustness evaluation failed: {e}")
            raise
    
    def analyze_failures(
        self,
        baseline_npz_path: Path,
        mutated_npz_path: Path,
        metadata_jsonl_path: Path = None,
        batch_size: int = 256
    ) -> Dict[str, Any]:
        """
        Analyze behavioral failures under mutation.
        
        Returns failure analysis with fragile mutations and feature shifts.
        """
        try:
            # Load data
            baseline_data = np.load(baseline_npz_path)
            mutated_data = np.load(mutated_npz_path)
            
            baseline_sessions = baseline_data['sessions']
            mutated_sessions = mutated_data['sessions']
            baseline_labels = baseline_data.get('labels', np.ones(len(baseline_sessions)))
            
            # Run predictions
            baseline_pred = self._batch_predict(baseline_sessions, batch_size)
            mutated_pred = self._batch_predict(mutated_sessions, batch_size)
            
            # Identify failures
            failures = (baseline_pred[:, 1] == 1) & (mutated_pred[:, 1] == 0)
            failure_indices = np.where(failures)[0]
            
            logger.info(f"Found {len(failure_indices)} failure samples")
            
            # Load mutation metadata
            fragile_mutations = defaultdict(int)
            if metadata_jsonl_path and Path(metadata_jsonl_path).exists():
                with open(metadata_jsonl_path, 'r') as f:
                    for line in f:
                        meta = json.loads(line)
                        if meta.get('original_label') == 1:  # Was C2
                            fragile_mutations[meta['mutation_type']] += 1
            
            # Feature shift analysis
            feature_shift = self._analyze_feature_shift(
                baseline_sessions[failure_indices],
                mutated_sessions[failure_indices]
            )
            
            analysis = {
                "total_failures": int(len(failure_indices)),
                "failure_rate": float(len(failure_indices) / len(baseline_sessions)),
                "fragile_mutations": dict(sorted(fragile_mutations.items(), key=lambda x: x[1], reverse=True)),
                "feature_shifts": feature_shift,
                "failed_samples_indices": failure_indices.tolist()[:100],  # First 100
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failure analysis failed: {e}")
            raise
    
    def _batch_predict(self, sessions: np.ndarray, batch_size: int) -> np.ndarray:
        """Run batch predictions, return (risk_score, is_suspicious, confidence)."""
        predictions = []
        
        for i in range(0, len(sessions), batch_size):
            batch = sessions[i:i+batch_size]
            session_dicts = [
                {
                    "duration": float(row[0]) if row.shape[0] > 0 else 0.0,
                    "bytes_in": float(row[1]) if row.shape[0] > 1 else 0.0,
                    "bytes_out": float(row[2]) if row.shape[0] > 2 else 0.0,
                    "packets_in": int(row[3]) if row.shape[0] > 3 else 0,
                    "packets_out": int(row[4]) if row.shape[0] > 4 else 0,
                    "protocol": "TCP",
                    "src_port": 0,
                    "dst_port": 0,
                    "timestamp": 0.0,
                }
                for row in batch
            ]
            
            batch_pred = self.detection_service.infer_batch(session_dicts)
            for risk_score, is_suspicious, confidence in batch_pred:
                predictions.append([risk_score, 1 if is_suspicious else 0, confidence])
        
        return np.array(predictions)
    
    def _compute_recall(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Compute recall (TP / (TP + FN))."""
        true_positives = np.sum((predictions[:, 1] == 1) & (labels == 1))
        positives = np.sum(labels == 1)
        return float(true_positives / max(positives, 1))
    
    def _analyze_feature_shift(
        self,
        baseline_samples: np.ndarray,
        mutated_samples: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Analyze feature changes between baseline and mutated."""
        feature_names = [
            "duration", "bytes_in", "bytes_out", "packets_in",
            "packets_out", "protocol", "src_port", "dst_port",
            "timestamp", "reserved"
        ]
        
        shifts = {}
        for i, name in enumerate(feature_names):
            if i < baseline_samples.shape[1]:
                baseline_vals = baseline_samples[:, i]
                mutated_vals = mutated_samples[:, i]
                
                shifts[name] = {
                    "baseline_mean": float(np.mean(baseline_vals)),
                    "mutated_mean": float(np.mean(mutated_vals)),
                    "shift": float(np.mean(mutated_vals) - np.mean(baseline_vals)),
                    "shift_pct": float((np.mean(mutated_vals) - np.mean(baseline_vals)) / max(abs(np.mean(baseline_vals)), 1e-6) * 100),
                }
        
        return shifts
