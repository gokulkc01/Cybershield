"""
Realism Discriminator: Behavioral Authenticity Validator

Prevents PPO agent from learning unrealistic feature-space exploits.
Evaluates: "Does this mutated sample still resemble plausible operational C2 behavior?"

This discriminator is the PRIMARY SAFEGUARD against reward hacking.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class RealismDiscriminator:
    """
    Discriminator for behavioral realism.
    
    Answers: Is this mutated sample statistically and operationally plausible?
    """

    def __init__(self, training_stats: Optional[Dict[str, np.ndarray]] = None):
        """
        Initialize discriminator.

        Args:
            training_stats: Pre-computed statistics from training data
                Format: {'orig_bytes': (mean, std), ...}
        """
        self.training_stats = training_stats or self._default_c2_stats()

    @staticmethod
    def _default_c2_stats() -> Dict[str, Tuple[float, float]]:
        """
        Default C2 behavioral statistics (derived from CTU-13 + MCFP).
        
        Format: feature_name -> (mean, std)
        """
        return {
            "duration": (45.0, 120.0),  # Mean 45 seconds, std 120 seconds
            "orig_bytes": (5000.0, 15000.0),  # Mean 5KB, std 15KB
            "resp_bytes": (3000.0, 10000.0),  # Mean 3KB, std 10KB
            "orig_pkts": (50.0, 100.0),  # Mean 50 packets
            "resp_pkts": (35.0, 80.0),  # Mean 35 packets
            "src_port": (40000.0, 8000.0),  # Ephemeral range
            "dst_port": (443.0, 1500.0),  # Common destinations (443, 80, 8443 etc)
            "iat": (0.5, 1.5),  # Inter-arrival time (seconds)
            "bytes_per_pkt": (150.0, 200.0),  # Average payload per packet
            "is_outbound": (0.6, 0.2),  # Slightly outbound-biased (C2 sends commands)
        }

    def score_realism(
        self,
        mutated_session: np.ndarray,
        original_session: np.ndarray,
        feature_names: List[str],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Score overall realism of mutated session [0, 1].
        
        Higher = more realistic.

        Args:
            mutated_session: Mutated tensor (20, D)
            original_session: Original tensor (20, D)
            feature_names: Feature names

        Returns:
            (realism_score: float [0, 1], component_scores: dict)
        """
        component_scores = {}

        # 1. Statistical plausibility
        component_scores["statistical"] = self._score_statistical_plausibility(
            mutated_session, feature_names
        )

        # 2. Distribution shift (how much did it deviate from original)
        component_scores["distribution_shift"] = self._score_distribution_shift(
            mutated_session, original_session, feature_names
        )

        # 3. Behavioral coherence
        component_scores["coherence"] = self._score_behavioral_coherence(
            mutated_session, feature_names
        )

        # 4. Feature correlation preservation
        component_scores["correlation"] = self._score_correlation_preservation(
            mutated_session, original_session
        )

        # 5. Protocol realism
        component_scores["protocol"] = self._score_protocol_realism(
            mutated_session, feature_names
        )

        # Weighted average (equal weight for now)
        weights = {
            "statistical": 0.25,
            "distribution_shift": 0.25,
            "coherence": 0.2,
            "correlation": 0.15,
            "protocol": 0.15,
        }

        overall_score = sum(
            component_scores[k] * weights[k] for k in weights.keys()
        )

        return overall_score, component_scores

    def _score_statistical_plausibility(
        self,
        session: np.ndarray,
        feature_names: List[str],
    ) -> float:
        """
        Score how plausible features are compared to C2 training distribution.
        
        Uses z-score analysis.
        """
        if len(feature_names) == 0:
            return 1.0

        scores = []

        for i, fname in enumerate(feature_names):
            if fname not in self.training_stats:
                continue

            if i >= session.shape[1]:
                continue

            col = session[:, i]
            col = col[col > 0]  # Filter padding

            if len(col) == 0:
                continue

            mean_val = np.mean(col)
            train_mean, train_std = self.training_stats[fname]

            if train_std > 0:
                z_score = abs((mean_val - train_mean) / train_std)
                # Within 3 sigma = plausible (score 1.0)
                # Outside 5 sigma = implausible (score 0.0)
                plausibility = max(0.0, 1.0 - z_score / 5.0)
                scores.append(plausibility)

        return float(np.mean(scores)) if scores else 1.0

    def _score_distribution_shift(
        self,
        mutated: np.ndarray,
        original: np.ndarray,
        feature_names: List[str],
    ) -> float:
        """
        Score how much the distribution shifted.
        
        Penalizes extreme shifts (indicates unrealistic mutation).
        """
        if mutated.shape != original.shape:
            return 0.5

        shifts = []

        for i in range(min(len(feature_names), mutated.shape[1])):
            mut_col = mutated[:, i]
            orig_col = original[:, i]

            mut_col = mut_col[mut_col > 0]
            orig_col = orig_col[orig_col > 0]

            if len(mut_col) == 0 or len(orig_col) == 0:
                continue

            # Kolmogorov-Smirnov test for distribution shift
            try:
                ks_stat = stats.ks_2samp(mut_col, orig_col)[0]
                # KS stat 0 = identical, 1 = completely different
                # Penalize large shifts
                shift_score = 1.0 - min(ks_stat, 1.0)  # Cap at 1.0
                shifts.append(shift_score)
            except Exception:
                shifts.append(0.5)

        # If we shifted more than 30% on average, that's suspicious
        if shifts:
            avg_shift = np.mean(shifts)
            # penalize if shift is too extreme
            if avg_shift < 0.7:
                return avg_shift
            else:
                return max(0.0, 1.0 - (avg_shift - 0.7) / 0.3)

        return 0.5

    def _score_behavioral_coherence(
        self,
        session: np.ndarray,
        feature_names: List[str],
    ) -> float:
        """
        Score whether behavioral relationships make sense.
        
        Examples:
        - bytes_per_pkt should be orig_bytes / orig_pkts
        - is_outbound should be 0 or 1
        - packet counts should be positive
        """
        feature_idx = {name: i for i, name in enumerate(feature_names) if i < session.shape[1]}
        coherence_scores = []

        # If we have bytes and packets, check bytes_per_pkt
        if "orig_bytes" in feature_idx and "orig_pkts" in feature_idx:
            orig_bytes = session[:, feature_idx["orig_bytes"]]
            orig_pkts = session[:, feature_idx["orig_pkts"]]

            # Ignore padding (0 values)
            mask = (orig_bytes > 0) & (orig_pkts > 0)
            if np.any(mask):
                computed_bpp = orig_bytes[mask] / orig_pkts[mask]
                if "bytes_per_pkt" in feature_idx:
                    reported_bpp = session[mask, feature_idx["bytes_per_pkt"]]
                    # Should be close to computed
                    error = np.abs(computed_bpp - reported_bpp) / (computed_bpp + 1e-6)
                    coherence_scores.append(1.0 - np.mean(np.minimum(error, 1.0)))

        # is_outbound should be 0 or 1 (or close)
        if "is_outbound" in feature_idx:
            is_outbound = session[:, feature_idx["is_outbound"]]
            is_outbound = is_outbound[is_outbound > 0]
            if len(is_outbound) > 0:
                # Should be ~0 or ~1
                error = np.minimum(is_outbound, 1.0 - is_outbound)
                coherence_scores.append(1.0 - np.mean(error))

        # Packet counts should be positive integers
        if "orig_pkts" in feature_idx and "resp_pkts" in feature_idx:
            orig_pkts = session[:, feature_idx["orig_pkts"]]
            resp_pkts = session[:, feature_idx["resp_pkts"]]
            # Check for negative values
            if np.any(orig_pkts < 0) or np.any(resp_pkts < 0):
                coherence_scores.append(0.0)
            else:
                coherence_scores.append(1.0)

        return float(np.mean(coherence_scores)) if coherence_scores else 0.8

    def _score_correlation_preservation(
        self,
        mutated: np.ndarray,
        original: np.ndarray,
    ) -> float:
        """
        Score whether feature correlations are preserved.
        
        Real C2 has real feature dependencies.
        Unrealistic mutations break those dependencies.
        """
        # Compute correlation matrices
        try:
            # Use only non-padding rows
            mut_mask = np.any(mutated != 0, axis=1)
            orig_mask = np.any(original != 0, axis=1)

            if np.sum(mut_mask) < 5 or np.sum(orig_mask) < 5:
                return 0.8

            mut_corr = np.corrcoef(mutated[mut_mask].T)
            orig_corr = np.corrcoef(original[orig_mask].T)

            # Frobenius norm of difference (normalized)
            diff = np.sqrt(np.sum((mut_corr - orig_corr) ** 2))
            max_diff = np.sqrt(np.sum(orig_corr ** 2))

            if max_diff > 0:
                rel_diff = diff / max_diff
                # Penalize large correlation changes
                correlation_preservation = 1.0 - min(rel_diff, 1.0)
                return correlation_preservation
            else:
                return 0.8

        except Exception as e:
            logger.debug(f"Correlation scoring failed: {e}")
            return 0.8

    def _score_protocol_realism(
        self,
        session: np.ndarray,
        feature_names: List[str],
    ) -> float:
        """
        Score protocol-level realism.
        
        Examples:
        - TCP should have appropriate packet ratios
        - Ports should be in valid ranges
        - Duration should correlate with packets and bytes
        """
        feature_idx = {name: i for i, name in enumerate(feature_names) if i < session.shape[1]}
        protocol_scores = []

        # Duration should correlate with activity
        if "duration" in feature_idx and "orig_pkts" in feature_idx:
            duration = session[:, feature_idx["duration"]]
            orig_pkts = session[:, feature_idx["orig_pkts"]]

            mask = (duration > 0) & (orig_pkts > 0)
            if np.any(mask):
                # Approximate packets per second
                pps = orig_pkts[mask] / (duration[mask] + 1e-6)
                # Realistic range: 0.1 to 1000 packets per second
                pps_realistic = np.mean((pps >= 0.1) & (pps <= 1000))
                protocol_scores.append(pps_realistic)

        # Src/dst port realism
        if "src_port" in feature_idx and "dst_port" in feature_idx:
            src_ports = session[:, feature_idx["src_port"]]
            dst_ports = session[:, feature_idx["dst_port"]]

            src_ports = src_ports[src_ports > 0]
            dst_ports = dst_ports[dst_ports > 0]

            if len(src_ports) > 0 and len(dst_ports) > 0:
                src_valid = np.mean((src_ports >= 1024) & (src_ports <= 65535))  # Usually ephemeral
                dst_valid = np.mean((dst_ports >= 1) & (dst_ports <= 65535))
                protocol_scores.append((src_valid + dst_valid) / 2.0)

        return float(np.mean(protocol_scores)) if protocol_scores else 0.8

    def is_realistic(
        self,
        mutated_session: np.ndarray,
        original_session: np.ndarray,
        feature_names: List[str],
        threshold: float = 0.6,
    ) -> bool:
        """
        Simple boolean: is this mutation realistic?

        Args:
            mutated_session: Mutated tensor
            original_session: Original tensor
            feature_names: Feature names
            threshold: Realism score threshold [0, 1]

        Returns:
            True if realism_score >= threshold
        """
        score, _ = self.score_realism(mutated_session, original_session, feature_names)
        return score >= threshold


# Singleton instance
discriminator = RealismDiscriminator()
