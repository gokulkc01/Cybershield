"""
CRITICAL FOUNDATION: Behavioral Validity Constraints

Prevents Red-Agent from generating impossible, unrealistic, or adversarial-artifact mutations.

This module enforces hard bounds on behavioral transformations to ensure:
- Timing remains physically plausible
- Flow statistics remain coherent
- Protocol structures remain valid
- Persistence behavior remains operationally sound
- Feature distributions remain realistic

DO NOT disable or weaken these constraints.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class BehavioralValidityConstraints:
    """Enforce domain-specific realism constraints on behavioral mutations."""

    # ────────────────────────────────────────────────────────────────
    # TIMING VALIDITY BOUNDS
    # ────────────────────────────────────────────────────────────────

    # Session duration constraints (seconds)
    MIN_SESSION_DURATION = 0.1  # Absolute minimum
    MAX_SESSION_DURATION = 86400 * 30  # 30 days max realistic C2 session

    # Beacon interval constraints (seconds)
    MIN_BEACON_INTERVAL = 1.0  # Sub-second beacons are unrealistic
    MAX_BEACON_INTERVAL = 86400  # 24-hour intervals are about max realistic

    # Jitter application constraints (as fraction of base interval)
    MAX_JITTER_FRACTION = 0.5  # Don't jitter by more than 50% of base
    MIN_JITTER_FRACTION = 0.01  # Don't jitter by less than 1% (meaningless)

    # Sleep/inactivity constraints (seconds)
    MAX_SLEEP_DURATION = 86400 * 7  # 7-day sleeps max
    MIN_RECONNECT_DELAY = 0.1
    MAX_RECONNECT_DELAY = 3600  # 1-hour reconnect max without breaking persistence

    # ────────────────────────────────────────────────────────────────
    # FLOW VALIDITY BOUNDS
    # ────────────────────────────────────────────────────────────────

    # Packet count constraints (per session)
    MIN_PACKETS = 2  # At least SYN/ACK equivalent
    MAX_PACKETS = 1_000_000  # Unrealistic to have >1M packets in single flow

    # Byte volume constraints (per session)
    MIN_BYTES = 50  # Minimum realistic TCP/IP overhead
    MAX_BYTES = 10_000_000_000  # 10GB max (unrealistic for C2 command/response)

    # Byte ratio constraints
    MIN_BYTE_RATIO = 0.01  # At least 1% asymmetry (else symmetric=suspicious)
    MAX_BYTE_RATIO = 100.0  # At most 100:1 ratio (else nonsensical)

    # Packet ratio constraints
    MIN_PACKET_RATIO = 0.01
    MAX_PACKET_RATIO = 100.0

    # Bytes-per-packet constraints (average)
    MIN_BYTES_PER_PKT = 20  # Absolute minimum (mostly TCP/IP overhead)
    MAX_BYTES_PER_PKT = 65535  # MTU-sized packets

    # ────────────────────────────────────────────────────────────────
    # PROTOCOL VALIDITY BOUNDS
    # ────────────────────────────────────────────────────────────────

    # Port ranges (valid)
    MIN_PORT = 1
    MAX_PORT = 65535
    EPHEMERAL_PORT_MIN = 49152
    EPHEMERAL_PORT_MAX = 65535

    # Suspicious vs. benign port thresholds
    HIGH_PRIVILEGE_PORTS = set(range(1, 1024))  # Ports < 1024
    WELL_KNOWN_C2_PORTS = {25, 53, 80, 110, 143, 443, 587, 993, 995, 3389, 8443}

    # TLS/HTTPS constraints
    MIN_TLS_RECORD_SIZE = 29  # Minimum TLS record
    MAX_TLS_RECORD_SIZE = 16385  # Standard max

    # ────────────────────────────────────────────────────────────────
    # PERSISTENCE VALIDITY BOUNDS
    # ────────────────────────────────────────────────────────────────

    # Inter-arrival time (IAT) constraints (seconds)
    MIN_IAT = 0.001  # Millisecond granularity minimum
    MAX_IAT = 86400  # 24 hours max between packets in a session

    # Callback frequency constraints
    MIN_CALLBACKS_PER_HOUR = 0.1  # At least one callback every 10 hours
    MAX_CALLBACKS_PER_HOUR = 1000  # Absurd upper bound

    # Long-sleep viability constraint
    MAX_CONSECUTIVE_SLEEP = 86400 * 30  # 30 days max offline

    # ────────────────────────────────────────────────────────────────
    # STATISTICAL VALIDITY BOUNDS
    # ────────────────────────────────────────────────────────────────

    # Distribution constraints (prevents feature-space collapse)
    MIN_FEATURE_STD = 0.001  # Minimum standard deviation (prevent collapse to constant)
    MAX_FEATURE_SKEW = 3.0  # Maximum reasonable skewness
    MAX_FEATURE_KURTOSIS = 5.0  # Maximum reasonable kurtosis

    # Coefficient of variation constraints
    MIN_CV = 0.01  # Minimum CV (0.01 = 1%)
    MAX_CV = 10.0  # Maximum CV (1000%)

    def __init__(self):
        """Initialize constraint validator."""
        self.violations = []

    def validate_timing(
        self,
        original: np.ndarray,
        mutated: np.ndarray,
        feature_names: List[str],
    ) -> Tuple[bool, List[str]]:
        """
        Validate timing features remain plausible.

        Args:
            original: Original session tensor (20, D)
            mutated: Mutated session tensor (20, D)
            feature_names: List of feature names

        Returns:
            (is_valid, list_of_violations)
        """
        violations = []

        # Find duration index
        try:
            duration_idx = feature_names.index("duration")
        except ValueError:
            return True, []  # Skip if no duration

        duration_values = mutated[:, duration_idx]
        duration_values = duration_values[duration_values > 0]  # Filter padding

        if len(duration_values) == 0:
            return True, []

        min_dur = np.min(duration_values)
        max_dur = np.max(duration_values)

        if min_dur < self.MIN_SESSION_DURATION:
            violations.append(
                f"Timing: Session duration {min_dur:.3f}s < MIN {self.MIN_SESSION_DURATION}s"
            )

        if max_dur > self.MAX_SESSION_DURATION:
            violations.append(
                f"Timing: Session duration {max_dur:.1f}s > MAX {self.MAX_SESSION_DURATION}s"
            )

        # Validate IAT if available
        try:
            iat_idx = feature_names.index("iat")
            iat_values = mutated[:, iat_idx]
            iat_values = iat_values[iat_values > 0]

            if len(iat_values) > 0:
                if np.min(iat_values) < self.MIN_IAT:
                    violations.append(f"Timing: IAT {np.min(iat_values):.6f}s < MIN {self.MIN_IAT}s")
                if np.max(iat_values) > self.MAX_IAT:
                    violations.append(f"Timing: IAT {np.max(iat_values):.1f}s > MAX {self.MAX_IAT}s")
        except ValueError:
            pass  # No IAT feature

        return len(violations) == 0, violations

    def validate_flow(
        self,
        original: np.ndarray,
        mutated: np.ndarray,
        feature_names: List[str],
    ) -> Tuple[bool, List[str]]:
        """
        Validate flow statistics remain coherent.

        Args:
            original: Original session tensor (20, D)
            mutated: Mutated session tensor (20, D)
            feature_names: List of feature names

        Returns:
            (is_valid, list_of_violations)
        """
        violations = []

        feature_idx = {name: i for i, name in enumerate(feature_names)}

        # Validate packet counts
        if "orig_pkts" in feature_idx and "resp_pkts" in feature_idx:
            orig_pkts = mutated[:, feature_idx["orig_pkts"]]
            resp_pkts = mutated[:, feature_idx["resp_pkts"]]

            orig_pkts_total = np.sum(orig_pkts[orig_pkts > 0])
            resp_pkts_total = np.sum(resp_pkts[resp_pkts > 0])

            if orig_pkts_total < self.MIN_PACKETS or resp_pkts_total < self.MIN_PACKETS:
                violations.append(
                    f"Flow: Total packets ({orig_pkts_total}, {resp_pkts_total}) < MIN {self.MIN_PACKETS}"
                )

            if orig_pkts_total > self.MAX_PACKETS or resp_pkts_total > self.MAX_PACKETS:
                violations.append(
                    f"Flow: Total packets ({orig_pkts_total}, {resp_pkts_total}) > MAX {self.MAX_PACKETS}"
                )

            # Validate packet ratio
            if resp_pkts_total > 0:
                pkt_ratio = orig_pkts_total / resp_pkts_total
                if pkt_ratio < self.MIN_PACKET_RATIO or pkt_ratio > self.MAX_PACKET_RATIO:
                    violations.append(
                        f"Flow: Packet ratio {pkt_ratio:.2f} outside [{self.MIN_PACKET_RATIO}, {self.MAX_PACKET_RATIO}]"
                    )

        # Validate byte counts
        if "orig_bytes" in feature_idx and "resp_bytes" in feature_idx:
            orig_bytes = mutated[:, feature_idx["orig_bytes"]]
            resp_bytes = mutated[:, feature_idx["resp_bytes"]]

            orig_bytes_total = np.sum(orig_bytes[orig_bytes > 0])
            resp_bytes_total = np.sum(resp_bytes[resp_bytes > 0])

            if orig_bytes_total < self.MIN_BYTES or resp_bytes_total < self.MIN_BYTES:
                violations.append(
                    f"Flow: Total bytes ({orig_bytes_total}, {resp_bytes_total}) < MIN {self.MIN_BYTES}"
                )

            if orig_bytes_total > self.MAX_BYTES or resp_bytes_total > self.MAX_BYTES:
                violations.append(
                    f"Flow: Total bytes ({orig_bytes_total}, {resp_bytes_total}) > MAX {self.MAX_BYTES}"
                )

            # Validate byte ratio
            if resp_bytes_total > 0:
                byte_ratio = orig_bytes_total / resp_bytes_total
                if byte_ratio < self.MIN_BYTE_RATIO or byte_ratio > self.MAX_BYTE_RATIO:
                    violations.append(
                        f"Flow: Byte ratio {byte_ratio:.2f} outside [{self.MIN_BYTE_RATIO}, {self.MAX_BYTE_RATIO}]"
                    )

        return len(violations) == 0, violations

    def validate_port_validity(
        self,
        src_port: int,
        dst_port: int,
    ) -> Tuple[bool, List[str]]:
        """
        Validate port numbers are realistic.

        Args:
            src_port: Source port
            dst_port: Destination port

        Returns:
            (is_valid, list_of_violations)
        """
        violations = []

        if not (self.MIN_PORT <= src_port <= self.MAX_PORT):
            violations.append(f"Port: SRC {src_port} outside [{self.MIN_PORT}, {self.MAX_PORT}]")

        if not (self.MIN_PORT <= dst_port <= self.MAX_PORT):
            violations.append(f"Port: DST {dst_port} outside [{self.MIN_PORT}, {self.MAX_PORT}]")

        return len(violations) == 0, violations

    def validate_statistical_integrity(
        self,
        mutated: np.ndarray,
        original: np.ndarray,
    ) -> Tuple[bool, List[str]]:
        """
        Validate statistical integrity hasn't collapsed.

        Prevents feature-space adversarial artifacts.

        Args:
            mutated: Mutated session tensor
            original: Original session tensor

        Returns:
            (is_valid, list_of_violations)
        """
        violations = []

        # Ensure features aren't collapsing to constants
        for i in range(mutated.shape[1]):
            mut_col = mutated[:, i]
            mut_col = mut_col[mut_col != 0]  # Exclude padding

            if len(mut_col) > 1:
                std_val = np.std(mut_col)
                if std_val < self.MIN_FEATURE_STD:
                    violations.append(f"Statistical: Feature {i} std {std_val:.6f} < MIN {self.MIN_FEATURE_STD}")

                # Check if variance collapsed compared to original
                orig_col = original[:, i]
                orig_col = orig_col[orig_col != 0]
                if len(orig_col) > 1:
                    orig_std = np.std(orig_col)
                    if orig_std > 0 and std_val / orig_std < 0.1:
                        violations.append(
                            f"Statistical: Feature {i} variance collapsed {std_val / orig_std:.2%}"
                        )

        return len(violations) == 0, violations

    def check_all_constraints(
        self,
        original: np.ndarray,
        mutated: np.ndarray,
        feature_names: List[str],
        src_port: Optional[int] = None,
        dst_port: Optional[int] = None,
    ) -> Tuple[bool, List[str]]:
        """
        Run all validity checks.

        Args:
            original: Original session tensor
            mutated: Mutated session tensor
            feature_names: List of feature names
            src_port: Source port (optional)
            dst_port: Destination port (optional)

        Returns:
            (is_valid, list_of_all_violations)
        """
        all_violations = []

        # Timing
        _, timing_viols = self.validate_timing(original, mutated, feature_names)
        all_violations.extend(timing_viols)

        # Flow
        _, flow_viols = self.validate_flow(original, mutated, feature_names)
        all_violations.extend(flow_viols)

        # Ports
        if src_port is not None and dst_port is not None:
            _, port_viols = self.validate_port_validity(src_port, dst_port)
            all_violations.extend(port_viols)

        # Statistical integrity
        _, stat_viols = self.validate_statistical_integrity(mutated, original)
        all_violations.extend(stat_viols)

        return len(all_violations) == 0, all_violations


# Singleton instance for convenience
constraints = BehavioralValidityConstraints()
