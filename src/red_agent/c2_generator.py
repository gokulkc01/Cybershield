"""C2 traffic generation infrastructure and capture scripts.

This module provides documentation, configuration templates, and utility
scripts for generating diverse C2 traffic captures using modern frameworks.

IMPORTANT: Actual C2 generation requires isolated lab environments.
This module provides:
  1. Capture procedure documentation
  2. Malleable profile configurations
  3. Capture automation scripts
  4. Benign background traffic mixing recipes

Supported C2 Frameworks:
  - Sliver (primary)
  - Havoc (primary)
  - Metasploit / Meterpreter (baseline)
  - Cobalt Strike (if available)
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

# ──────────────────────────────────────────────────────────────────────
# Capture Configuration
# ──────────────────────────────────────────────────────────────────────

@dataclass
class C2CaptureConfig:
    """Configuration for a single C2 traffic capture session."""

    # Framework identification
    framework: str                      # sliver, havoc, metasploit, etc.
    profile_name: str                   # Descriptive profile name
    profile_description: str = ""

    # Timing parameters (seconds)
    beacon_sleep_min: float = 5.0       # Minimum sleep between beacons
    beacon_sleep_max: float = 60.0      # Maximum sleep between beacons
    beacon_jitter_pct: float = 0.30     # Jitter percentage (0.0–1.0)

    # Protocol configuration
    protocol: str = "https"             # http, https, dns, tcp
    port: int = 443
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

    # Capture parameters
    capture_duration_hours: float = 24.0   # Total capture duration
    zeek_enabled: bool = True               # Run Zeek on the capture
    pcap_enabled: bool = True               # Also save raw PCAP

    # Activity patterns
    activity_pattern: str = "interactive"  # interactive, automated, mixed, idle
    commands_per_session: int = 5           # Commands to execute per activity burst
    idle_periods_hours: List[float] = field(
        default_factory=lambda: [0.5, 2.0, 6.0]  # Idle periods to simulate
    )

    # Benign background
    benign_background: bool = True          # Mix with benign background traffic
    benign_sources: List[str] = field(
        default_factory=lambda: [
            "web_browsing",
            "saas_traffic",
            "update_checks",
        ]
    )

    def to_dict(self) -> Dict:
        return asdict(self)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[INFO] Config saved to {path}")

    @classmethod
    def load(cls, path: str) -> "C2CaptureConfig":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


# ──────────────────────────────────────────────────────────────────────
# Pre-built capture profiles
# ──────────────────────────────────────────────────────────────────────

def get_sliver_profiles() -> List[C2CaptureConfig]:
    """Generate diverse Sliver C2 capture configurations."""
    return [
        # Fast beaconing (easy to detect)
        C2CaptureConfig(
            framework="sliver",
            profile_name="sliver_fast_beacon",
            profile_description="Fast beaconing (5–15s) over HTTPS — easy baseline",
            beacon_sleep_min=5.0,
            beacon_sleep_max=15.0,
            beacon_jitter_pct=0.10,
            protocol="https",
            capture_duration_hours=24.0,
            activity_pattern="interactive",
        ),
        # Medium beaconing with jitter (moderate difficulty)
        C2CaptureConfig(
            framework="sliver",
            profile_name="sliver_medium_jitter",
            profile_description="Medium beaconing (30–120s) with high jitter — moderate",
            beacon_sleep_min=30.0,
            beacon_sleep_max=120.0,
            beacon_jitter_pct=0.50,
            protocol="https",
            capture_duration_hours=48.0,
            activity_pattern="mixed",
        ),
        # Low-and-slow (hard to detect)
        C2CaptureConfig(
            framework="sliver",
            profile_name="sliver_low_slow",
            profile_description="Low-and-slow (5–30 min) with high jitter — difficult",
            beacon_sleep_min=300.0,
            beacon_sleep_max=1800.0,
            beacon_jitter_pct=0.70,
            protocol="https",
            capture_duration_hours=72.0,
            activity_pattern="idle",
            commands_per_session=2,
        ),
        # DNS-based C2 (different protocol)
        C2CaptureConfig(
            framework="sliver",
            profile_name="sliver_dns_tunnel",
            profile_description="DNS tunneling C2 — different protocol signature",
            beacon_sleep_min=60.0,
            beacon_sleep_max=300.0,
            beacon_jitter_pct=0.40,
            protocol="dns",
            port=53,
            capture_duration_hours=48.0,
            activity_pattern="automated",
        ),
        # Event-driven (mimics user activity)
        C2CaptureConfig(
            framework="sliver",
            profile_name="sliver_event_driven",
            profile_description="Event-driven beaconing — mimics user activity patterns",
            beacon_sleep_min=60.0,
            beacon_sleep_max=600.0,
            beacon_jitter_pct=0.80,
            protocol="https",
            capture_duration_hours=72.0,
            activity_pattern="mixed",
            idle_periods_hours=[1.0, 4.0, 8.0, 12.0],
        ),
    ]


def get_havoc_profiles() -> List[C2CaptureConfig]:
    """Generate diverse Havoc C2 capture configurations."""
    return [
        C2CaptureConfig(
            framework="havoc",
            profile_name="havoc_standard",
            profile_description="Standard Havoc HTTPS beaconing",
            beacon_sleep_min=10.0,
            beacon_sleep_max=60.0,
            beacon_jitter_pct=0.30,
            protocol="https",
            capture_duration_hours=24.0,
        ),
        C2CaptureConfig(
            framework="havoc",
            profile_name="havoc_evasive",
            profile_description="Evasive Havoc with long sleep and high jitter",
            beacon_sleep_min=120.0,
            beacon_sleep_max=900.0,
            beacon_jitter_pct=0.60,
            protocol="https",
            capture_duration_hours=72.0,
            activity_pattern="idle",
        ),
        C2CaptureConfig(
            framework="havoc",
            profile_name="havoc_bursty",
            profile_description="Bursty activity followed by long idle periods",
            beacon_sleep_min=5.0,
            beacon_sleep_max=30.0,
            beacon_jitter_pct=0.20,
            protocol="https",
            capture_duration_hours=48.0,
            activity_pattern="mixed",
            idle_periods_hours=[0.5, 2.0, 6.0, 12.0],
            commands_per_session=10,
        ),
    ]


def get_all_profiles() -> List[C2CaptureConfig]:
    """Return all pre-built capture profiles."""
    return get_sliver_profiles() + get_havoc_profiles()


def save_all_profiles(output_dir: str = "data/capture_configs") -> None:
    """Save all pre-built profiles as JSON files."""
    profiles = get_all_profiles()
    os.makedirs(output_dir, exist_ok=True)

    for profile in profiles:
        path = os.path.join(output_dir, f"{profile.profile_name}.json")
        profile.save(path)

    print(f"[INFO] Saved {len(profiles)} capture profiles to {output_dir}/")


# ──────────────────────────────────────────────────────────────────────
# Capture procedure documentation
# ──────────────────────────────────────────────────────────────────────

CAPTURE_PROCEDURE = """
# C2 Traffic Capture Procedure

## Environment Setup
1. Isolated lab network (NO real enterprise network exposure)
2. Victim VM (Windows 10/11) with realistic software installed
3. C2 server VM (Linux)
4. Zeek sensor on network tap/span port
5. Background traffic generator VM

## Step-by-Step

### 1. Start Zeek Capture
```bash
zeek -i <interface> -C local.zeek
```

### 2. Start Background Traffic (if enabled)
```bash
# Run benign traffic generator for realistic background noise
python scripts/benign_traffic_gen.py --duration 72h --profile enterprise
```

### 3. Deploy C2 Implant
- Generate implant with the specified profile configuration
- Transfer and execute on victim VM
- Verify C2 callback

### 4. Execute Activity Pattern
- **interactive**: Manual command execution at irregular intervals
- **automated**: Scripted command sequences at beacon intervals
- **mixed**: Combination of both + extended idle periods
- **idle**: Minimal activity, mostly just beacon keepalives

### 5. Stop Capture
- Wait for full capture_duration_hours
- Stop Zeek
- Collect conn.log, ssl.log, dns.log

### 6. Process Capture
```bash
python -m src.data_loader.uwf_processor \\
    --data_dir data/raw/lab_captures/<capture_name> \\
    --output_dir data/processed/lab/<capture_name>
```

## Critical Requirements
- Each capture MUST have at least 24 hours of data
- Background benign traffic is MANDATORY for realism
- Label files must clearly mark C2 vs benign flows
- Multiple jitter profiles per framework
"""


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="C2 Capture Configuration Generator")
    parser.add_argument(
        "--action", choices=["save_profiles", "show_procedure", "list"],
        default="list",
    )
    parser.add_argument("--output_dir", default="data/capture_configs")
    args = parser.parse_args()

    if args.action == "save_profiles":
        save_all_profiles(args.output_dir)
    elif args.action == "show_procedure":
        print(CAPTURE_PROCEDURE)
    else:
        profiles = get_all_profiles()
        print(f"\nAvailable C2 Capture Profiles ({len(profiles)}):")
        print("-" * 60)
        for p in profiles:
            sleep_range = f"{p.beacon_sleep_min:.0f}–{p.beacon_sleep_max:.0f}s"
            print(
                f"  {p.profile_name:<30s} | {p.framework:<12s} | "
                f"Sleep: {sleep_range:<12s} | Jitter: {p.beacon_jitter_pct:.0%} | "
                f"{p.capture_duration_hours:.0f}h"
            )
