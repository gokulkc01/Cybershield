# Phase 1 Implementation Guide: Enhanced Features & Cross-Dataset Validation

**Timeline**: 6 weeks (Starting Week of May 13, 2026)  
**Goal**: Add 30 new features (TLS, DNS, temporal) + validate cross-dataset generalization

---

## Week 1: Feature Extraction Foundation

### Task 1.1: TLS Feature Extractor (3 days)

**File to create**: `src/features/tls_features.py`

```python
"""TLS behavioral feature extraction from Zeek ssl.log"""

import numpy as np
import pandas as pd
from typing import Dict, List
import json

class TLSFeatureExtractor:
    """Extract TLS-based behavioral features from Zeek SSL logs."""
    
    TLS_FEATURES = [
        "tls_version",              # Most common TLS version
        "tls_cipher_count",         # Unique ciphers used
        "tls_extension_count",      # Number of TLS extensions
        "tls_supported_groups",     # Supported elliptic curves
        "tls_alpn_count",           # ALPN protocols offered
        "tls_session_reuse_rate",   # % of sessions reused
        "tls_cert_reuse_rate",      # % of certs reused
        "tls_handshake_size",       # Average handshake size (bytes)
        "tls_version_diversity",    # How many different TLS versions
        "tls_cipher_diversity",     # How many different ciphers
        "tls_cert_validity_days",   # Days until cert expires
        "tls_ja3_hash",             # JA3 fingerprint (categorical)
    ]
    
    def __init__(self):
        self.cache = {}
    
    def extract_from_zeek_ssl_log(self, ssl_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Extract TLS features from Zeek ssl.log
        
        Expected columns:
        - ts (timestamp)
        - id.orig_h (source IP)
        - id.resp_h (destination IP)
        - id.resp_p (destination port)
        - version (TLS version)
        - cipher (cipher suite)
        - server_name (SNI)
        - ja (JA3 fingerprint)
        - ja3s (JA3S fingerprint)
        - validation_status
        - cert_chain_fuids
        """
        
        features = {}
        
        # Extract TLS versions
        features["tls_version"] = ssl_df["version"].mode()[0] if len(ssl_df) > 0 else 0
        features["tls_version_diversity"] = ssl_df["version"].nunique()
        
        # Extract cipher information
        features["tls_cipher_count"] = ssl_df["cipher"].nunique()
        features["tls_cipher_diversity"] = ssl_df["cipher"].nunique() / max(len(ssl_df), 1)
        
        # Extract extension information (from JA3 typically)
        features["tls_extension_count"] = self._extract_extension_count(ssl_df)
        
        # Session reuse
        features["tls_session_reuse_rate"] = self._compute_session_reuse_rate(ssl_df)
        
        # Certificate reuse
        features["tls_cert_reuse_rate"] = self._compute_cert_reuse_rate(ssl_df)
        
        # Handshake size
        features["tls_handshake_size"] = ssl_df.get("ssl_record_version", pd.Series([512])).mean()
        
        # ALPN
        features["tls_alpn_count"] = ssl_df["next_protocol"].nunique() if "next_protocol" in ssl_df else 0
        
        # Supported groups (from handshake extensions)
        features["tls_supported_groups"] = self._extract_supported_groups(ssl_df)
        
        # Certificate validity
        features["tls_cert_validity_days"] = self._compute_cert_validity_days(ssl_df)
        
        return features
    
    def _extract_extension_count(self, ssl_df: pd.DataFrame) -> int:
        """Extract number of TLS extensions from ja/ja3 fingerprints"""
        if len(ssl_df) == 0:
            return 0
        # Simplified: count unique JA3 fingerprints as proxy for extension variety
        return ssl_df["ja"].nunique() if "ja" in ssl_df else 0
    
    def _compute_session_reuse_rate(self, ssl_df: pd.DataFrame) -> float:
        """Compute fraction of sessions that reuse previous sessions"""
        if len(ssl_df) < 2:
            return 0.0
        # Sessions with session_id matching previous = session reuse
        session_ids = ssl_df.get("session_id", pd.Series([]))
        if len(session_ids) == 0:
            return 0.0
        reused = len(session_ids) - len(session_ids.unique())
        return reused / len(session_ids)
    
    def _compute_cert_reuse_rate(self, ssl_df: pd.DataFrame) -> float:
        """Compute fraction of sessions reusing same certificate"""
        if len(ssl_df) < 2:
            return 0.0
        certs = ssl_df.get("cert_chain_fuids", pd.Series([]))
        if len(certs) == 0:
            return 0.0
        reused = len(certs) - len(certs.unique())
        return reused / len(certs)
    
    def _extract_supported_groups(self, ssl_df: pd.DataFrame) -> int:
        """Extract supported groups (elliptic curves) from handshake"""
        # Placeholder: would require parsing TLS handshake details
        # From Zeek: supported_groups field
        if "supported_groups" in ssl_df:
            return ssl_df["supported_groups"].nunique()
        return 0
    
    def _compute_cert_validity_days(self, ssl_df: pd.DataFrame) -> float:
        """Compute average days until certificate expiry"""
        if "cert_not_valid_after" not in ssl_df or len(ssl_df) == 0:
            return 365.0  # Default
        
        import datetime
        now = datetime.datetime.now()
        expires = pd.to_datetime(ssl_df["cert_not_valid_after"], errors="coerce")
        days_until_expiry = (expires - now).dt.days
        return days_until_expiry.mean()
```

**Testing**: Create `tests/test_tls_features.py`

```python
import pytest
import pandas as pd
from src.features.tls_features import TLSFeatureExtractor

def test_tls_feature_extraction():
    """Test TLS feature extraction"""
    extractor = TLSFeatureExtractor()
    
    # Create mock SSL log
    ssl_df = pd.DataFrame({
        "version": ["TLSv1.2", "TLSv1.2", "TLSv1.3"],
        "cipher": ["ECDHE-RSA-AES128", "ECDHE-RSA-AES128", "TLS_AES_256_GCM_SHA384"],
        "ja": ["abc123", "abc123", "def456"],
    })
    
    features = extractor.extract_from_zeek_ssl_log(ssl_df)
    
    assert "tls_version_diversity" in features
    assert features["tls_version_diversity"] == 2  # TLSv1.2 and TLSv1.3
    assert features["tls_cipher_count"] == 3
```

---

### Task 1.2: DNS Feature Extractor (3 days)

**File to create**: `src/features/dns_features.py`

```python
"""DNS behavioral feature extraction from Zeek dns.log"""

import numpy as np
import pandas as pd
from typing import Dict, List
import re

class DNSFeatureExtractor:
    """Extract DNS-based behavioral features from Zeek DNS logs."""
    
    DNS_FEATURES = [
        "dns_query_count",           # Total DNS queries
        "dns_unique_domains",        # Unique domains queried
        "dns_nxdomain_rate",         # % of failed lookups
        "dns_dga_score",             # DGA probability (0-1)
        "dns_domain_age_avg",        # Average domain age (days)
        "dns_ttl_consistency",       # Std dev of TTLs
        "dns_query_type_entropy",    # Entropy of query types
        "dns_lookup_frequency",      # Queries per session minute
        "dns_rapid_fire_rate",       # % of queries in rapid bursts
        "dns_subdomain_depth_avg",   # Average subdomain levels
    ]
    
    def extract_from_zeek_dns_log(self, dns_df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract DNS features from Zeek dns.log
        
        Expected columns:
        - ts (timestamp)
        - query (domain name)
        - qtype (query type: A, AAAA, MX, etc.)
        - rcode (response code: 0=success, 3=NXDOMAIN)
        - ttls (time-to-live list)
        - answers (response IPs)
        """
        
        features = {}
        
        if len(dns_df) == 0:
            # Default values for empty log
            return {
                "dns_query_count": 0,
                "dns_unique_domains": 0,
                "dns_nxdomain_rate": 0.0,
                "dns_dga_score": 0.0,
                "dns_domain_age_avg": 0.0,
                "dns_ttl_consistency": 0.0,
                "dns_query_type_entropy": 0.0,
                "dns_lookup_frequency": 0.0,
                "dns_rapid_fire_rate": 0.0,
                "dns_subdomain_depth_avg": 0.0,
            }
        
        # Query count
        features["dns_query_count"] = len(dns_df)
        
        # Unique domains
        domains = dns_df["query"].dropna().unique()
        features["dns_unique_domains"] = len(domains)
        
        # NXDOMAIN rate (rcode == 3 = NXDOMAIN)
        nxdomain_count = (dns_df.get("rcode", pd.Series([])) == 3).sum()
        features["dns_nxdomain_rate"] = nxdomain_count / max(len(dns_df), 1)
        
        # DGA score (heuristic)
        features["dns_dga_score"] = self._compute_dga_score(domains)
        
        # Domain age (would need WHOIS data, simplified here)
        features["dns_domain_age_avg"] = self._estimate_domain_age(domains)
        
        # TTL consistency
        features["dns_ttl_consistency"] = self._compute_ttl_consistency(dns_df)
        
        # Query type entropy
        features["dns_query_type_entropy"] = self._compute_query_type_entropy(dns_df)
        
        # Lookup frequency (queries per minute)
        time_span = (dns_df["ts"].max() - dns_df["ts"].min()).total_seconds() / 60
        features["dns_lookup_frequency"] = len(dns_df) / max(time_span, 1.0)
        
        # Rapid fire rate (queries within 1 second)
        features["dns_rapid_fire_rate"] = self._compute_rapid_fire_rate(dns_df)
        
        # Subdomain depth
        features["dns_subdomain_depth_avg"] = self._compute_subdomain_depth(domains)
        
        return features
    
    def _compute_dga_score(self, domains: np.ndarray) -> float:
        """
        Heuristic DGA score based on domain characteristics.
        Higher = more likely to be DGA.
        """
        if len(domains) == 0:
            return 0.0
        
        scores = []
        for domain in domains:
            domain_name = str(domain).lower()
            
            # Feature 1: High entropy in domain name
            entropy = self._domain_entropy(domain_name)
            
            # Feature 2: Uncommon TLD
            tld = domain_name.split(".")[-1]
            common_tlds = {"com", "org", "net", "edu", "gov", "co", "uk"}
            uncommon_tld = 1.0 if tld not in common_tlds else 0.0
            
            # Feature 3: Random consonant clusters
            vowels = "aeiou"
            consonant_runs = len(re.findall(r'[^aeiou\.]{3,}', domain_name))
            
            # Feature 4: Unusually long domain
            long_domain = 1.0 if len(domain_name) > 20 else 0.0
            
            score = (entropy + uncommon_tld + consonant_runs * 0.1 + long_domain) / 4.0
            scores.append(min(score, 1.0))
        
        return np.mean(scores) if scores else 0.0
    
    def _domain_entropy(self, domain: str) -> float:
        """Compute Shannon entropy of domain string"""
        from collections import Counter
        import math
        
        domain_name = domain.replace(".", "")
        if len(domain_name) == 0:
            return 0.0
        
        counts = Counter(domain_name)
        entropy = 0.0
        for count in counts.values():
            p = count / len(domain_name)
            entropy -= p * math.log2(p)
        
        # Normalize to 0-1
        max_entropy = math.log2(26)  # 26 letters
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _estimate_domain_age(self, domains: np.ndarray) -> float:
        """
        Estimate average domain age.
        Simplified: assume popular domains are older (365 days avg),
        unknown domains are newer (30 days avg).
        """
        common_domains = {
            "google.com", "facebook.com", "github.com", "microsoft.com",
            "amazon.com", "apple.com", "cloudflare.com", "akamai.com"
        }
        
        ages = []
        for domain in domains:
            if str(domain).lower() in common_domains:
                ages.append(365.0)
            else:
                ages.append(30.0)
        
        return np.mean(ages) if ages else 30.0
    
    def _compute_ttl_consistency(self, dns_df: pd.DataFrame) -> float:
        """Compute TTL standard deviation (consistency)"""
        if "ttls" not in dns_df or len(dns_df) == 0:
            return 0.0
        
        ttls = []
        for ttl_list in dns_df["ttls"]:
            if isinstance(ttl_list, list):
                ttls.extend(ttl_list)
            elif isinstance(ttl_list, (int, float)):
                ttls.append(ttl_list)
        
        if len(ttls) == 0:
            return 0.0
        
        return float(np.std(ttls))
    
    def _compute_query_type_entropy(self, dns_df: pd.DataFrame) -> float:
        """Compute entropy of query types (A, AAAA, MX, etc.)"""
        from collections import Counter
        import math
        
        qtypes = dns_df.get("qtype", pd.Series([])).dropna()
        if len(qtypes) == 0:
            return 0.0
        
        counts = Counter(qtypes)
        entropy = 0.0
        for count in counts.values():
            p = count / len(qtypes)
            entropy -= p * math.log2(p + 1e-10)
        
        return entropy
    
    def _compute_rapid_fire_rate(self, dns_df: pd.DataFrame) -> float:
        """Compute fraction of queries in rapid bursts (within 1 second)"""
        if len(dns_df) < 2:
            return 0.0
        
        times = dns_df["ts"].sort_values()
        time_diffs = times.diff().dt.total_seconds()
        
        rapid_queries = (time_diffs <= 1.0).sum()
        return rapid_queries / max(len(time_diffs) - 1, 1)
    
    def _compute_subdomain_depth(self, domains: np.ndarray) -> float:
        """Compute average number of subdomain levels"""
        if len(domains) == 0:
            return 0.0
        
        depths = [str(d).count(".") + 1 for d in domains]
        return np.mean(depths)
```

---

### Task 1.3: Temporal Statistics Extractor (2 days)

**File to create**: `src/features/temporal_features.py`

```python
"""Advanced temporal statistics feature extraction"""

import numpy as np
from scipy import stats
from typing import Dict
import pandas as pd

class TemporalStatsExtractor:
    """Extract temporal behavioral features from session data."""
    
    TEMPORAL_FEATURES = [
        "iat_entropy",              # Shannon entropy of inter-arrival times
        "iat_autocorr",             # Autocorrelation of IAT
        "iat_p25", "iat_p50", "iat_p75", "iat_p90",  # IAT percentiles
        "iat_mean", "iat_std",      # IAT statistics
        "burst_count",              # Number of packet bursts
        "burst_avg_size",           # Average packets per burst
        "burst_avg_spacing",        # Average time between bursts
        "periodicity_confidence",   # How regular is the timing?
        "recurrence_stability",     # How consistent are gaps?
    ]
    
    def extract_from_session(self, session_data: Dict) -> Dict[str, float]:
        """
        Extract temporal features from a session.
        
        session_data should contain:
        - 'timestamps' or 'iat' (inter-arrival times)
        - 'packets' or packet count info
        """
        
        features = {}
        
        # Compute inter-arrival times if not provided
        if "iat" in session_data:
            iats = np.array(session_data["iat"], dtype=np.float32)
        elif "timestamps" in session_data:
            timestamps = np.array(session_data["timestamps"], dtype=np.float64)
            iats = np.diff(timestamps)
        else:
            # No timing data
            return {f: 0.0 for f in self.TEMPORAL_FEATURES}
        
        if len(iats) == 0:
            return {f: 0.0 for f in self.TEMPORAL_FEATURES}
        
        # Remove outliers (use 99th percentile)
        threshold = np.percentile(iats, 99)
        iats_clean = iats[iats <= threshold]
        
        if len(iats_clean) == 0:
            iats_clean = iats
        
        # IAT Statistics
        features["iat_entropy"] = self._compute_entropy(iats_clean)
        features["iat_autocorr"] = self._compute_autocorrelation(iats_clean, lag=5)
        features["iat_p25"] = float(np.percentile(iats_clean, 25))
        features["iat_p50"] = float(np.percentile(iats_clean, 50))
        features["iat_p75"] = float(np.percentile(iats_clean, 75))
        features["iat_p90"] = float(np.percentile(iats_clean, 90))
        features["iat_mean"] = float(np.mean(iats_clean))
        features["iat_std"] = float(np.std(iats_clean))
        
        # Burst Analysis
        features["burst_count"] = len(self._detect_bursts(iats_clean))
        burst_sizes = self._get_burst_sizes(iats_clean)
        features["burst_avg_size"] = float(np.mean(burst_sizes)) if burst_sizes else 0.0
        features["burst_avg_spacing"] = self._compute_burst_spacing(iats_clean)
        
        # Periodicity
        features["periodicity_confidence"] = self._compute_periodicity_confidence(iats_clean)
        
        # Recurrence Stability
        features["recurrence_stability"] = self._compute_recurrence_stability(iats_clean)
        
        return features
    
    def _compute_entropy(self, iats: np.ndarray) -> float:
        """Compute Shannon entropy of IAT distribution"""
        # Discretize into bins
        bins = np.logspace(np.log10(iats.min() + 1e-6), np.log10(iats.max() + 1e-6), 20)
        hist, _ = np.histogram(iats, bins=bins)
        hist = hist[hist > 0]
        
        if len(hist) == 0:
            return 0.0
        
        p = hist / hist.sum()
        entropy = -np.sum(p * np.log2(p + 1e-10))
        return float(entropy)
    
    def _compute_autocorrelation(self, iats: np.ndarray, lag: int = 5) -> float:
        """Compute autocorrelation of IAT series"""
        if len(iats) <= lag:
            return 0.0
        
        mean = np.mean(iats)
        c0 = np.sum((iats - mean) ** 2) / len(iats)
        c_lag = np.sum((iats[:-lag] - mean) * (iats[lag:] - mean)) / len(iats)
        
        acf = c_lag / c0 if c0 > 0 else 0.0
        return float(max(-1, min(1, acf)))  # Clamp to [-1, 1]
    
    def _detect_bursts(self, iats: np.ndarray, threshold_percentile: float = 75.0) -> list:
        """Detect packet bursts (clusters of short inter-arrivals)"""
        threshold = np.percentile(iats, threshold_percentile)
        
        bursts = []
        current_burst = []
        
        for iat in iats:
            if iat <= threshold:
                current_burst.append(iat)
            else:
                if len(current_burst) > 1:
                    bursts.append(current_burst)
                current_burst = []
        
        if len(current_burst) > 1:
            bursts.append(current_burst)
        
        return bursts
    
    def _get_burst_sizes(self, iats: np.ndarray) -> list:
        """Get sizes of detected bursts"""
        bursts = self._detect_bursts(iats)
        return [len(burst) for burst in bursts]
    
    def _compute_burst_spacing(self, iats: np.ndarray) -> float:
        """Compute average time between bursts"""
        bursts = self._detect_bursts(iats)
        if len(bursts) < 2:
            return 0.0
        
        # Find gap between end of one burst and start of next
        gaps = []
        for i in range(len(bursts) - 1):
            gap = np.sum(bursts[i]) + np.sum(bursts[i + 1])
            gaps.append(gap)
        
        return float(np.mean(gaps)) if gaps else 0.0
    
    def _compute_periodicity_confidence(self, iats: np.ndarray) -> float:
        """
        Compute how periodic/regular the timing is.
        0 = random, 1 = perfectly periodic
        """
        # Coefficient of variation of IAT
        mean = np.mean(iats)
        std = np.std(iats)
        
        if mean == 0:
            return 0.0
        
        cv = std / mean  # Coefficient of variation
        
        # Low CV = high periodicity
        # cv=0 → confidence=1, cv=1 → confidence=0, cv>1 → confidence=-
        confidence = max(0.0, 1.0 - cv)
        return float(confidence)
    
    def _compute_recurrence_stability(self, iats: np.ndarray) -> float:
        """
        Compute how stable/consistent the recurrence is.
        Tests if IATs are similar to themselves (e.g., beacon with fixed interval).
        """
        # Use autocorrelation at lag 1
        acf_lag1 = self._compute_autocorrelation(iats, lag=1)
        
        # Also check if IAT is "unimodal" (single peak, not multi-modal)
        # Simplified: check if std is low relative to mean
        mean = np.mean(iats)
        std = np.std(iats)
        
        if mean == 0:
            return 0.0
        
        cv = std / mean
        unimodality = max(0.0, 1.0 - cv)
        
        # Combine autocorrelation and unimodality
        stability = (acf_lag1 + unimodality) / 2.0
        return float(max(0.0, min(1.0, stability)))
```

---

### Validation Test (1 day)

**Create**: `tests/test_temporal_features.py`

```python
import pytest
import numpy as np
from src.features.temporal_features import TemporalStatsExtractor

def test_temporal_feature_extraction():
    """Test temporal feature extraction"""
    extractor = TemporalStatsExtractor()
    
    # Create mock beacon-like data (regular intervals)
    regular_iats = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    
    session_data = {"iat": regular_iats}
    features = extractor.extract_from_session(session_data)
    
    # For regular beacon, periodicity should be high
    assert features["periodicity_confidence"] > 0.8
    assert features["iat_std"] < 0.1  # Very low variance
    
    # Create mock random data
    random_iats = np.random.exponential(1.0, size=100)
    session_data_random = {"iat": random_iats}
    features_random = extractor.extract_from_session(session_data_random)
    
    # For random data, periodicity should be low
    assert features_random["periodicity_confidence"] < 0.5
```

---

### Integration Test

**Update**: `src/data_loader/feature_transforms.py`

Add imports and integration:

```python
from src.features.tls_features import TLSFeatureExtractor
from src.features.dns_features import DNSFeatureExtractor
from src.features.temporal_features import TemporalStatsExtractor

def extract_extended_40_features(session, zeek_ssl_log=None, zeek_dns_log=None):
    """
    Extract all 40 features (original 10 + TLS 12 + DNS 8 + Temporal 10).
    
    Args:
        session: session data dict
        zeek_ssl_log: Optional Zeek SSL log for this session
        zeek_dns_log: Optional Zeek DNS log for this session
    
    Returns:
        numpy array of 40 features
    """
    
    # Original 10 features
    base_features = extract_base_10_features(session)
    
    # TLS features (12)
    tls_extractor = TLSFeatureExtractor()
    if zeek_ssl_log is not None:
        tls_features = tls_extractor.extract_from_zeek_ssl_log(zeek_ssl_log)
    else:
        tls_features = {f: 0.0 for f in tls_extractor.TLS_FEATURES}
    
    # DNS features (8)
    dns_extractor = DNSFeatureExtractor()
    if zeek_dns_log is not None:
        dns_features = dns_extractor.extract_from_zeek_dns_log(zeek_dns_log)
    else:
        dns_features = {f: 0.0 for f in dns_extractor.DNS_FEATURES}
    
    # Temporal features (10)
    temporal_extractor = TemporalStatsExtractor()
    temporal_features = temporal_extractor.extract_from_session(session)
    
    # Combine all
    all_features = np.concatenate([
        base_features,                                    # 10
        np.array(list(tls_features.values())),           # 12
        np.array(list(dns_features.values())),           # 8
        np.array(list(temporal_features.values())),      # 10
    ])
    
    assert len(all_features) == 40, f"Expected 40 features, got {len(all_features)}"
    return all_features
```

---

## Week 2: Feature Integration & Testing

### Task 2.1: Update Feature Schema

**Update**: `src/features/feature_config.py`

```python
# Add to feature_config.py

FEATURE_NAMES_EXTENDED = [
    # Original 10
    "duration",
    "bytes_in",
    "bytes_out",
    "packets_in",
    "packets_out",
    "protocol",
    "src_port",
    "dst_port",
    "timestamp",
    "reserved",
    
    # TLS (12)
    "tls_version",
    "tls_cipher_count",
    "tls_extension_count",
    "tls_supported_groups",
    "tls_alpn_count",
    "tls_session_reuse_rate",
    "tls_cert_reuse_rate",
    "tls_handshake_size",
    "tls_version_diversity",
    "tls_cipher_diversity",
    "tls_cert_validity_days",
    "tls_ja3_hash",
    
    # DNS (8)
    "dns_query_count",
    "dns_unique_domains",
    "dns_nxdomain_rate",
    "dns_dga_score",
    "dns_domain_age_avg",
    "dns_ttl_consistency",
    "dns_query_type_entropy",
    "dns_lookup_frequency",
    "dns_rapid_fire_rate",
    "dns_subdomain_depth_avg",
    
    # Temporal (10)
    "iat_entropy",
    "iat_autocorr",
    "iat_p25",
    "iat_p50",
    "iat_p75",
    "iat_p90",
    "iat_mean",
    "iat_std",
    "burst_count",
    "burst_avg_size",
    "burst_avg_spacing",
    "periodicity_confidence",
    "recurrence_stability",
]

FEATURE_DIM_EXTENDED = len(FEATURE_NAMES_EXTENDED)  # 40 features
assert FEATURE_DIM_EXTENDED == 40
```

---

### Task 2.2: Unit Tests & Documentation

- [ ] Run all unit tests
- [ ] Fix any failing tests
- [ ] Document feature meanings
- [ ] Create feature importance baseline

---

## Week 3: Dataset Preparation

### Task 3.1: UGR'16 Dataset Preparation

Create `src/pipelines/prepare_ugr16.py`

```python
"""Prepare UGR'16 dataset to standard NPZ format"""

def prepare_ugr16_dataset(raw_zeek_logs_dir, output_npz_path):
    """
    Convert UGR'16 Zeek logs to standard NPZ format.
    
    UGR'16 source: https://nesg.ugr.es/nesg-ugr-ips/
    """
    # Load Zeek logs
    # Extract sessions
    # Build numpy arrays
    # Save as NPZ
```

### Task 3.2: CICIDS2017 Dataset Preparation

Create `src/pipelines/prepare_cicids.py`

---

## Week 4: Cross-Dataset Training

### Task 4.1: Cross-Dataset Training Pipeline

Create `src/pipelines/cross_dataset_train_eval.py`

```python
def train_on_ctu13_eval_on_ugr16():
    """Train on CTU-13, evaluate on UGR'16"""
    
def train_on_ugr16_eval_on_ctu13():
    """Train on UGR'16, evaluate on CTU-13"""

def train_on_mixed_eval_on_each():
    """Train on mixed, evaluate on each dataset"""
```

---

## Week 5-6: Analysis & Documentation

### Task 5.1: Feature Importance

- Run ablation studies
- Measure contribution of each feature set
- Document findings

### Task 5.2: Baseline Comparisons

- Train XGBoost on 40 features
- Train Random Forest on 40 features  
- Compare against Transformer

---

## Success Checklist

✅ Week 1:
- [ ] TLS extractor implemented & tested
- [ ] DNS extractor implemented & tested
- [ ] Temporal extractor implemented & tested

✅ Week 2:
- [ ] Feature schema updated to 40 features
- [ ] All unit tests passing
- [ ] Documentation complete

✅ Week 3:
- [ ] UGR'16 dataset prepared
- [ ] CICIDS2017 dataset prepared
- [ ] MAWI sample prepared

✅ Week 4:
- [ ] Cross-dataset training executed
- [ ] Results logged and analyzed

✅ Week 5-6:
- [ ] Feature importance analysis complete
- [ ] Baseline model comparisons complete
- [ ] Report written

---

## Key Files to Create/Modify

```
NEW:
  src/features/tls_features.py
  src/features/dns_features.py
  src/features/temporal_features.py
  src/pipelines/prepare_ugr16.py
  src/pipelines/prepare_cicids.py
  src/pipelines/cross_dataset_train_eval.py
  tests/test_tls_features.py
  tests/test_dns_features.py
  tests/test_temporal_features.py

MODIFY:
  src/data_loader/feature_transforms.py
  src/features/feature_config.py
  backend/app/services/detection.py
  
MAINTAIN:
  src/models/transformer.py (unchanged, just accept 40 features)
  src/training/train_transformer.py (unchanged)
  src/evaluation/evaluate_transformer.py (unchanged)
```

---

## Command Sequence for Week 1

```bash
# Setup
cd d:\CyberShield
source .venv\Scripts\activate

# Create feature files
touch src/features/tls_features.py
touch src/features/dns_features.py
touch src/features/temporal_features.py

# Run tests
pytest tests/test_tls_features.py -v
pytest tests/test_dns_features.py -v
pytest tests/test_temporal_features.py -v

# Check integration
python -c "from src.features.tls_features import TLSFeatureExtractor; print('TLS OK')"
python -c "from src.features.dns_features import DNSFeatureExtractor; print('DNS OK')"
python -c "from src.features.temporal_features import TemporalStatsExtractor; print('Temporal OK')"
```

