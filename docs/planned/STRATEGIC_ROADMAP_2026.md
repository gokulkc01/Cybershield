# CyberShield Strategic Roadmap 2026-2027

## Vision Statement

Transform CyberShield from a **session-centric behavioral classifier** into a **hierarchical behavioral reasoning system** for modern C2 detection that generalizes across malware families, network environments, protocol evolution, and attacker adaptations.

**Core Philosophy**: Learn *behavioral invariants of malicious coordination*, not *dataset-specific artifacts or signatures*.

---

## Strategic Phases Overview

```
Phase 0 (Current)  → Session-centric baseline (96.36% recall on cross-family test)
Phase 1 (Q2 2026)  → Enhanced features + multi-dataset validation
Phase 2 (Q3 2026)  → Host-aware architecture + temporal memory
Phase 3 (Q4 2026)  → Hierarchical Transformer + self-supervised pretraining
Phase 4 (2027)     → Production deployment + continual adaptation
Phase 5 (2027+)    → Foundation models for network behavior
```

---

# PHASE 1: Enhanced Features & Multi-Dataset Validation (Q2 2026, 6 weeks)

## Objective
Prove session-centric model generalizes across **independent datasets** while adding critical missing features.

## 1.1 Feature Expansion

### Add TLS Behavioral Features

**What to Add**:
- JA3/JA4 fingerprints (TLS handshake signature)
- TLS extension ordering
- ALPN protocols used
- Certificate reuse patterns
- Session resumption behavior
- Cipher suite selection

**Implementation**:
```python
# src/features/tls_features.py (NEW)
class TLSFeatureExtractor:
    def extract_from_zeek_logs(zeek_df):
        # JA3/JA4 already computed by Zeek (ssl.log)
        # Extract: version, ciphers, extensions, ec_curves, etc.
        # Output: 12-15 new features
```

**Data Source**: Zeek `ssl.log` files (already available in CTU-13)

**Integration Point**: Feature pipeline before Transformer

---

### Add DNS Behavioral Features

**What to Add**:
- DNS lookup frequency (queries/sec)
- Domain reuse rate
- NXDOMAIN frequency (failed lookups)
- DGA probability score
- Domain age / freshness
- TTL anomalies
- Query type distribution (A, AAAA, MX, etc.)

**Implementation**:
```python
# src/features/dns_features.py (NEW)
class DNSFeatureExtractor:
    def extract_from_zeek_logs(zeek_df):
        # From dns.log
        # Output: 8-10 new features per session
```

**Data Source**: Zeek `dns.log` files

---

### Add Advanced Temporal Statistics

**What to Add**:
- Inter-arrival time (IAT) entropy
- Burst morphology (cluster size, spacing)
- Periodicity confidence (how regular is the beaconing)
- Recurrence stability (callback consistency)
- Timing distribution percentiles (p25, p50, p75, p90)
- Autocorrelation of inter-arrival times

**Implementation**:
```python
# src/features/temporal_features.py (NEW)
class TemporalStatsExtractor:
    def extract_from_session(session):
        iat = compute_inter_arrival_times(session)
        return {
            'iat_entropy': entropy(iat),
            'iat_autocorr': autocorrelation(iat, lag=5),
            'burst_morphology': cluster_bursts(iat),
            'periodicity_confidence': detect_periodicity(iat),
            'iat_p25': percentile(iat, 25),
            'iat_p50': percentile(iat, 50),
            'iat_p75': percentile(iat, 75),
            'iat_p90': percentile(iat, 90),
        }
```

---

### Update Feature Schema

**Current** (10 features):
```
duration, bytes_in, bytes_out, packets_in, packets_out, 
protocol, src_port, dst_port, timestamp, reserved
```

**Enhanced** (35-40 features):
```
# Original 10
duration, bytes_in, bytes_out, packets_in, packets_out,
protocol, src_port, dst_port, timestamp, reserved,

# TLS (12 new)
tls_version, tls_cipher_count, tls_extension_count,
tls_extensions_list, tls_curves, tls_alpn,
tls_session_reuse, tls_cert_reuse, tls_handshake_size,
tls_version_diversity, tls_cipher_diversity, tls_cert_validity_days,

# DNS (8 new)
dns_query_count, dns_nxdomain_rate, dns_domain_reuse_rate,
dns_dga_score, dns_domain_age_days, dns_ttl_anomaly,
dns_query_type_entropy, dns_lookup_frequency,

# Temporal (10 new)
iat_entropy, iat_autocorr, burst_morphology, periodicity_confidence,
iat_p25, iat_p50, iat_p75, iat_p90, iat_mean, iat_std,
```

**Backward Compatibility**: Keep old 10-feature model working + add "extended" 35-feature model.

---

## 1.2 Cross-Dataset Validation

### Prepare Additional Datasets

| Dataset | Purpose | Status | Size | Timeline |
|---------|---------|--------|------|----------|
| CTU-13 | Baseline (current) | ✅ Ready | 1.5M sessions | Week 1 |
| UGR'16 | Realistic benign | ⚠️ Need to prepare | 1M+ flows | Week 2-3 |
| CICIDS2017 | Diverse attacks | ⚠️ Need to prepare | 2.8M flows | Week 2-3 |
| MAWI Archive | Longitudinal benign | ⚠️ Need to prepare | 100M+ flows | Week 3-4 |
| Enterprise (Synthetic) | Cloud-native | ⚠️ Need to generate | 500K sessions | Week 4-5 |

**Implementation**:
```python
# src/pipelines/prepare_datasets.py (NEW)
class DatasetPreparer:
    def prepare_ugr16(raw_dir):
        """Convert UGR'16 to standard NPZ format"""
        
    def prepare_cicids(raw_dir):
        """Convert CICIDS2017 to standard NPZ format"""
        
    def prepare_mawi(raw_dir):
        """Sample MAWI archive, convert to standard NPZ format"""
```

### Cross-Dataset Training

```python
# src/pipelines/cross_dataset_train_eval.py (NEW)
def train_and_validate():
    """
    Train/val on one dataset
    Test on entirely different datasets
    """
    configs = [
        {"train": "ctu13", "test": "ugr16", "name": "CTU→UGR"},
        {"train": "ctu13", "test": "cicids", "name": "CTU→CICIDS"},
        {"train": "ugr16", "test": "ctu13", "name": "UGR→CTU"},
        {"train": "cicids", "test": "ctu13", "name": "CICIDS→CTU"},
        {"train": "ugr16+cicids", "test": "ctu13", "name": "UGR+CICIDS→CTU"},
    ]
    
    results = {}
    for config in configs:
        model = train(config["train"])
        metrics = evaluate(model, config["test"])
        results[config["name"]] = metrics
        print(f"{config['name']}: Recall={metrics['recall']:.2%}")
```

**Expected Outcome**: Measure true generalization (not just cross-family, but cross-dataset).

---

## 1.3 Feature Importance Analysis

### Ablation Study v2

Test importance of each new feature category:

```python
# src/analysis/feature_importance_v2.py (NEW)
def ablation_study():
    """
    Train models with different feature sets:
    1. Baseline (10 features)
    2. + TLS features
    3. + DNS features
    4. + Temporal stats
    5. All features (35-40)
    
    Measure recall, AUC, FPR for each
    """
    
    results = {
        "baseline_10": {"recall": 0.9636, "auc": 0.9887},
        "baseline+tls": {"recall": 0.97, "auc": 0.991},
        "baseline+dns": {"recall": 0.974, "auc": 0.989},
        "baseline+temporal": {"recall": 0.971, "auc": 0.990},
        "all_features_35": {"recall": 0.978, "auc": 0.994},
    }
```

### Feature Importance Ranking

```python
# Identify which features matter most
# Use SHAP, permutation importance, or attention weights
# Output: ranked list of feature importance
```

---

## 1.4 Baseline Model Comparisons

Compare Transformer against simpler baselines:

```python
# src/models/baseline_comparisons.py (NEW)
def compare_models():
    """
    Train all models on same dataset (CTU-13)
    Test on same holdout (Conficker)
    """
    
    models = {
        "random_forest": RandomForestClassifier(),
        "xgboost": XGBClassifier(),
        "lstm": LSTMClassifier(),
        "gru": GRUClassifier(),
        "temporal_cnn": TemporalCNNClassifier(),
        "transformer": C2Transformer(),
        "statistical_beacon": StatisticalBeaconDetector(),
    }
    
    results = train_and_eval_all(models)
    # Output: comparison table showing Transformer superiority
```

**Goal**: Justify Transformer complexity vs. simpler approaches.

---

## 1.5 Implementation Timeline

| Week | Task | Owner | Status |
|------|------|-------|--------|
| Week 1 | Implement TLS feature extractor | Dev | ⚠️ Start |
| Week 1-2 | Implement DNS feature extractor | Dev | ⚠️ Start |
| Week 2 | Implement temporal stats extractor | Dev | ⚠️ Start |
| Week 2 | Update feature schema & tests | Dev | ⚠️ Start |
| Week 2-3 | Prepare UGR'16 dataset | Data Eng | ⚠️ Start |
| Week 3 | Prepare CICIDS2017 dataset | Data Eng | ⚠️ Start |
| Week 3 | Prepare MAWI sample dataset | Data Eng | ⚠️ Start |
| Week 3-4 | Run cross-dataset training | ML Eng | ⚠️ Start |
| Week 4 | Feature importance analysis | ML Eng | ⚠️ Start |
| Week 4-5 | Baseline model comparisons | ML Eng | ⚠️ Start |
| Week 5-6 | Documentation & validation | All | ⚠️ Start |

---

## 1.6 Success Criteria

✅ **Phase 1 Success** when:
- [ ] 35-40 feature set implemented and tested
- [ ] Model trains on 35-feature set without degradation
- [ ] Cross-dataset recall ≥ 70% (CTU trained, UGR/CICIDS tested)
- [ ] Feature importance analysis shows TLS/DNS/temporal features matter
- [ ] Transformer outperforms baselines by ≥ 5% in AUC
- [ ] Ablation study published
- [ ] All cross-dataset comparisons completed

---

---

# PHASE 2: Host-Aware Architecture & Temporal Memory (Q3 2026, 8 weeks)

## Objective
Add host-level reasoning. Move beyond session isolation.

## 2.1 Host Timeline Construction

### Define HostTimeline

```python
# src/features/host_timeline.py (REIMPLEMENT)
class HostTimeline:
    """
    Reconstructs temporal behavior of a host across multiple sessions.
    """
    
    def __init__(self, host_id: str):
        self.host_id = host_id
        self.sessions = []  # Chronologically ordered
        self.inter_session_gaps = []
        self.process_chains = {}
        self.domain_access_history = {}
        self.tls_cert_reuse = {}
        
    def add_session(self, session: SessionData, timestamp: float):
        """Add session to timeline, maintain chronological order"""
        self.sessions.append((timestamp, session))
        
    def get_persistence_score(self, session_idx: int) -> float:
        """
        For a given session, measure:
        - How many times has this source-dest pair communicated?
        - What's the inter-session gap distribution?
        - Is communication regular/beacon-like?
        """
        
    def get_multi_session_context(self, session_idx: int, lookback_sessions=10):
        """
        Return features aggregated from previous 10 sessions:
        - Average bytes
        - Average duration
        - Common destinations
        - Certificate reuse
        - Domain churn
        """
        
    def detect_callback_cluster(self, session_idx: int, window=300):
        """
        Detect if session is part of a callback cluster
        (multiple sessions to same dest within time window)
        """
```

### Host Behavior Aggregation

```python
# src/features/host_aggregates.py (NEW)
class HostBehaviorAggregator:
    
    def get_session_context(host_timeline: HostTimeline, session_idx: int):
        """
        Aggregate features from host's session history:
        
        Returns dict with:
        - persistence_score: how often has this host communicated?
        - callback_count: in past hour, how many sessions to same dest?
        - domain_reuse_rate: % of new vs. repeated destinations
        - tls_cert_reuse_rate: % of sessions reusing certs
        - inter_session_regularity: how uniform are gaps?
        - historical_bytes_distribution: how typical is current session?
        - process_correlation: which processes active during session?
        """
```

---

## 2.2 Multi-Session Feature Set

### Add Host-Centric Features

```python
MULTI_SESSION_FEATURES = {
    "persistence": [
        "session_count_past_hour",
        "session_count_past_day",
        "unique_destinations_past_hour",
        "unique_destinations_past_day",
        "callback_cluster_size",
        "callback_cluster_regularity",
    ],
    
    "continuity": [
        "inter_session_gap_mean",
        "inter_session_gap_std",
        "inter_session_gap_autocorr",
        "same_dest_revisit_rate",
        "port_reuse_rate",
    ],
    
    "coordination": [
        "multi_dest_burst_frequency",
        "sequential_session_bytes_correlation",
        "tls_cert_reuse_persistence",
        "domain_ttl_consistency",
    ],
    
    "historical": [
        "bytes_mean_from_history",
        "bytes_std_from_history",
        "duration_mean_from_history",
        "packets_mean_from_history",
        "process_identity_consistency",
    ]
}
```

---

## 2.3 Hierarchical Feature Extraction

Update feature pipeline to support both:
- **Session-only mode** (for single-session inference)
- **Host-aware mode** (for correlated sessions)

```python
# src/data_loader/feature_transforms_v2.py (NEW)
def extract_session_features(session, mode="session_only"):
    """
    Extract features from single session.
    """
    base_features = extract_base_10_features(session)
    tls_features = extract_tls_features(session)
    dns_features = extract_dns_features(session)
    temporal_features = extract_temporal_features(session)
    
    if mode == "session_only":
        return concatenate([
            base_features,      # 10
            tls_features,       # 12
            dns_features,       # 8
            temporal_features,  # 10
        ])  # 40 total
    
    elif mode == "host_aware":
        # Requires host_timeline and session_idx
        raise ValueError("Use extract_host_aware_features instead")

def extract_host_aware_features(session, host_timeline, session_idx):
    """
    Extract features with host context.
    """
    session_features = extract_session_features(session, mode="session_only")
    host_context = get_session_context(host_timeline, session_idx)
    multi_session_features = extract_multi_session_features(host_context)
    
    return concatenate([
        session_features,        # 40
        multi_session_features,  # 15-20
    ])  # 55-60 total
```

---

## 2.4 Host-Aware Dataset Construction

```python
# src/data_loader/host_aware_loader.py (NEW)
class HostAwareSessionDataset:
    """
    Loads sessions WITH host timeline context.
    """
    
    def __init__(self, npz_path, host_metadata_path):
        self.sessions = load_npz(npz_path)
        self.host_metadata = load_json(host_metadata_path)
        self.host_timelines = self._build_host_timelines()
        
    def _build_host_timelines(self):
        """Group sessions by source host, maintain chronological order"""
        timelines = {}
        for session in sorted_by_timestamp(self.sessions):
            host_id = session['src_ip']
            if host_id not in timelines:
                timelines[host_id] = HostTimeline(host_id)
            timelines[host_id].add_session(session, session['timestamp'])
        return timelines
        
    def __getitem__(self, idx):
        """
        Returns:
        - session tensor (40-60 features)
        - label
        - host_context (for explainability)
        """
        session, host_id = self.get_session_with_host(idx)
        host_timeline = self.host_timelines[host_id]
        session_idx = host_timeline.get_session_index(session['id'])
        
        features = extract_host_aware_features(session, host_timeline, session_idx)
        label = session['label']
        
        return {
            "features": features,
            "label": label,
            "host_id": host_id,
            "session_idx": session_idx,
            "host_context": host_timeline.get_metadata(),
        }
```

---

## 2.5 Update Transformer for Multi-Session Reasoning

Add **host memory mechanism**:

```python
# src/models/transformer_v2_host_aware.py (NEW)
class C2TransformerV2HostAware(nn.Module):
    """
    Enhanced Transformer with host-level temporal memory.
    
    Architecture:
    
    Session Features (40-60 dims)
        ↓
    Session Encoder (Transformer)
        ↓
    Session Embedding (64 dims)
        ↓
    Host Memory Module
    (stores last N session embeddings)
        ↓
    Attention over session history
        ↓
    Host Context Vector
        ↓
    Classification
    """
    
    def __init__(self, feature_dim=60, session_memory_size=10):
        super().__init__()
        
        # Session encoding
        self.session_encoder = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
        )
        
        # Host memory
        self.session_memory_size = session_memory_size
        self.host_memory = nn.Linear(64, 64)  # Learnable memory projection
        
        # Multi-session attention
        self.attention = nn.MultiheadAttention(64, 4)
        
        # Classification
        self.classifier = nn.Sequential(
            nn.Linear(64 * 2, 128),  # concat session + host context
            nn.GELU(),
            nn.Linear(128, 1),
        )
    
    def forward(self, session_features, host_context_features):
        """
        session_features: (B, 20, 60) - time steps × features
        host_context_features: (B, 15) - aggregated host history
        """
        
        # Encode session
        session_embedding = self.session_encoder(session_features)  # (B, 20, 64)
        session_cls = session_embedding[:, 0]  # CLS token
        
        # Host memory
        host_embedding = self.host_memory(host_context_features.unsqueeze(1))  # (B, 1, 64)
        
        # Attention between session and host
        attended, _ = self.attention(
            session_cls.unsqueeze(1),      # Query
            host_embedding,                 # Key
            host_embedding,                 # Value
        )
        
        # Combine
        combined = torch.cat([session_cls, attended.squeeze(1)], dim=1)  # (B, 128)
        logits = self.classifier(combined)
        
        return logits.squeeze(-1)
```

---

## 2.6 Training with Host Context

```python
# src/training/train_host_aware.py (NEW)
def train_host_aware_model():
    """
    Train model that uses host timeline context.
    """
    
    dataset = HostAwareSessionDataset(
        npz_path="data/processed/ctu13_host_aware.npz",
        host_metadata_path="data/processed/ctu13_host_metadata.json",
    )
    
    model = C2TransformerV2HostAware(feature_dim=60, session_memory_size=10)
    
    for batch in dataloader:
        session_features = batch["features"]  # (B, 20, 60)
        host_context = batch["host_context"]  # (B, 15)
        labels = batch["label"]
        
        logits = model(session_features, host_context)
        loss = focal_loss(logits, labels)
        
        loss.backward()
        optimizer.step()
```

---

## 2.7 Implementation Timeline

| Week | Task |
|------|------|
| Week 1 | Reimplement HostTimeline class |
| Week 1-2 | Build host metadata extraction |
| Week 2 | Implement multi-session features |
| Week 2 | Update feature extraction pipeline |
| Week 3 | Create HostAwareSessionDataset |
| Week 3 | Implement C2TransformerV2HostAware |
| Week 4 | Train model on CTU-13 with host context |
| Week 4-5 | Evaluate on cross-dataset with host context |
| Week 5-6 | Compare single-session vs. host-aware performance |
| Week 6-7 | Error analysis & debugging |
| Week 7-8 | Documentation & publication |

---

## 2.8 Success Criteria

✅ **Phase 2 Success** when:
- [ ] HostTimeline class functional and tested
- [ ] Multi-session features extracted correctly
- [ ] HostAwareSessionDataset loads data without errors
- [ ] C2TransformerV2HostAware trains to convergence
- [ ] Host-aware model recall ≥ 97.5% on CTU-13
- [ ] Cross-dataset recall ≥ 72% (improvement over session-only)
- [ ] Inference latency still < 100ms with host context
- [ ] Ablation shows host context improves metrics

---

---

# PHASE 3: Hierarchical Transformer & Self-Supervised Pretraining (Q4 2026, 10 weeks)

## Objective
Build foundation model capabilities. Learn general network behavior before fine-tuning on C2.

## 3.1 Self-Supervised Pretraining

Move beyond supervised learning. Train on massive unlabeled traffic.

### Pretraining Task 1: Masked Flow Prediction

```python
# src/training/pretrain_masked_flow.py (NEW)
class MaskedFlowPretraining:
    """
    Similar to BERT masked language modeling.
    
    Given: [flow_1, flow_2, MASKED, flow_4, flow_5]
    Predict: flow_3
    
    Goal: Learn "flow grammar"
    """
    
    def create_pretraining_data(traffic_flows):
        """Randomly mask 15% of flows in sequences"""
        
    def train_step(model, batch):
        """
        Mask flows → predict masked flow
        Loss: MSE between predicted and actual features
        """
```

### Pretraining Task 2: Temporal Consistency

```python
# src/training/pretrain_temporal_consistency.py (NEW)
class TemporalConsistencyPretraining:
    """
    Contrastive learning on traffic sequences.
    
    Positive pairs: same host, slightly different time windows
    Negative pairs: different hosts
    
    Goal: Learn that host behavior is persistent
    """
    
    def create_positive_pairs(host_timeline):
        """Sample two time windows from same host"""
        
    def train_step(model, positive_batch, negative_batch):
        """Contrastive loss: similar hosts close, different hosts far"""
```

### Pretraining Task 3: Next Flow Prediction

```python
# src/training/pretrain_next_flow.py (NEW)
class NextFlowPretraining:
    """
    Causal language modeling for traffic.
    
    Given: [flow_1, flow_2, flow_3, flow_4]
    Predict: flow_5
    
    Goal: Learn traffic sequences are predictable
    """
    
    def train_step(model, batch):
        """Autoregressive prediction, MSE loss"""
```

### Pretraining Architecture

```python
# src/models/foundation_encoder.py (NEW)
class FoundationNetworkEncoder(nn.Module):
    """
    General-purpose network behavior encoder.
    
    Pretrain on massive unlabeled traffic.
    Transfer to C2 detection.
    Transfer to DGA detection, botnet detection, etc.
    """
    
    def __init__(self):
        self.flow_encoder = TransformerEncoder(...)
        self.host_encoder = TemporalTransformer(...)
        self.pooling = AdaptivePooling(...)
        
    def forward(self, flows, host_context):
        """Encode flows + host context into embedding"""
        
        flow_embeddings = self.flow_encoder(flows)
        host_embedding = self.host_encoder(host_context)
        
        combined = self.pooling([flow_embeddings, host_embedding])
        return combined
```

---

## 3.2 Hierarchical Transformer Architecture

```python
# src/models/hierarchical_transformer.py (NEW)
class HierarchicalBehaviorTransformer(nn.Module):
    """
    Multi-level hierarchy for behavioral reasoning.
    
    Packet Level → Flow Level → Session Level → Host Level
    
    Each level learns its own encoder, then passes to next level.
    """
    
    def __init__(self):
        # Level 1: Packet encoder
        self.packet_encoder = nn.TransformerEncoderLayer(
            d_model=32,     # Small: individual packets
            nhead=2,
            num_layers=1,
        )
        
        # Level 2: Flow encoder (sequence of packets)
        self.flow_encoder = nn.TransformerEncoderLayer(
            d_model=64,     # Medium: flows
            nhead=4,
            num_layers=2,
        )
        
        # Level 3: Session encoder (sequence of flows)
        self.session_encoder = nn.TransformerEncoderLayer(
            d_model=128,    # Larger: sessions
            nhead=8,
            num_layers=3,
        )
        
        # Level 4: Host encoder (sequence of sessions)
        self.host_encoder = nn.TransformerEncoderLayer(
            d_model=256,    # Largest: host timeline
            nhead=16,
            num_layers=4,
        )
        
        # Detection head
        self.detector = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
    
    def forward(self, packets, flows, sessions, host_timeline):
        """
        packets: (B, P, 8) - P packets per flow
        flows: (B, F, 20) - F flows per session
        sessions: (B, S, 40) - S sessions per host
        host_timeline: (B, T, 30) - T time bins per host
        
        Each level processes and embeds its input,
        then passes to next level.
        """
        
        # Level 1: Encode packets
        packet_embeds = self.packet_encoder(packets)  # (B, P, 32)
        packet_summary = packet_embeds.mean(dim=1)    # (B, 32)
        
        # Level 2: Encode flows
        flow_input = torch.cat([
            flows,                           # (B, F, 20)
            packet_summary.unsqueeze(1).expand(-1, flows.shape[1], -1)  # (B, F, 32)
        ], dim=-1)  # (B, F, 52)
        flow_embeds = self.flow_encoder(flow_input)  # (B, F, 64)
        flow_summary = flow_embeds.mean(dim=1)       # (B, 64)
        
        # Level 3: Encode sessions
        session_input = torch.cat([
            sessions,                        # (B, S, 40)
            flow_summary.unsqueeze(1).expand(-1, sessions.shape[1], -1)  # (B, S, 64)
        ], dim=-1)  # (B, S, 104)
        session_embeds = self.session_encoder(session_input)  # (B, S, 128)
        session_summary = session_embeds.mean(dim=1)          # (B, 128)
        
        # Level 4: Encode host timeline
        host_input = torch.cat([
            host_timeline,                   # (B, T, 30)
            session_summary.unsqueeze(1).expand(-1, host_timeline.shape[1], -1)  # (B, T, 128)
        ], dim=-1)  # (B, T, 158)
        host_embeds = self.host_encoder(host_input)  # (B, T, 256)
        host_summary = host_embeds.mean(dim=1)       # (B, 256)
        
        # Detection
        logits = self.detector(host_summary)
        return logits.squeeze(-1)
```

---

## 3.3 Hierarchical Dataset Construction

```python
# src/data_loader/hierarchical_loader.py (NEW)
class HierarchicalTrafficDataset:
    """
    Loads traffic at multiple levels of hierarchy.
    """
    
    def __init__(self, zeek_logs_dir, npz_dir):
        self.zeek_logs = load_zeek_logs(zeek_logs_dir)
        self.sessions = load_npz(npz_dir)
        
    def __getitem__(self, idx):
        """
        Returns:
        - packets: raw packet headers
        - flows: flow-level features
        - sessions: session-level features
        - host_timeline: host temporal context
        - label
        """
        
        session_id = self.sessions[idx]['id']
        host_id = self.sessions[idx]['src_ip']
        
        # Extract packets for this session
        packets = extract_packets(
            self.zeek_logs,
            session_id,
            max_packets=100
        )  # (100, 8)
        
        # Extract flows in this session
        flows = extract_flows(
            self.zeek_logs,
            session_id,
            max_flows=20
        )  # (20, 20)
        
        # Get session features
        sessions = extract_session_context(
            host_id,
            max_sessions=10
        )  # (10, 40)
        
        # Get host timeline
        host_timeline = extract_host_timeline(
            host_id,
            window_size=1_hour,
            time_bins=30
        )  # (30, 30)
        
        label = self.sessions[idx]['label']
        
        return {
            "packets": packets,
            "flows": flows,
            "sessions": sessions,
            "host_timeline": host_timeline,
            "label": label,
        }
```

---

## 3.4 Training Strategy

### Stage 1: Unsupervised Pretraining (4 weeks)

Train on massive unlabeled traffic (MAWI archive + UGR'16):
- Masked flow prediction
- Temporal consistency
- Next flow prediction

**No C2 labels needed.**

```python
def pretrain_on_unlabeled_traffic():
    """
    Pretraining loop:
    
    1. Load 100M+ flows from MAWI
    2. Create self-supervised tasks
    3. Train FoundationNetworkEncoder
    4. Save checkpoint
    """
```

### Stage 2: Supervised Fine-Tuning (3 weeks)

Fine-tune pretrained model on labeled C2 data:

```python
def finetune_on_c2_data():
    """
    Fine-tuning loop:
    
    1. Load pretrained FoundationNetworkEncoder
    2. Attach C2 detection head
    3. Train on CTU-13 + other labeled C2
    4. Evaluate on cross-dataset
    """
```

### Stage 3: Adversarial Training (2 weeks)

Train against Red-Agent mutations:

```python
def adversarial_training():
    """
    Adversarial training loop:
    
    1. Load baseline C2 model
    2. For each epoch:
        a. Sample C2 sessions
        b. Apply Red-Agent mutations at increasing severity
        c. Train model to detect mutated C2
        d. Update detector
    
    Goal: Model learns to detect C2 despite mutations
    """
```

---

## 3.5 Implementation Timeline

| Week | Task |
|------|------|
| Week 1-2 | Implement FoundationNetworkEncoder |
| Week 2 | Implement self-supervised pretraining tasks |
| Week 2-3 | Prepare MAWI + UGR'16 for pretraining |
| Week 3-4 | Run unsupervised pretraining (weeks 3-5 of calendar) |
| Week 5 | Load pretrained model, attach C2 head |
| Week 5-6 | Supervised fine-tuning on CTU-13 |
| Week 6 | Evaluate on cross-dataset |
| Week 6-7 | Integrate Red-Agent into training loop |
| Week 7-8 | Adversarial training with mutations |
| Week 8 | Evaluate robustness under mutations |
| Week 9-10 | Documentation & publication |

---

## 3.6 Success Criteria

✅ **Phase 3 Success** when:
- [ ] Unsupervised pretraining runs to convergence
- [ ] Pretrained model transfers to C2 detection
- [ ] Fine-tuned model recall ≥ 97% on CTU-13
- [ ] Cross-dataset recall ≥ 75%
- [ ] Adversarial training improves robustness
- [ ] Model maintains ≥ 94% recall under Red-Agent mutations
- [ ] Inference latency < 200ms (hierarchical architecture)
- [ ] Foundation encoder can be transferred to other tasks (DGA, botnet, etc.)

---

---

# PHASE 4: Production Deployment & Continual Adaptation (Q1-Q2 2027, ongoing)

## Objective
Operationalize the system. Monitor performance. Adapt to new threats.

## 4.1 Shadow Mode Deployment

```python
# src/inference/shadow_mode_pipeline.py (NEW)
class ShadowModePipeline:
    """
    Deploy model to production without alerts.
    Log all detections for analyst review.
    """
    
    def process_live_zeek_logs(zeek_log_path):
        """
        1. Load Zeek log
        2. Extract sessions
        3. Run batch inference
        4. Log results (no alerts)
        5. Collect ground truth labels
        """
        
        sessions = load_zeek_log(zeek_log_path)
        
        # Reconstruct host timelines
        host_timelines = build_host_timelines(sessions)
        
        # Batch inference
        results = []
        for batch in batches(sessions, batch_size=64):
            logits = model(batch, host_timelines)
            scores = torch.sigmoid(logits).detach().cpu().numpy()
            results.extend(scores)
        
        # Log all results
        log_to_siem(results)
        
        return results
```

## 4.2 Continual Adaptation

```python
# src/training/continual_learning.py (NEW)
class ContinualLearner:
    """
    Periodically retrain on new data + analyst feedback.
    """
    
    def monthly_retraining():
        """
        Every month:
        1. Collect new unlabeled traffic
        2. Collect analyst labels (ground truth)
        3. Retrain on new C2 families
        4. Evaluate on held-out test set
        5. Deploy if performance maintained
        """
        
    def drift_detection():
        """
        Monitor for concept drift:
        - If recall drops > 5%, alert
        - If FPR increases > 2%, alert
        - If new malware family emerges, retrain
        """
```

---

# PHASE 5: Foundation Models for Network Behavior (2027+)

## Vision

Transfer CyberShield to broader tasks:
- DGA detection
- Botnet detection
- Data exfiltration detection
- Anomaly detection
- Network intrusion detection

All using same pretrained encoder.

```python
# src/models/foundation_applications.py
class DGADetector(FoundationNetworkEncoder):
    """Transfer learning: C2 encoder → DGA detector"""
    
class BotnetDetector(FoundationNetworkEncoder):
    """Transfer learning: C2 encoder → Botnet detector"""

class DataExfiltrationDetector(FoundationNetworkEncoder):
    """Transfer learning: C2 encoder → Data exfil detector"""
```

---

---

# IMPLEMENTATION CHECKLIST

## Phase 1 (Q2 2026) - Enhanced Features

- [ ] TLS feature extractor
- [ ] DNS feature extractor
- [ ] Temporal statistics extractor
- [ ] Updated feature schema (40 features)
- [ ] Backward compatibility layer
- [ ] UGR'16 dataset preparation
- [ ] CICIDS2017 dataset preparation
- [ ] MAWI archive sampling
- [ ] Cross-dataset training pipeline
- [ ] Feature importance analysis
- [ ] Baseline model comparisons
- [ ] All tests passing
- [ ] Documentation & publication

## Phase 2 (Q3 2026) - Host-Aware Architecture

- [ ] HostTimeline class reimplementation
- [ ] Host metadata extraction
- [ ] Multi-session features (15-20 new features)
- [ ] HostAwareSessionDataset
- [ ] C2TransformerV2HostAware model
- [ ] Training loop for host-aware model
- [ ] Evaluation on CTU-13
- [ ] Cross-dataset evaluation
- [ ] Error analysis
- [ ] Performance benchmarking
- [ ] All tests passing
- [ ] Documentation & publication

## Phase 3 (Q4 2026) - Hierarchical & Pretraining

- [ ] FoundationNetworkEncoder
- [ ] Masked flow prediction task
- [ ] Temporal consistency task
- [ ] Next flow prediction task
- [ ] HierarchicalBehaviorTransformer
- [ ] HierarchicalTrafficDataset
- [ ] Unsupervised pretraining (4 weeks)
- [ ] Supervised fine-tuning
- [ ] Adversarial training integration
- [ ] Cross-dataset evaluation
- [ ] Robustness testing
- [ ] All tests passing
- [ ] Documentation & publication

## Phase 4 (Q1-Q2 2027) - Production

- [ ] Shadow mode deployment
- [ ] Live metrics collection
- [ ] Ground truth labeling
- [ ] Error analysis on live data
- [ ] Continual learning pipeline
- [ ] Drift detection
- [ ] Monthly retraining automation
- [ ] Monitoring & alerts
- [ ] Operations runbook
- [ ] Quarterly adversarial testing

## Phase 5 (2027+) - Foundation Models

- [ ] Transfer to DGA detection
- [ ] Transfer to botnet detection
- [ ] Transfer to data exfiltration
- [ ] Multi-task learning
- [ ] Domain adaptation

---

---

# RESOURCE REQUIREMENTS

## Team Composition

- **2-3 ML Engineers** (full-time)
  - Feature engineering
  - Model training & evaluation
  - Benchmarking & analysis
  
- **1-2 Data Engineers** (full-time)
  - Dataset preparation & validation
  - Feature pipeline
  - Zeek log processing
  
- **1 Security Researcher** (part-time)
  - Threat modeling
  - Domain expertise validation
  - Analyst feedback collection
  
- **1 DevOps Engineer** (part-time)
  - Deployment automation
  - Monitoring & logging
  - SIEM integration

## Infrastructure

- **GPU Compute**: 1x A100 or 2x V100 for training
- **Storage**: 500GB minimum (MAWI samples, preprocessed datasets)
- **Processing**: 16+ CPU cores for data preprocessing
- **Monitoring**: ELK stack or similar for production logging

---

---

# SUCCESS METRICS

## Phase 1 Success

| Metric | Target |
|--------|--------|
| Feature set size | 40 features |
| Model trains without degradation | ✅ |
| Cross-dataset recall | ≥70% |
| Feature importance discovered | ✅ |
| Transformer vs. baselines | ≥5% AUC improvement |

## Phase 2 Success

| Metric | Target |
|--------|--------|
| Host-aware recall on CTU-13 | ≥97.5% |
| Cross-dataset recall | ≥72% |
| Inference latency | <100ms |
| Host context improves metrics | ✅ |

## Phase 3 Success

| Metric | Target |
|--------|--------|
| Pretraining converges | ✅ |
| Transfer to C2 detection | ✅ |
| Fine-tuned recall | ≥97% |
| Cross-dataset recall | ≥75% |
| Robustness under mutations | ≥94% |
| Inference latency | <200ms |

## Phase 4 Success

| Metric | Target |
|--------|--------|
| Live recall | ≥75% |
| Live FPR | ≤5% |
| System uptime | ≥99% |
| Inference latency | <100ms (p95) |
| Monthly retraining | Automated |

---

---

# Risk Mitigation

## Risk 1: Phase 1 Features Don't Help

**Mitigation**:
- Extensive feature importance analysis
- Ablation studies before moving forward
- Fallback: proceed with baseline 10-feature model to Phase 2

## Risk 2: Host Timeline Construction Breaks

**Mitigation**:
- Extensive testing of HostTimeline class
- Validation against manual inspection
- Fallback: session-only model suffices for Phase 2

## Risk 3: Pretraining Doesn't Transfer

**Mitigation**:
- Test transfer before full Phase 3 commit
- Self-supervised pretraining can be optional
- Fallback: supervised fine-tuning alone

## Risk 4: Production Performance Degrades

**Mitigation**:
- Shadow mode first (no alerts)
- Extensive testing before alerting
- Quick rollback plan
- Fallback: old model remains active

---

---

# Timeline Summary

```
Q2 2026 (6 weeks):  Phase 1 - Enhanced Features
Q3 2026 (8 weeks):  Phase 2 - Host-Aware Architecture
Q4 2026 (10 weeks): Phase 3 - Hierarchical & Pretraining
Q1-Q2 2027: Phase 4 - Production Deployment
2027+:      Phase 5 - Foundation Models
```

**Total**: ~6 months to complete Phases 1-3. Phases 4-5 ongoing.

---

---

# Documentation & Publishing

Each phase includes:
- **Technical report** (paper-ready)
- **Code documentation**
- **Reproducibility materials**
- **Benchmark comparisons**
- **Lessons learned**

Goal: Publish findings in top-tier security/ML venues.

---

---

# Final Notes

This is an **ambitious but achievable** roadmap to transform CyberShield into a sophisticated behavioral reasoning system.

The key principles:
- **One feature at a time** (Phase 1: add features + cross-dataset validation)
- **Validate before advancing** (each phase has success criteria)
- **Fallback plans** (if something doesn't work, we have alternatives)
- **Production first** (Phase 4 brings operational reality to the research)
- **Long-term vision** (Phase 5 opens the door to foundation models)

**Start with Phase 1** (6 weeks). If successful, proceed to Phase 2.

