# CyberShield v2: Red-Agent Architecture

## Executive Summary

The Red-Agent is a **domain-constrained adversarial behavioral mutation system** for studying C2 robustness under realistic mutation pressure.

**NOT a malware framework. NOT offensive tooling.**

**IS**: A behavioral robustness research system for understanding which behavioral invariants survive adversarial mutation.

---

## Core Philosophy

### What This System Does

The Red-Agent learns which **behavioral transformations** reduce detector confidence while:
- Preserving realistic operational properties
- Maintaining functional C2 behavior
- Avoiding adversarial artifacts
- Staying within physical constraints

### What This System Does NOT Do

- Generate malware payloads
- Generate exploits
- Generate offensive persistence mechanisms
- Generate invalid or impossible traffic
- Perform offensive operations

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Red-Agent System                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Behavioral Session Tensors (20, 12)                        │
│          │                                                   │
│          ├─→ [Mutation Engines] ──→ Mutated Session         │
│          │   - Timing (jitter, sleep)                       │
│          │   - Flow (packets, bytes)                        │
│          │   - TLS (padding, reuse)                         │
│          │   - Composite (chained)                          │
│          │                                                   │
│          ├─→ [Validity Constraints] ──→ HARD BOUNDS         │
│          │   - Timing validity                              │
│          │   - Flow plausibility                            │
│          │   - Protocol realism                             │
│          │   - Statistical integrity                        │
│          │                                                   │
│          ├─→ [Realism Discriminator] ──→ Realism Score      │
│          │   (PRIMARY SAFEGUARD)                            │
│          │   - Statistical plausibility                     │
│          │   - Distribution shift                           │
│          │   - Behavioral coherence                         │
│          │   - Correlation preservation                     │
│          │   - Protocol realism                             │
│          │                                                   │
│          ├─→ [Detector] ──→ Risk Scores                     │
│          │   Original & Mutated                             │
│          │                                                   │
│          └─→ [Reward Engine] ──→ Multi-Objective Reward     │
│              (5 Components)                                  │
│              - Evasion reward                               │
│              - Realism reward (⭐ CRITICAL)                 │
│              - Functionality reward                         │
│              - Stability reward                             │
│              - Diversity bonus                              │
│                                                              │
│  [PPO Policy Network]                                        │
│  Learns mutation selection & severity                       │
│                                                              │
│  [PPO Trainer]                                              │
│  Updates policy via policy gradient + value estimation      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Module Structure

```
src/red_agent/
├── __init__.py
├── orchestrator.py              [Main coordinator]
│
├── mutations/
│   ├── __init__.py
│   ├── base_mutation_engine.py  [Abstract interface]
│   ├── timing_mutations.py      [Jitter, sleep]
│   ├── flow_mutations.py        [Packets, bytes, duration]
│   ├── tls_mutations.py         [Padding, session reuse]
│   └── composite_mutations.py   [Multi-stage mutations]
│
├── validation/
│   ├── __init__.py
│   ├── behavioral_validity_constraints.py  [HARD BOUNDS]
│   └── realism_discriminator.py            [SAFEGUARD]
│
├── policy/
│   └── __init__.py
│   └── policy_network.py                   [PPO agent]
│
└── training/
    ├── __init__.py
    ├── reward_engine.py                    [Multi-objective]
    └── ppo_trainer.py                      [PPO update]
```

---

## Critical Components

### 1. Behavioral Validity Constraints ⭐⭐⭐

**Purpose**: Prevent impossible or nonsensical mutations.

**Location**: `src/red_agent/validation/behavioral_validity_constraints.py`

**Enforces Bounds**:

| Category | Lower Bound | Upper Bound |
|----------|------------|------------|
| Session Duration | 0.1 sec | 30 days |
| Beacon Interval (IAT) | 1 ms | 24 hours |
| Jitter Fraction | 1% | 50% |
| Packet Count | 2 | 1 million |
| Byte Volume | 50B | 10GB |
| Byte Ratio | 0.01x | 100x |
| Port Range | 1 | 65535 |
| Packets/sec | 0.1 | 1000 |

**Violations**: Penalize reward by -0.5 per violation.

---

### 2. Realism Discriminator ⭐⭐⭐

**Purpose**: PRIMARY SAFEGUARD against reward hacking.

**Location**: `src/red_agent/validation/realism_discriminator.py`

**Scores** (0-1):
1. **Statistical Plausibility**: Feature means within training distribution (±3σ)
2. **Distribution Shift**: KS-test on mutated vs. original
3. **Behavioral Coherence**: Feature relationships remain logical
4. **Correlation Preservation**: Real dependencies maintained
5. **Protocol Realism**: Bytes/packets/duration relationships valid

**Threshold**: 0.6 = realistic, 0.4-0.6 = questionable, <0.4 = unrealistic

**Reward Impact**:
- Realism ≥ 0.7: Mild positive reward (+0.0 to +0.3)
- Realism 0.4-0.7: Neutral (-0.2)
- Realism < 0.4: STRONG PENALTY (-1.0 to -1.8)

---

### 3. Multi-Objective Reward Engine ⭐⭐

**Purpose**: Balance evasion, realism, functionality, stability, diversity.

**Location**: `src/red_agent/training/reward_engine.py`

**Reward Components**:

```python
Total Reward = (
    0.40 * evasion_reward          # Confidence decrease
  + 0.30 * realism_reward          # ⭐ CRITICAL (safeguard)
  + 0.15 * functionality_reward    # C2 operational properties
  + 0.10 * stability_reward        # No adversarial artifacts
  + 0.05 * diversity_reward        # Exploration
  + constraint_penalty             # Hard violations
)
```

**Evasion Reward**:
- Positive if detector confidence decreased
- Extra bonus if classification flipped (C2 → benign)

**Realism Reward** (THE MOST IMPORTANT):
- Realism < 0.4: -1.0 to -1.8 (strong penalty)
- Realism 0.4-0.7: -0.2 (neutral)
- Realism ≥ 0.7: +0.0 to +0.3 (mild positive)

**Constraint Penalty**:
- -0.5 per constraint violation (MANDATORY)

---

### 4. Mutation Engines

#### A. Timing Mutations

**What**: Jitter inter-arrival times, extend session duration (sleep).

**Parameters**:
- Jitter: up to 20% of base IAT at max severity
- Sleep: up to 10x duration extension at max severity

**Example**: Beacon every 30 seconds → adds ±3 sec jitter, extends to 300 sec total

#### B. Flow Mutations

**What**: Modify packet counts, byte volumes, duration.

**Mechanisms**:
- Poisson noise on packet counts
- Multiplicative Gaussian on byte volumes
- Multiplicative noise on duration

**Example**: 50 orig packets → 40-60 packets; 5KB → 2.5-10KB

#### C. TLS Mutations

**What**: TLS padding, session reuse patterns.

**Mechanisms**:
- Add padding bytes (hide payload size)
- Remove packets via session reuse

**Example**: 5KB + 30% padding → 6.5KB; 100 packets - 20 handshakes → 80 packets

#### D. Composite Mutations

**What**: Coordinated multi-stage transformation chains.

**Strategies**:
- `stealth_persistence`: Timing + flow (jitter + sparse)
- `tls_evasion`: TLS + flow (padding + reshaping)
- `sophisticated_c2`: All three (timing + TLS + flow)

**Example**: High jitter + long sleep + TLS padding + byte spreading

---

### 5. PPO Policy Network

**Location**: `src/red_agent/policy/policy_network.py`

**Input**: Session tensor (20, D)

**Architecture**:
```
Session (20,12) → Flatten → Encoder (2 layers, 256 hidden)
                    ├→ Mutation Head → Categorical (5 mutations)
                    ├→ Severity Head → Continuous [0, 1]
                    └→ Value Head → Scalar (critic)
```

**Output**:
- Mutation type (5 classes: timing, flow, tls, composite, none)
- Severity level [0, 1]
- Value estimate (for advantage computation)

---

### 6. PPO Trainer

**Location**: `src/red_agent/training/ppo_trainer.py`

**Algorithm**: Proximal Policy Optimization

**Key Parameters**:
- Clip ratio: 0.2 (PPO clipping)
- Gamma: 0.99 (discount factor)
- GAE lambda: 0.95 (advantage estimation)
- Entropy coefficient: 0.01 (exploration)
- Value coefficient: 0.5 (value function weight)

**Training Loop**:
1. Collect trajectories (mutation → evaluation → reward)
2. Compute GAE advantages
3. Multiple epochs of mini-batch PPO updates
4. Gradient clipping + policy + value losses

---

## Training Progression

### Phase 1: Frozen Detector (STARTING POINT)

**Goal**: Understand mutation space and discover behavioral weaknesses.

**Setup**:
- Load pre-trained CyberShield detector
- Freeze detector weights (no retraining)
- Train Red-Agent policy against frozen target

**Duration**: 2-4 weeks

**Output**:
- Mutation effectiveness rankings
- Fragile behavioral assumptions
- Evasive mutation corpus

### Phase 2: Robustness Analysis

**Goal**: Characterize detector vulnerability.

**Tasks**:
- Analyze mutation success rates
- Rank mutations by effectiveness
- Identify behavioral invariants
- Generate robustness report

**Output**:
- Mutation impact ranking
- Failure patterns
- Behavioral fragility analysis

### Phase 3: Curated Adversarial Corpus

**Goal**: Build realistic evasive dataset.

**Tasks**:
- Filter mutations by realism threshold
- Validate against external datasets (UWF, MCFP)
- Create corpus of realistic evasions
- Document constraints

**Output**:
- Curated adversarial dataset
- Validation metrics
- Constraint documentation

### Phase 4: Controlled Adversarial Retraining

**Goal**: Improve detector robustness.

**Setup**:
- Train detector with curated evasive samples
- Evaluate on held-out test sets
- Measure robustness improvement

**Important**: Use external validation (never co-adapt exclusively).

---

## Usage Examples

### Example 1: Single Mutation

```python
from src.red_agent.orchestrator import RedAgentOrchestrator
import numpy as np

# Initialize
agent = RedAgentOrchestrator()

# Create synthetic session
session = np.random.randn(20, 12) * 0.1  # Normalized
mask = np.ones(20, dtype=bool)
mask[10:] = False  # Mask out padding

# Mutate
mutation_result = agent.mutate_session(
    session,
    mask,
    mutation_type="timing",
    severity=0.7
)

# Check realism
print(f"Realistic: {mutation_result.is_realistic}")
print(f"Realism score: {mutation_result.realism_score:.3f}")
print(f"Violations: {mutation_result.constraint_violations}")
```

### Example 2: Batch Evaluation

```python
from src.red_agent.orchestrator import RedAgentOrchestrator

agent = RedAgentOrchestrator()

# Load detector (mock)
def detector_fn(session):
    # Returns confidence [0, 1]
    return np.random.uniform(0, 1)

# Create batch
sessions = np.random.randn(10, 20, 12) * 0.1
masks = np.ones((10, 20), dtype=bool)

# Evaluate
results = agent.batch_evaluate_mutations(
    sessions,
    masks,
    detector_fn
)

# Analyze
for i, result in enumerate(results):
    print(f"Sample {i}:")
    print(f"  Reward: {result.reward:.3f}")
    print(f"  Realism: {result.mutation_result.realism_score:.3f}")
    print(f"  Conf orig→mutated: {result.detector_confidence_original:.3f} → {result.detector_confidence_mutated:.3f}")
```

### Example 3: PPO Training

```python
from src.red_agent.policy import MutationPolicyNetwork
from src.red_agent.training import PPOTrainer
import torch

# Create network
policy = MutationPolicyNetwork(
    input_dim=12,
    hidden_dim=256,
    num_mutations=5
)

# Create trainer
trainer = PPOTrainer(policy, learning_rate=3e-4)

# Training loop (pseudocode)
for epoch in range(num_epochs):
    # Collect trajectory
    sessions = torch.randn(32, 20, 12)
    mutations = [(i % 5, 0.5) for i in range(32)]
    rewards = [agent.evaluate_mutation(...) for ...]
    
    # PPO update
    metrics = trainer.update(
        sessions,
        mutations,
        rewards,
        old_log_probs=[...],
        num_epochs=3
    )
    print(f"Epoch {epoch}: loss={metrics['loss']:.3f}")
```

---

## Integration with CyberShield Backend

### Detector Integration

```python
# In backend/app/api/robustness.py
from src.red_agent.orchestrator import RedAgentOrchestrator
from app.services.detection import detection_service

agent = RedAgentOrchestrator()

def detector_inference(session):
    """Wrapper for detector."""
    risk_score, is_suspicious, confidence = detection_service.infer_tensor(session)
    return confidence  # Return confidence for adversarial feedback

# Use in Red-Agent
results = agent.batch_evaluate_mutations(
    sessions,
    masks,
    detector_inference
)
```

### API Endpoint

```python
@router.post("/red-agent/evaluate")
async def evaluate_mutations(request: EvaluateMutationsRequest):
    """Run Red-Agent mutation evaluation."""
    agent = RedAgentOrchestrator()
    
    results = agent.batch_evaluate_mutations(
        sessions=request.sessions,
        masks=request.masks,
        detector_inference_fn=detector_inference,
        mutation_types=request.mutation_types
    )
    
    return {
        "mutations": [r.to_dict() for r in results],
        "effectiveness": compute_effectiveness(results)
    }
```

---

## Validation & Metrics

### Phase 1 Validation Metrics

- **Evasion Rate**: Fraction of mutations that reduce detector confidence
- **Average Confidence Delta**: Mean decrease in detector confidence
- **Realism Preservation**: % of mutations with realism > 0.6
- **Constraint Violation Rate**: % of mutations with violations
- **Diversity**: Number of distinct mutation strategies discovered

### Phase 2 Robustness Metrics

- **Recall Degradation**: Loss in detection recall under mutation
- **Behavioral Invariance**: Recall on mutated vs. baseline
- **Fragile Samples**: Samples detected baseline but not mutated
- **Mutation Effectiveness Ranking**: Top N most effective mutations

### External Validation

**Critical**: Evaluate on held-out datasets ONLY.

Supported datasets:
- UWF (external benign)
- MCFP (external benign)
- CTU-13 scenarios 10-13 (cross-dataset C2)
- Real Sliver captures (if available)

---

## Safety & Constraints

### Hard Guarantees

1. **No Malware Generation**: System operates on behavioral tensors only
2. **No Exploit Generation**: No payload/persistence/infection code
3. **Behavioral Realism**: Hard bounds on timing, flow, protocol
4. **Realism Discriminator**: Primary safeguard against artifacts
5. **Constraint Penalties**: Reward hacking prevention

### Monitoring

- Log all mutations (type, severity, realism)
- Track constraint violations
- Monitor reward distributions
- Validate detector integrity after each phase

---

## Future Extensions

### Planned

- [ ] Distributed mutation evaluation (multi-GPU)
- [ ] Self-play adversarial co-training
- [ ] Multi-family robustness study
- [ ] Protocol-specific mutation strategies
- [ ] Behavioral generalization analysis

### Research Directions

- How do mutations transfer across families?
- What are fundamental behavioral invariants?
- Can realism + evasion co-optimize?
- Which real C2 behaviors are most fragile?

---

## References

- PPO: Schulman et al., 2017
- GAE: Schulman et al., 2016
- Adversarial ML: Goodfellow et al., 2014
- C2 Behavioral Analysis: CyberShield project

---

## Authors & Attribution

Red-Agent System: CyberShield v2 Project
Behavioral Robustness Research: [Your Team]
Date: May 2026

---

**Remember**: This system is for research and defensive robustness evaluation only.
**Never** use for offensive operations.
