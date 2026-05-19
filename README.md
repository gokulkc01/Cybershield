# CyberShield: Host-Aware Adversarial C2 Detection

CyberShield is an advanced, research-grade Command-and-Control (C2) detection platform. It shifts the paradigm of network security from session-centric heuristics to **causal host-history modeling**, validated dynamically via a Proximal Policy Optimization (PPO) Reinforcement Learning adversary known as the **Red-Agent**.

## 🌟 Key Innovations

### 1. Host-Aware Domain-Adaptive Transformer
Modern C2 frameworks (e.g., Cobalt Strike, Sliver) spread malicious beacons across multiple sessions over long durations to blend in with benign traffic. Evaluating sessions in isolation leaves critical blind spots. 
To solve this, CyberShield features a novel architecture:
- **Causal Host History Encoder:** Processes a chronological sequence of up to 32 historical sessions per host using a 2-layer Transformer encoder (`nhead=4`). 
- **Session Backbone:** A shared 3-layer Transformer encoder (`d_model=64`) that independently extracts dense representations for both the current session and historical sessions.
- **Host-Aware Fusion Head:** Concatenates the current session embedding with the temporal host context.
- **Domain-Adaptive Calibration:** Calibrates the final C2 logit using domain-specific learnable scale and bias parameters, ensuring strict False Positive Rate (FPR) constraints across varied network environments.

### 2. Red-Agent (RL-based Adversarial Evaluation)
To dynamically test and validate the model's structural robustness, CyberShield employs **Red-Agent**, an adversarial RL pipeline.
- **Mutation Engines:** Mutates real C2 traffic via Timing (jitter), Flow (byte ratios), TLS (SNI/ports), and Composite alterations.
- **Hybrid Action Space (PPO):** The policy network uses a discrete `Categorical` distribution to pick the mutation type, and a continuous `Normal` distribution for mutation severity.
- **Realism Constraints:** The agent is bounded by strict `BehavioralValidityConstraints` (e.g., no impossible latencies or negative packets) and a `RealismDiscriminator` to ensure mutations are operationally viable.
- **Stable Training:** Powered by Generalized Advantage Estimation (GAE) with batch-level advantage normalization.

## 📊 Empirical Results

The platform was rigorously evaluated on a curated, leakage-free dataset combining legacy botnets (CTU-13), diverse malware families (Stratosphere MCFP), and modern post-exploitation frameworks (UWF-ZeekData24).

### Benchmark Performance (FPR ≤ 0.0150 Budget)
| Architecture | Test Recall | Test FPR | Test F1 |
| :--- | :--- | :--- | :--- |
| **Session-Only Baseline** | 89.97% | 0.00% | 0.9472 |
| **Host-Aware Transformer** | **99.91%** | **0.00%** | **0.9996** |

By anchoring on temporal history, the Host-Aware model achieved a massive ~10% absolute recall improvement over the baseline.

### Adversarial Robustness
Subjected to a 100-episode training loop against the Red-Agent:
- **Evasion Rate:** **2.0%**
- **Conclusion:** The adversary was effectively squashed. Because the detector anchors on the unalterable history of the host, session-level feature spoofing is mathematically insufficient to evade detection.

---

## 🚀 Quickstart & Architecture

### Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Running the Red-Agent Training Loop
To train the Red-Agent against the Host-Aware model:
```bash
python -m scripts.train_host_aware_red_agent \
  --checkpoint experiments/host_aware_uwf_attack_validation/best_host_aware_transformer.pth \
  --data data/processed/host_aware_uwf_attack_mvp/train_host_windows.npz \
  --episodes 100 \
  --batch-size 32
```

### System Components
- `src/models/`: Contains the `HostAwareDomainAdaptiveTransformer` and baseline models.
- `src/red_agent/`: Contains the PPO Trainer, Mutation Policy Network, Orchestrator, and Reward engines.
- `src/data_loader/`: Contains the canonical `HostWindowArrays` dataset builders and `.npz` processing tools.

---
*Built for rigorous, leakage-free cybersecurity ML research.*
