# Self-Predictive Representation Change in Transformers

## Project Status: HYPOTHESIS LOCKED 🔒

---

## 1. Core Hypothesis (FINAL - No More Drift)

> **Transformers fail silently because internal processing quality is not observable from outputs. We hypothesize that self-predicted representation change (Δ) at each layer provides an internal signal of processing difficulty that detects failure earlier and more reliably than output entropy, particularly under distribution shift.**

### What This Is NOT
- ❌ NOT early exit (we never stop early)
- ❌ NOT uncertainty estimation (post-hoc)
- ❌ NOT "just another aux head" (predicts internal dynamics)
- ❌ NOT efficiency paper (diagnostic, not control)

### What This IS
- ✅ Self-prediction of representation dynamics
- ✅ Internal signal of processing difficulty
- ✅ Silent failure detection
- ✅ Comparative claim: Δ > entropy under failure

---

## 2. Technical Specifications (FROZEN)

### 2.1 Δ Definition

**Cosine Distance (Layer-wise):**
```python
Δ_l = 1 - cos_sim(h_l, h_{l+1})
```

**Why Cosine:**
- Invariant to norm (layers can scale without "changing")
- Sensitive to rotation (actual representational change)
- Cheap to compute
- Easy to explain to reviewers

### 2.2 Training Objective

**Binary Δ with Learned Threshold:**
```python
# Actual representation change
actual_delta = 1 - F.cosine_similarity(h_l, h_{l+1}, dim=-1)

# Learned threshold per layer
y_l = (actual_delta > tau_l).float()  # tau_l is learned

# Predictor loss
loss = F.binary_cross_entropy_with_logits(predicted_useful_l, y_l)
```

### 2.3 Architecture Addition

```
Standard Transformer Layer:
    h_l → Attention → FFN → h_{l+1}

Our Addition:
    h_l → Attention → FFN → h_{l+1}
                ↓
         Δ-Predictor (small MLP)
                ↓
         predicted_delta_l
```

**Δ-Predictor:** 2-layer MLP, hidden_dim = 64, input = h_l (or pooled h_l)

---

## 3. Make-or-Break Experiments (DO THESE FIRST)

### Philosophy
> These experiments MUST be run before any paper writing. If they fail, hypothesis is dead. No sunk cost fallacy.

---

### 🔴 Experiment 1: Silent Failure Detection (CRITICAL)

**Question:** Does Δ detect failure before accuracy drops, while entropy stays falsely confident?

**Setup:**
1. Train small transformer (6-layer, 256-dim) on clean data
2. Add Δ-predictors to each layer
3. Apply perturbations at inference time

**Perturbation Ladder:**
| Level | Perturbation | Expected Behavior |
|-------|--------------|-------------------|
| 0 | Clean | High accuracy, low Δ, low entropy |
| 1 | Gaussian noise (20dB SNR) | Slight accuracy drop |
| 2 | Gaussian noise (10dB SNR) | Moderate accuracy drop |
| 3 | Token shuffle (10%) | Attention disruption |
| 4 | Token shuffle (30%) | Severe disruption |
| 5 | Random input | Complete failure |

**Measurements:**
```python
for perturbation_level in range(6):
    accuracy = measure_task_accuracy(model, perturbed_data)
    entropy = measure_output_entropy(model, perturbed_data)
    delta_signal = measure_mean_predicted_delta(model, perturbed_data)
    
    results.append({
        'level': perturbation_level,
        'accuracy': accuracy,
        'entropy': entropy,
        'delta': delta_signal
    })
```

**Kill Criteria:**
- ❌ DEAD if: Δ and entropy degrade at same rate
- ❌ DEAD if: Δ doesn't increase under perturbation
- ❌ DEAD if: Δ increases but doesn't correlate with accuracy drop

**Success Criteria:**
- ✅ ALIVE if: Δ increases BEFORE accuracy drops significantly
- ✅ ALIVE if: Entropy stays flat while Δ increases (silent failure detected)

**Visualization (Figure 1 Draft):**
```
Y-axis: Normalized signal value
X-axis: Perturbation level (0-5)

Lines:
- Accuracy (blue, solid) - should drop late
- Entropy (red, dashed) - should stay falsely low
- Δ signal (green, solid) - should rise early
```

---

### 🔴 Experiment 2: Δ Predicts Actual Layer Utility (MECHANISTIC)

**Question:** Does predicted Δ correlate with actual layer importance?

**Setup:**
1. Take trained model from Exp 1
2. Freeze all weights
3. For each layer l:
   - Record predicted Δ_l on test set
   - Ablate layer l (replace with identity or skip)
   - Measure accuracy drop

**Procedure:**
```python
layer_importance = []
predicted_deltas = []

for l in range(num_layers):
    # Get predicted delta for this layer
    pred_delta = model.get_predicted_delta(layer=l, data=test_data)
    predicted_deltas.append(pred_delta.mean())
    
    # Ablate layer and measure impact
    accuracy_with_layer = evaluate(model, test_data)
    accuracy_without_layer = evaluate(model, test_data, skip_layer=l)
    importance = accuracy_with_layer - accuracy_without_layer
    layer_importance.append(importance)

correlation = spearmanr(predicted_deltas, layer_importance)
```

**Kill Criteria:**
- ❌ DEAD if: Spearman correlation < 0.5
- ❌ DEAD if: Predicted Δ is uniform across layers (learned nothing)
- ❌ DEAD if: Correlation is negative (predictor is wrong)

**Success Criteria:**
- ✅ ALIVE if: Spearman correlation > 0.7
- ✅ ALIVE if: High-Δ layers cause big accuracy drops when removed
- ✅ ALIVE if: Low-Δ layers can be removed with minimal impact

**Visualization (Figure 2 Draft):**
```
Scatter plot:
- X-axis: Predicted Δ (mean per layer)
- Y-axis: Accuracy drop when layer ablated
- Each point = one layer
- Show correlation coefficient
```

---

### 🔴 Experiment 3: OOD Generalization (THE KILL SHOT)

**Question:** Does Δ detect OOD inputs better than entropy?

**Setup:**
1. Train on Domain A (e.g., clean speech / standard text)
2. Test on Domain B (e.g., noisy speech / different domain)
3. Binary classification: Is this input OOD?

**Domains (choose based on your setup):**

*Option A - Speech:*
| Train | Test (OOD) |
|-------|------------|
| Clean speech | Noisy speech (babble, music) |
| Native accent | Foreign accent |
| Studio recording | Phone quality |

*Option B - Text:*
| Train | Test (OOD) |
|-------|------------|
| Wikipedia | Reddit comments |
| English | Code-switched text |
| Formal | Informal/slang |

**OOD Detection Signals:**
```python
# For each test input
signals = {
    'entropy': -output_entropy,           # Lower entropy = more "confident"
    'max_softmax': max_softmax_prob,      # Higher = more "confident"  
    'delta': mean_predicted_delta,        # Higher = more "difficult"
}

# Ground truth
is_ood = [0]*len(in_domain) + [1]*len(out_domain)

# AUROC for each signal
for name, signal in signals.items():
    auroc = roc_auc_score(is_ood, signal)
    print(f"{name}: AUROC = {auroc:.3f}")
```

**Kill Criteria:**
- ❌ DEAD if: Δ AUROC ≤ Entropy AUROC
- ❌ DEAD if: Δ AUROC < 0.65 (barely better than random)
- ❌ DEAD if: All methods fail equally

**Success Criteria:**
- ✅ ALIVE if: Δ AUROC > Entropy AUROC by ≥ 0.05
- ✅ ALIVE if: Δ AUROC > 0.75
- ✅ PAPER-WORTHY if: Δ AUROC > 0.80 AND Entropy < 0.65

**Visualization (Figure 3 Draft):**
```
ROC curves:
- Entropy-based detection (red)
- Max-softmax detection (orange)  
- Δ-based detection (green)

Show AUROC values in legend
```

---

## 4. Experimental Protocol

### Phase 1: Make-or-Break (Week 1-2)

**Day 1-3: Setup**
- [ ] Implement Δ-predictor module
- [ ] Implement cosine distance Δ calculation
- [ ] Setup perturbation pipeline
- [ ] Setup evaluation metrics

**Day 4-7: Experiment 1 (Silent Failure)**
- [ ] Train base model
- [ ] Run perturbation ladder
- [ ] Plot accuracy vs entropy vs Δ
- [ ] **DECISION POINT: Continue or kill?**

**Day 8-10: Experiment 2 (Layer Utility)**
- [ ] Run layer ablation study
- [ ] Compute correlation
- [ ] **DECISION POINT: Correlation > 0.5?**

**Day 11-14: Experiment 3 (OOD)**
- [ ] Prepare OOD test sets
- [ ] Compute AUROC comparisons
- [ ] **FINAL DECISION: Paper exists?**

### Phase 2: Paper-Ready (Week 3-4, only if Phase 1 passes)

- [ ] Clean up figures
- [ ] Additional baselines (if compute allows)
- [ ] Ablation studies
- [ ] Write paper

---

## 5. Implementation Skeleton

### Model Architecture
```python
class DeltaPredictiveTransformer(nn.Module):
    def __init__(self, base_config):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(base_config) 
            for _ in range(base_config.num_layers)
        ])
        
        # Δ-predictors: one per layer
        self.delta_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(base_config.hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
            for _ in range(base_config.num_layers)
        ])
        
        # Learned thresholds
        self.tau = nn.Parameter(torch.ones(base_config.num_layers) * 0.1)
    
    def forward(self, x, return_deltas=False):
        hidden_states = [x]
        predicted_deltas = []
        actual_deltas = []
        
        h = x
        for i, layer in enumerate(self.layers):
            h_prev = h
            h = layer(h)
            
            # Predict delta from previous hidden state
            pred_delta = self.delta_predictors[i](h_prev.mean(dim=1))  # pooled
            predicted_deltas.append(pred_delta)
            
            # Actual delta (for training)
            with torch.no_grad():
                actual = 1 - F.cosine_similarity(
                    h_prev.mean(dim=1), 
                    h.mean(dim=1), 
                    dim=-1
                )
            actual_deltas.append(actual)
            
            hidden_states.append(h)
        
        if return_deltas:
            return h, predicted_deltas, actual_deltas
        return h
```

### Training Loop Addition
```python
def compute_delta_loss(predicted_deltas, actual_deltas, tau):
    """Auxiliary loss for Δ prediction"""
    total_loss = 0
    for i, (pred, actual) in enumerate(zip(predicted_deltas, actual_deltas)):
        # Binary target: did this layer change representation meaningfully?
        target = (actual > tau[i]).float()
        loss = F.binary_cross_entropy_with_logits(pred.squeeze(), target)
        total_loss += loss
    return total_loss / len(predicted_deltas)

# In training loop:
output, pred_deltas, actual_deltas = model(batch, return_deltas=True)
task_loss = compute_task_loss(output, labels)
delta_loss = compute_delta_loss(pred_deltas, actual_deltas, model.tau)

total_loss = task_loss + lambda_delta * delta_loss  # lambda_delta = 0.1
```

### Evaluation Utilities
```python
def get_delta_signal(model, data):
    """Get mean predicted Δ as failure signal"""
    model.eval()
    _, pred_deltas, _ = model(data, return_deltas=True)
    # Average across layers
    return torch.stack([d.squeeze() for d in pred_deltas]).mean(dim=0)

def get_entropy_signal(model, data):
    """Get output entropy as baseline signal"""
    model.eval()
    logits = model.get_logits(data)
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * probs.log()).sum(dim=-1)
    return entropy
```

---

## 6. Success/Failure Decision Matrix

| Experiment | Kill Threshold | Pass Threshold | Paper-Worthy |
|------------|---------------|----------------|--------------|
| Exp 1: Silent Failure | Δ ≈ entropy curve | Δ rises before accuracy drops | Δ rises 2+ levels before accuracy |
| Exp 2: Layer Utility | Correlation < 0.5 | Correlation > 0.7 | Correlation > 0.85 |
| Exp 3: OOD Detection | Δ AUROC ≤ Entropy | Δ AUROC > Entropy + 0.05 | Δ > 0.80, Entropy < 0.65 |

**Overall Decision:**
- **All 3 pass** → Full paper mode
- **2/3 pass** → Workshop paper, investigate failure
- **1/3 pass** → Pivot or abandon
- **0/3 pass** → Hypothesis falsified, move on

---

## 7. Reviewer Defense Preparation

### Attack: "This is just early exit"
> **Defense:** "Our method does not perform early exit during training or inference. Δ is diagnostic, not a control signal. We analyze predicted Δ to detect silent failures, not to skip computation."

### Attack: "Representation change is arbitrary"
> **Defense:** "We do not claim Δ = usefulness theoretically. We empirically validate that predicted Δ correlates with (1) actual ablation impact and (2) failure detection, making it a useful proxy for processing difficulty."

### Attack: "Entropy is a weak baseline"
> **Defense:** "Entropy is the standard baseline for uncertainty. Our contribution is showing it fails systematically under silent failure conditions where Δ succeeds. We engineer scenarios that expose this gap."

### Attack: "Why cosine distance?"
> **Defense:** "Cosine distance is invariant to representation scale while capturing rotational change. We show in ablations that norm-based Δ fails to detect certain failure modes (Appendix X)."

---

## 8. What NOT To Do

- ❌ Do NOT add attention confidence as second main contribution
- ❌ Do NOT claim hallucination detection (scope creep)
- ❌ Do NOT claim interpretability (different paper)
- ❌ Do NOT claim efficiency gains (we're not doing early exit)
- ❌ Do NOT compare against CALM/ACT unless forced by reviewers
- ❌ Do NOT run experiments beyond the 3 core ones until Phase 1 passes

---

## 9. Paper Framing (For Later)

**Title Options:**
1. "Self-Predictive Layers Reveal Silent Failures in Transformers"
2. "Detecting Transformer Failures Before They Happen"
3. "Representation Change Prediction for Failure-Aware Transformers"

**One-Sentence Contribution:**
> We show that transformers can learn to predict their own representation dynamics, and this internal signal detects processing failures earlier and more reliably than output-based measures.

**Three Contribution Bullets (Draft):**
1. We identify *silent failure* as a fundamental limitation: transformers can fail without observable output signals.
2. We propose *self-predictive layers* that estimate representation change (Δ) as an internal processing difficulty signal.
3. We demonstrate Δ detects failures earlier than entropy and generalizes to OOD detection where entropy-based methods collapse.

---

## 10. References to Keep Handy

- Early Exit: Schwartz et al., "The Right Tool for the Job: Matching Model and Instance Complexities"
- ACT: Graves, "Adaptive Computation Time for Recurrent Neural Networks"
- CALM: Schuster et al., "Confident Adaptive Language Modeling"
- Calibration: Guo et al., "On Calibration of Modern Neural Networks"
- CKA: Kornblith et al., "Similarity of Neural Network Representations Revisited"

---

## Document Metadata

- **Created:** [Current Date]
- **Status:** Hypothesis Locked, Awaiting Phase 1 Experiments
- **Decision Point:** After Experiment 3
- **Target Venue:** Interspeech / ICASSP (speech) or EMNLP (NLP)

---

*Upload this document in future conversations to resume context.*
