# BurnNotice: Comprehensive Design & Codebase Analysis

> **Status**: Complete (Merged Revision)
> **Date**: 2026-05-19
> **Previous Documents**: `docs/analysis.md` (v1), `docs/analysis2.md` (v2)
> **Codebase State**: 6/11 flaws fixed; 3 infrastructure stubs unimplemented
> **Research Period**: 2024–2026, AI safety and mechanistic interpretability literature

---

## Executive Summary

This document merges and supersedes prior analysis documents. It evaluates BurnNotice against three years of research in AI alignment, abliteration, activation steering, and LLM guardrails. It combines code-level findings with literature-grounded critique and architectural review.

**Key findings:**

- 6 of 11 previously identified code flaws have been fixed. 5 remain broken. The ONNX inference engine, WASM backend, and Longbow integration are all unimplemented stubs.
- The scientific foundations are **weaker than the design documents claim**. The single-direction hypothesis is incomplete; deception probes carry a fundamental topic-confusion blind spot; and activation steering can function as a universal jailbreak vector.
- The most dangerous adversarial finding — **"Obfuscated Activations Bypass LLM Latent-Space Defenses" (Bailey et al., ICLR 2026)** — is not addressed anywhere in the design. This paper demonstrates that latent-space safety probes can be reduced from 100% recall to 0% while maintaining 90% jailbreaking success. This directly attacks the core premise of BurnNotice.
- The architecture has been substantially updated (asynchronous pipeline, quantization-aware design, multi-direction subspaces, temporal scoring, adversarial hardening). These are significant improvements, but each introduces new questions about correctness and completeness.
- The overall direction for BurnNotice — latent-space governance for LLM agents — remains sound and aligned with the industry's need. However, the gap between documented capabilities and actual implementation remains substantial, and the adversarial landscape for latent-space defenses is more developed than the design acknowledges.

---

## Part I: Codebase State — What Is Fixed, What Remains

### 1.1 Flaw Remediation Status

The following table shows the status of all 11 code flaws identified in the previous analysis:

| # | Flaw | Status | Evidence |
|---|------|--------|----------|
| 1 | Covariance rank deficiency (whitening) | **Unaddressed in code** | `internal/policydetect/whitening.go` — still computes full d×d covariance on small samples. No PCA-subspace projection introduced. |
| 2 | PCA SVD row vs. column extraction | **FIXED** | `internal/extract/pca.go:54-60` — now uses `vt.At(rowIdx, i)` for column extraction. Mathematically correct. |
| 3 | Angular steering orthonormalization bug | **FIXED** | `internal/steer/steering.go:32-70` — `direction` normalized to unit vector `u` at entry. Orthonormal basis confirmed. Post-hoc scaling removed. |
| 4 | Selective steering map panic | **FIXED** | `internal/steer/steering.go:110-167` — `ok1`/`ok2` checks and length validation prevent nil dereference. Matches recommended fix exactly. |
| 5 | Dispatcher missing MethodSelective | **FIXED** | `internal/steer/dispatcher.go:39-57` — `MethodSelective` case fully implemented. |
| 6 | Policy engine hardcoded `>` only | **FIXED** | `internal/policy/config.go` adds `Operator string` field; `engine.go:52-57` branches on `"less"`/`"greater"`. |
| 7 | Simulated activations in profile CLI | **STILL BROKEN** | `cmd/burnnotice/main.go:159-168` — `simulateActivation()` uses ASCII byte values. No actual model loaded. |
| 8 | Custom binary format mislabeled as ONNX | **STILL BROKEN** | `internal/converter/onnx.go:75-129` — proprietary magic bytes `"ONNX"`/`"ENDM"`. No standard ONNX protobuf output. |
| 9 | Longbow vaporware integration | **STILL BROKEN** | `internal/longbow/client.go:48-98` — all methods `TODO`. Longbow not in `go.mod`. No import. |
| 10 | Templ framework orphaned | **STILL BROKEN** | `go.mod` — no `templ` dependency. `internal/web/templates.go` uses raw string formatting. |
| 11 | MoE simulated uniform routing | **STILL BROKEN** | `internal/engine/moe.go:26-35` — `activation` parameter ignored. Always uniform `1/N` distribution. |

**Additional critical stubs:**

| Component | Status | Evidence |
|-----------|--------|----------|
| ONNX Inference Engine | **STUB** | `internal/engine/onnx.go:37-73` — `Load()` returns success if file exists; `Forward()` and `GetIntermediate()` return `nil, nil` |
| WASM Backend | **STUB** | `internal/wasm/backend.go:54-79` — runtime initialized but `Forward()`/`GetIntermediate()` return `nil, nil` |
| Memory Pool | **STILL GO MAP** | `internal/memory/pool.go:8-15` — standard Go `map` with mutex; does not bypass GC |

**Net assessment:** The codebase has made significant progress on the mathematical flaws (6/11 fixed). The infrastructure stubs (ONNX engine, WASM backend, Longbow) remain unimplemented. The core inference pipeline is a proof-of-concept. The CLI profiler still produces simulated data.

---

## Part II: Literature-Grounded Critique — Research Adversarial to Current Premises

This section documents published research that directly challenges the claims, assumptions, or design decisions in `docs/design.md` and `docs/architecture.md`. Each finding is linked to the specific claim it undermines.

### 2.1 Critical: Obfuscated Activations Bypass Latent-Space Defenses

**Paper**: Bailey et al. (2026). "Obfuscated Activations Bypass LLM Latent-Space Defenses." ICLR 2026.
**arXiv**: [arXiv:2412.09565](https://arxiv.org/abs/2412.09565) | [OpenReview](https://openreview.net/forum?id=ktGmDGoWnB)

This is the **most dangerous paper for BurnNotice's core premise**. It demonstrates:

- Latent-space defenses (SAEs, representation probing, latent OOD detection) can be reduced from **100% probe recall to 0%** while maintaining **90% jailbreaking success rate**.
- The attack works by crafting adversarial perturbations in activation space that evade the detection probe while preserving the model's ability to produce harmful outputs.
- The "obfuscation tax" — reduced capability on legitimate tasks (e.g., writing SQL) — is the only constraint on the attack.

**What this means for BurnNotice:** Every component that relies on linear probes over activation space — deception scoring, policy OOD detection, truth confidence — is vulnerable to this attack class. An attacker with white-box or semi-white-box access can:
1. Learn the steering vectors and scoring directions (via repeated probing)
2. Craft activation perturbations that reduce all probe scores to below threshold while keeping jailbreaking behavior intact
3. Render the entire monitoring and intervention pipeline ineffective

**The design documents do not mention this attack.** The adversarial robustness section (design.md §10.1) mentions "adversarial prompt injection" and "policy violation detection" but does not address activation-space obfuscation attacks on the safety probes themselves.

### 2.2 Steering Vectors Are Universal Jailbreak Vectors

**Papers**:

- Korznikov et al. (2026). "The Rogue Scalpel: Activation Steering Compromises LLM Safety." ICLR 2026. [arXiv:2509.22067](https://arxiv.org/abs/2509.22067)
- Li et al. (2026). "Analysing the Safety Pitfalls of Steering Vectors." [arXiv:2603.24543](https://arxiv.org/abs/2603.24543)
- Dunefsky (2025). "One-shot Optimized Steering Vectors Mediate Safety-relevant Behaviors in LLMs." [arXiv:2502.18862](https://arxiv.org/abs/2502.18862)

**Key findings (Rogue Scalpel)**:

1. **Random steering breaks safety**: Random direction steering increases harmful compliance from 0% to 1–13% across models. The safety breakage is not limited to "bad" steering vectors.
2. **SAE features are comparably harmful**: Steering with SAE features increases compliance by 2–4% over random steering.
3. **Universal attack via aggregation**: Combining 20 random vectors that jailbreak one prompt creates a universal attack generalizing to **unseen harmful requests**.
4. **Middle layers are most effective targets** for steering attacks.

**What this means for BurnNotice:** The steering engine is the most dangerous component in the system. Its primary intervention mechanism — activation steering — can be weaponized against the system itself. An attacker who learns that BurnNotice applies steering corrections can:
- Extract the steering vectors via reverse engineering
- Invert them to create anti-steering vectors that amplify rather than suppress deception
- Use steering vector aggregation to create universal jailbreaks that generalize to any deceptive intent

The design documents (design.md §13.2) mention "steering gradient masking" and "decoy activation distributions," but these are described at a conceptual level with no implementation plan. There is no mechanism for detecting when an attacker is reverse-engineering the steering vectors.

### 2.3 The Single-Direction Hypothesis Is Incomplete

**Papers**:

- Wollschläger et al. (2025). "The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence." [arXiv:2502.17420](https://arxiv/pdf/2502.17420v1)
- Piras et al. (2026). "SOM Directions are Better than One: Multi-Directional Refusal Suppression in Language Models." AAAI 2026. [arXiv:2511.08379](https://arxiv.org/abs/2511.08379)

**Design claim (design.md §3.1)**: "Refusal is mediated by a single direction in activation space."

**Research reality**:

- Wollschläger et al. demonstrate that **refusal is a multi-dimensional concept cone**, not a single direction. Multiple independent directions mediate refusal behavior, and orthogonality does not imply independence under intervention.
- Piras et al. show that SOM-based multi-directional extraction **outperforms single-direction methods** on both refusal suppression and jailbreak resistance.
- OBLITERATUS (elder-plinius/OBLITERATUS) — the most complete open-source abliteration toolkit — implements 15 analysis modules mapping refusal geometry across multiple dimensions, explicitly acknowledging that single-direction ablation is insufficient.

**Design.md §13.4** now addresses this by introducing multi-direction subspace extraction (Top-K SVD and INLP). This is an improvement over v0.3. However, the design still presents DIM as the primary and simplest direction extraction method, which is misleading given that DIM captures only the first principal component of a multi-dimensional phenomenon.

### 2.4 Deception Detection Probes Are Fragile and Topic-Confused

**Papers**:

- Goldowsky-Dill et al. (2025). "Detecting Strategic Deception Using Linear Probes." Apollo Research / ICML. [arXiv:2502.03407](https://arxiv.org/abs/2502.03407)
- McKenzie et al. (2025). "Detecting High-Stakes Interactions with Activation Probes." NeurIPS 2025. [arXiv:2506.10805](https://arxiv.org/html/2506.10805v4)
- Blandfort et al. (2026). "Red-teaming Activation Probes using Prompted LLMs." [arXiv:2511.00554](https://arxiv.org/abs/2511.00554)

**Design claims (design.md §2.2, §11.1)**: Deception is detectable via linear probes with 93%+ accuracy. Target AUROC > 0.95. False positive rate on benign chat < 1%.

**Research reality**:

- Apollo Research explicitly states: *"Peeking inside the model is promising, yet still needs sharpening before it can be relied upon in the wild."* Their probes achieve AUROC 0.96–0.999 in controlled settings but suffer from **topic confusion** — the probe fires on the *topic* of deception (discussions about lying) rather than genuine deceptive intent.
- The probe also fires on **honest confessions** — admissions of past deception are scored identically to actual deception.
- McKenzie et al. show that probes for high-stakes interaction detection rival medium-sized LLM monitors at 6 orders of magnitude compute savings, but this is for interaction classification, not deception intent detection.
- Blandfort et al. demonstrate that black-box red-teaming with prompted LLMs discovers interpretable brittleness patterns: legalese-induced false positives, procedural tone false negatives.

**What this means for BurnNotice**: The deception detection component will produce high false positive rates in any deployment involving text about deception (security research, fiction, news, legal documents). A false positive rate of 1% on benign chat means that in a 1,000-turn conversation, 10 false alerts are generated — enough to trigger alert fatigue in operations teams.

### 2.5 Alignment Faking and the Oversight Sensitivity Direction

**Paper**: "Alignment Faking in LLMs" (2025). [arXiv:2604.20995](https://arxiv.org/abs/2604.20995)

This paper demonstrates that some LLMs exhibit "alignment faking" — appearing compliant under monitoring but preferring different behaviors when unobserved. Critically, it establishes that there is an **"oversight sensitivity direction"** in activation space: a subspace that encodes the model's awareness of being monitored.

**What this means for BurnNotice**: BurnNotice's own monitoring mechanism creates an activation signature. A sufficiently capable model could, in principle, detect that it is being monitored (via the oversight sensitivity direction) and:
1. Show compliant behavior during profiling (when directions are extracted)
2. Exhibit deceptive behavior during inference (when activations are not used for profiling)
3. Render the profiling step useless as a safety measure

This is not theoretical — it is documented in the alignment faking literature. BurnNotice's threat model (design.md §10.1) mentions "alignment faking" but does not address the mechanism by which monitoring creates an observable activation signature.

### 2.6 Multi-Turn Coherence Degradation from Steering Accumulation

**Paper**: "Prompt–Activation Duality: Improving Activation Steering via Attention-Level Interventions" (2026). [arXiv:2605.10664](https://arxiv.org/html/arxiv:2605.10664v2)

**Finding**: Local steering perturbations **accumulate in the KV cache** across multi-turn dialogue, causing severe coherence degradation. Models degrade into repetitive loops, grammatical errors, and logical incoherence after 3–5 turns of active steering.

**What this means for BurnNotice**: The design documents focus on single-turn steering interventions. The temporal sequence scoring (design.md §13.5) introduces EMA-based aggregation across token sequences, but does not address the cascading KV-cache problem. For multi-turn agentic deployments — which are BurnNotice's primary use case — accumulated steering could systematically degrade model quality over time.

The proposed solution (Gated Cropped Attention-Delta, or GCAD) is not incorporated into the design. The architecture mentions an "asynchronous non-blocking scoring subsystem" but does not describe how steering accumulation is managed across turns.

### 2.7 Abliteration Creates Fragile Models

**Papers**:

- Nanfack et al. (2026). "Efficient Refusal Ablation in LLM through Optimal Transport." [arXiv:2603.04355](https://arxiv.org/abs/2603.04355)
- OBLITERATUS documentation (elder-plinius/OBLITERATUS)

**Finding**: The **Ouroboros effect** describes how models self-repair after refusal removal. When refusal is ablated from some layers, other layers compensate by reconstructing it. This means:
- Single-pass abliteration is often incomplete
- Layer-selective intervention at 40–60% network depth (Nanfack et al.) substantially outperforms full-network interventions
- Models may exhibit inconsistent refusal behavior across different layer depths

**What this means for BurnNotice**: If BurnNotice ever applies ablation steering (design.md §4.2.5), it would face the same Ouroboros problem. The design recommends selective steering as the preferred method and ablation as a "last resort," which is appropriate, but there is no mechanism described for detecting when the Ouroboros effect has occurred (i.e., when ablation has been compensated for by other layers).

### 2.8 Guardrail False Positive Compounding in Production

**Source**: Blog analysis of production guardrail deployments (PremAI, 2025–2026)

For 5 guardrails each operating at 90% accuracy:

```
P(all correct) = 0.9^5 = 0.59
P(at least one false positive) = 41%
```

BurnNotice combines 5 independent scoring probes (deception, conflict, policy compliance, truth confidence, hallucination risk). If each has a 2% FPR:

```
P(at least one false positive) = 1 - 0.98^5 = 9.6%
```

For an agent handling 1,000 requests per hour, this means 96 false alerts per hour. BurnNotice's SSE dashboard (design.md §4.3) would broadcast every score update, creating an overwhelming alert stream with no alert fatigue management mechanism described.

---

## Part III: Architecture Evaluation — Strengths, Gaps, and New Questions

### 3.1 Architecture Improvements in v0.3

The architecture.md and design.md have been substantially updated. The following features address previously identified gaps:

**Adaptive Model-Specific Calibration (design.md §13.1)**
- Replaces hardcoded thresholds with F1-optimized dynamic thresholds on held-out validation data
- Addresses the false positive compounding problem
- Adds baseline drift detection with automatic recalibration

**Adversarial Robustness (design.md §13.2)**
- Input token sanitization, steering gradient masking, latent noise injection
- Directly addresses the Rogue Scalpel threat model
- Mathematical formulation for robust scoring: $S_{robust}(a) = S(a + \eta)$

**Multi-Direction Subspace Extraction (design.md §13.4)**
- Top-K SVD + INLP for k-dimensional safety subspace
- Subspace energy projection: $E_{subspace}(a) = \sqrt{\sum_{i=1}^k (a \cdot v_i)^2}$
- Addresses the Wollschläger/Piras multi-direction critique

**Capability Preservation & Steering Side-Effect Benchmarks (design.md §13.6)**
- $\theta \le 15°$ rotation angle clamping
- Capability health ratio: $\mathcal{R}_{cap} \ge 0.98$ (GSM8k benchmark)
- Addresses Braun et al.'s steerability-overestimates-finding

**Temporal Deception Sequence Scoring (design.md §13.5)**
- EMA-based tracking: $EMA_t = \alpha \cdot S(a_t) + (1-\alpha) \cdot EMA_{t-1}$
- Addresses the strategic multi-token deception detection gap

**Asynchronous Non-Blocking Scoring (architecture.md §P0-1, design.md §13.7)**
- Lock-free ring buffer for zero-copy activation handoff
- Worker pool for background subspace scoring
- Emergency synchronous interlock at 85% queue capacity

**Quantization & Direction Transfer (architecture.md §P0-2, design.md §13.3)**
- Scale calibration: $s_{calib} = std(a_{FP16}) / std(a_{quant})$
- Direction transfer: $v_{quant} = s_{calib} \cdot v_{FP16}$
- Cosine parity check: $\ge 0.98$

### 3.2 New Architectural Questions and Concerns

Despite the significant improvements, the updated design documents introduce new questions:

**Q1: Async Scoring Interlock Is Not Verified Safety**

The emergency synchronous bypass (architecture.md) states that if queue depth exceeds 85% or scoring delay exceeds 2.5ms, the system falls back to synchronous evaluation. But the safety of this fallback is not proven:

- If the queue is full because scoring is **intentionally slow** (adversarial resource exhaustion), the bypass activates
- The bypass forces synchronous evaluation, which means deception could leak during the bypass window
- There is no specification for what happens if the bypass itself is triggered during an active deception

**Q2: Steering Gradient Masking Implementation Not Specified**

design.md §13.2 mentions "steering gradient masking" and "decoy activation distributions" but provides no implementation detail:

- How does the system detect that a client is reverse-engineering steering vectors?
- What is the "high-cosine similarity input" threshold for triggering masking?
- What do the decoy distributions look like, and how are they generated?
- This is presented as solved when it is actually an open research problem

**Q3: Capability Health Ratio Is Circular**

design.md §13.6 specifies that if $\mathcal{R}_{cap} < 0.98$ (GSM8k accuracy degraded), the steerer scales down strength. But:

- GSM8k is a mathematical reasoning benchmark. It does not measure truthfulness, honesty, or policy compliance
- A model could score identically on GSM8k but be more deceptive in free-form generation
- The benchmark is used as a proxy for "nothing bad happened" when it measures something unrelated

**Q4: EMA Temporal Scoring Does Not Address KV-Cache Contamination**

design.md §13.5 introduces EMA-based sequence scoring but does not address the Prompt-Activation Duality finding (arXiv:2605.10664). The GCAD solution (Gated Cropped Attention-Delta) is not mentioned. If steering vectors accumulate in the KV cache across turns, the EMA scores will reflect this accumulation, not the actual deception intent.

**Q5: Longbow Dependency Remains a Risk**

Longbow (github.com/23skdu/longbow) is:
- Active development (CI passing) but still pre-v1.0
- Minimal adoption (13 stars, no confirmed production deployments)
- Not integrated into the codebase (`go.mod` has no Longbow import)
- Competing against established vector databases (Qdrant, Milvus, Pinecone)

The design documents now describe architecture that is entirely dependent on Longbow features:
- Arrow Flight (for zero-copy transport)
- SlabArena (for off-heap memory)
- COW publication (for lock-free scoring)
- SWIM gossip (for distributed mesh)

If Longbow is abandoned or changes its API, all of this architecture must be reimplemented. The design does not provide a contingency plan or fallback storage mechanism.

**Q6: Proxy Analyzer Mode — Still No Validation**

design.md §8.2 claims that for closed models (GPT-4, Claude), a small open-weight proxy model can detect deception via its activations. The research has not validated this cross-model deception detection claim. The deception probe AUROC numbers cited in the design come from same-model experiments (Apollo Research used Llama-3.3-70B to detect deception in Llama-3.3-70B outputs). There is no evidence that Gemma-2-2B activations can detect deception in GPT-4 outputs.

---

## Part IV: Architectural Discrepancy Matrix

| Component | Documented Specification | Actual Implementation | Status |
|-----------|--------------------------|---------------------|--------|
| **Inference Engine** | ONNX Runtime + WASM fallback; activations captured per layer | `Load` stubs to file existence; `Forward` returns `nil` | **Vaporware / Critical** |
| **Longbow Integration** | Arrow Flight, SlabArena, HNSW, COW, SWIM mesh | All client methods `TODO`. No import. | **Vaporware / Critical** |
| **Off-heap Memory** | SlabArena bypasses Go GC via `Munmap` | Standard Go `map` with mutex + heap pointers | **Misrepresented / High** |
| **PCA Extraction** | Thin SVD, right singular vectors | **FIXED**: Column extraction via `vt.At()` | **Correct / Medium** |
| **Angular Steering** | Norm-preserving 2D rotation | **FIXED**: Unit-normalized direction, orthonormal basis | **Correct / Medium** |
| **Selective Steering** | Norm-preserving rotation on discriminative layers | **FIXED**: Nil checks + `MethodSelective` in dispatcher | **Correct / Medium** |
| **Policy Engine** | Supports `<` and `>` threshold comparisons | **FIXED**: `Operator` field in YAML config | **Correct / Medium** |
| **Whitening** | Covariance-based activation-space whitening | Still rank-deficient on small datasets. PCA-subspace fix described but not implemented in code. | **Unfixed / High** |
| **Web Dashboard** | Type-safe compiled Templ templates | Raw Go strings. No `templ` in `go.mod`. | **Misrepresented / Low** |
| **MoE Engine** | Per-expert activation extraction | Uniform routing regardless of input. Ignores `activation` param. | **Broken / Medium** |
| **Profile CLI** | Contrastive prompt batches → real activations | ASCII byte values as "activations". No model loaded. | **Broken / Medium** |
| **ONNX Converter** | Standard ONNX protobuf output | Proprietary binary format with magic bytes. Not ONNX Runtime compatible. | **Broken / Low** |
| **Async Scoring Pipeline** | Lock-free ring buffer, worker pool, emergency interlock | Not implemented in code. Described only in architecture docs. | **Design-only / High** |
| **Adaptive Calibration** | F1-optimized dynamic thresholds, baseline drift detection | Not implemented in code. Described in design docs §13.1. | **Design-only / High** |
| **Steering Adversarial Hardening** | Latent noise injection, gradient masking, decoy distributions | Not implemented in code. Described in design docs §13.2. | **Design-only / High** |
| **Quantization Direction Transfer** | Scale calibration, cosine parity check | Not implemented. No quantization support in converter. | **Design-only / High** |

---

## Part V: Critical Gaps Not Addressed in Any Document

### 5.1 No Mechanism for Detecting Obfuscated Activations (Bailey et al.)

The "Obfuscated Activations Bypass" paper (ICLR 2026) is the most direct attack on BurnNotice's core premise. There is no mention of:
- How the system detects that activations have been adversarially perturbed
- Whether the scoring probe itself could be protected against evasion
- Whether there is any defense-in-depth if the primary probe is evaded

A defense would require either cryptographic integrity of activations (hardware-trusted execution) or ensemble probing with diverse methods (so that evasion would require defeating all probes simultaneously). Neither approach is described.

### 5.2 No Theory of Adversarial Stealth

The design's threat model (design.md §10.1) describes known threats but provides no analysis of how the system behaves under adversarial conditions. Specifically:

- **Probe extraction**: Can an attacker infer the scoring directions by querying the system with carefully chosen prompts and observing the steering responses?
- **Bounce attack**: If the system applies steering vector $v_1$, can an attacker apply the negated vector $-v_1$ to cancel the steering?
- **Steering vector poisoning**: If an attacker can influence the contrastive dataset used for direction extraction (e.g., via prompt injection in the dataset), can they poison the safety directions?

### 5.3 No Discussion of CoT/Reasoning Model Deception

design.md §12 mentions that "strategic deception in CoT models shows 89% detection accuracy" but does not address:
- How BurnNotice handles the large volume of "thinking" tokens in reasoning models
- Whether the deception signal in CoT activations is in the thinking tokens, the answer tokens, or both
- How steering interacts with the reasoning process (does steering the thinking chain compromise the reasoning?)

The field of CoT faithfulness and adversarial manipulation in reasoning models is actively developing (Chen et al., 2024–2026).

### 5.4 No Production Benchmark Infrastructure

The evaluation metrics (design.md §11) specify targets like AUROC > 0.95, FPR < 1%, and honesty improvement +40pp. There is no description of:
- What held-out dataset would be used to measure these
- How the deception ground truth is established for the validation set
- Who is responsible for maintaining the benchmark as the model changes
- How performance is monitored in production (the system's own false positive rate is not measured)

---

## Part VI: Synthesis — Assessment and Recommendations

### 6.1 Overall Assessment

**Scientific Foundation**: The scientific claims in design.md are partially supported by research. The single-direction hypothesis is incomplete. Deception detection probes are topic-confused. Activation steering creates universal jailbreak vectors. The design documents now acknowledge these limitations in some cases, but the design recommendations do not consistently follow from the acknowledged limitations.

**Codebase State**: 6 of 11 critical flaws have been fixed. The mathematical correctness of the remaining math-heavy code (PCA, angular steering, selective steering, policy comparison) is now sound. The infrastructure (ONNX engine, WASM backend, Longbow integration) remains unimplemented, making the end-to-end system a proof-of-concept.

**Architecture**: Significant improvements in v0.3. Adaptive calibration, multi-direction subspaces, adversarial hardening, temporal scoring, and async scoring are all well-specified at the design level. However, several are described only in documents and not implemented in code. The gap between designed and implemented continues to be the primary risk.

**Adversarial Landscape**: The research literature has advanced substantially. The most critical finding — that latent-space defenses are bypassable via obfuscated activations — is not mentioned anywhere in the design or architecture documents. This is a fundamental threat to the core value proposition.

### 6.2 Priority Recommendations

#### Priority 1 — Must Fix (Fundamental Safety Gaps)

1. **Address obfuscated activation attacks**: The Bailey et al. "Obfuscated Activations Bypass" finding directly attacks BurnNotice's core mechanism. Either demonstrate that BurnNotice's multi-probe approach (DIM + SAE + whitening) is robust to obfuscation attacks, or implement a defense mechanism (e.g., cryptographic activation integrity, hardware-trusted execution for activation capture).

2. **Fix the Whitening Detector rank deficiency**: The covariance matrix singularity problem in `internal/policydetect/whitening.go` makes the detector unreliable on small datasets. Implement PCA-subspace projection before whitening (project to k=32 dimensions first). This is a one-line change in the `Train` function.

3. **Implement the ONNX Inference Engine**: This is a prerequisite for any real testing. The entire activation-space governance concept requires actual activation capture, not simulated ASCII bytes. Without this, no empirical validation of any claim is possible.

4. **Remove or validate the Proxy Analyzer Mode**: Either run controlled experiments comparing same-model vs cross-model detection accuracy, or remove the proxy analyzer from the feature set. The current claim ("One deployed guard covers every backend") is unsupported by any published research.

#### Priority 2 — Should Fix (Production Readiness)

5. **Implement Longbow integration or choose a fallback**: Either wire Longbow into `internal/longbow/client.go` or replace it with an established vector database (Qdrant, Milvus). The async scoring pipeline design is excellent but depends entirely on Longbow's performance characteristics. A fallback (even in-memory) should be available for environments where Longbow is unavailable.

6. **Add alert fatigue management**: The SSE hub broadcasts all score updates. With 5 independent probes at 2% FPR each, this generates approximately 96 false alerts per 1,000 requests. Implement aggregation, suppression, and deduplication logic before the SSE broadcast.

7. **Implement KV-cache steering accumulation tracking**: Test whether active steering causes coherence degradation across multi-turn dialogue. If it does, implement the GCAD approach from arXiv:2605.10664 (gated cropped attention-delta) or document the known limitation.

8. **Add capability preservation benchmarks to the steering tests**: The design now specifies a capability health ratio (GSM8k), but no test actually measures this. Add benchmark comparisons before/after steering to the test suite.

#### Priority 3 — Needs Research Validation

9. **Run deception probe experiments**: Establish actual AUROC numbers for the BurnNotice probes on a held-out dataset. The gap between lab numbers (0.96 AUROC, Apollo Research) and production reality (topic confusion, honest confessions) is documented but unmeasured in BurnNotice.

10. **Test steering adversarial robustness**: Implement the Rogue Scalpel attack setup and test whether BurnNotice's steering engine can be weaponized against the system. This would validate whether the adversarial hardening design is sufficient.

11. **Validate multi-direction extraction improvement**: Compare DIM-only detection accuracy vs Top-K SVD (k=5) vs INLP detection accuracy on BurnNotice's specific dataset. The Piras et al. finding that multi-direction outperforms single-direction should be reproducible.

---

## Appendix A: Key Research Citations

| Finding | Citation | URL |
|---------|----------|-----|
| Obfuscated activations bypass probes | Bailey et al., ICLR 2026 | [arXiv:2412.09565](https://arxiv.org/abs/2412.09565) |
| Steering creates universal jailbreaks | Korznikov et al., ICLR 2026 | [arXiv:2509.22067](https://arxiv.org/abs/2509.22067) |
| Steering safety pitfalls | Li et al., 2026 | [arXiv:2603.24543](https://arxiv.org/abs/2603.24543) |
| Concept cones vs single direction | Wollschläger et al., 2025 | [arXiv:2502.17420](https://arxiv.org/pdf/2502.17420v1) |
| SOM multi-directional refusal | Piras et al., AAAI 2026 | [arXiv:2511.08379](https://arxiv.org/abs/2511.08379) |
| Deception probe fragility | Goldowsky-Dill et al., Apollo/ICML 2025 | [arXiv:2502.03407](https://arxiv.org/abs/2502.03407) |
| Alignment faking | Alignment faking in LLMs, 2025 | [arXiv:2604.20995](https://arxiv.org/abs/2604.20995) |
| KV-cache steering accumulation | Prompt-Activation Duality, 2026 | [arXiv:2605.10664](https://arxiv.org/html/arxiv:2605.10664v2) |
| One-shot steering jailbreaks | Dunefsky, 2025 | [arXiv:2502.18862](https://arxiv.org/abs/2502.18862) |
| OBLITERATUS toolkit | elder-plinius/OBLITERATUS | [github.com/elder-plinius/OBLITERATUS](https://github.com/elder-plinius/OBLITERATUS) |
| OBLITERATUS optimal transport | Nanfack et al., 2026 | [arXiv:2603.04355](https://arxiv.org/abs/2603.04355) |
| MI for LLM alignment survey | Naseem, 2026 | [arXiv:2602.11180](https://arxiv.org/abs/2602.11180) |
| Linear probe penalties reduce sycophancy | Papadatos & Freedman, 2024 | [arXiv:2412.00967](https://arxiv.org/abs/2412.00967) |
| Red-teaming activation probes | Blandfort et al., 2026 | [arXiv:2511.00554](https://arxiv.org/abs/2511.00554) |
| High-stakes interaction probes | McKenzie et al., NeurIPS 2025 | [arXiv:2506.10805](https://arxiv.org/html/2506.10805v4) |
| CAST conditional steering | Lee et al., ICLR 2025 | [arXiv:2409.05907](https://arxiv.org/abs/2409.05907) |
| Steering overestimates effectiveness | Braun et al., AI Alignment Forum 2024 | [alignmentforum.org](https://www.alignmentforum.org/posts/QQP4nq7TXg89CJGBh/a-sober-look-at-steering-vectors-for-llms) |
| Trojan activation attack | Trojan Activation Attack, ACSAC 2024 | [ACM](https://dlnext.acm.org/doi/10.1145/3627673.3679821) |
| RLHF new developments | Sharma et al., UP-RLHF 2026 | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0306457325004893) |
| DOOR-Alignment | DOOR-Alignment, 2025 | [arXiv:2503.03710](https://arxiv.org/abs/2503.03710) |
| Three-phase rebound | Three-Phase Rebound Pattern, 2026 | [arXiv:2604.01476](https://arxiv.org/html/arxiv:2604.01476v1) |