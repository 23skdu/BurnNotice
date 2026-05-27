# BurnNotice: Technical Implementation Roadmap

> **Status**: Draft
> **Date**: 2026-05-18
> **Version**: 0.2

---

## P0 Blockers: Production-Ready Enhancements

The following features represent high-priority (P0) engineering requirements for production-readiness, hardening the real-time latent governance engine against adversarial attacks, performance degradation, and model drift.

### P0.0 Hardware-Accelerated SIMD Vector Math
- [ ] **P0.0.1** Replace manual `for` loops computing vector norms in `internal/scoring/scorer.go` (`vecNorm`) and `internal/engine/moe.go` with hardware-optimized `floats.Norm(..., 2)`.
- [ ] **P0.0.2** Replace manual `for` loops computing dot products in `internal/scoring/scorer.go` (`vecDot`) with `floats.Dot()`.
- [ ] **P0.0.3** Refactor `ApplyLoRA` and `ReverseLoRA` matrix multiplications in `internal/steer/lora.go` to use AVX/AVX-512 Fused-Multiply-Add via `floats.AddScaled()`.
- [ ] **P0.0.4** Benchmark the hotpaths using `go test -bench` to ensure `gonum/floats` assembly routines provide the expected SIMD latency reductions.

### P0.1 Adaptive, Model-Specific Threshold Calibration

- [x] **P0.1.1** Implement an automated calibration pipeline that ingests a held-out contrastive validation dataset and performs ROC curve analysis to select optimal detection thresholds.
- [x] **P0.1.2** Develop an optimization routine to automatically balance precision and recall, setting model-specific thresholds that maximize the F1-score for deception and safety metrics.
- [x] **P0.1.3** Implement dynamic threshold updating inside `internal/policy/engine.go` to adapt active policies based on real-time baseline activation drifts.
- [x] **P0.1.4** Write comprehensive unit and fuzz tests simulating extreme validation datasets (e.g. all-positive, all-negative, high-variance scores) and verifying calibration safety invariants.
- [x] **P0.1.5** **UI/UX Update**: Add a calibration section to the model profile dashboard showing ROC curves, precision-recall graphs, and a single-click "Recalibrate" trigger.
- [x] **P0.1.6** **e2e Update**: Create an end-to-end integration test running a full model profile dataset pipeline, verifying calibration values are written to `profile.gob` and immediately loaded by the policy engine.

### P0.2 Adversarial Robustness & Defense-in-Depth

- [x] **P0.2.1** Integrate an input-level sanitization layer inside the agent SDK to filter out token patterns typical of activation-space adversarial suffix attacks.
- [x] **P0.2.2** Implement gradient-masking strategies at the latent layer during active steering, preventing attackers from using white-box gradients to construct jailbreak suffixes.
- [x] **P0.2.3** Add a low-amplitude noise injection layer (random Gaussian perturbations) to the extracted activations prior to scoring to defend against high-precision vector projection attacks.
- [x] **P0.2.4** Write fuzz tests injecting adversarial and high-frequency noise inputs to the scorer, ensuring similarity metrics and classification labels remain stable.
- [x] **P0.2.5** **UI/UX Update**: Add an "Adversarial Shield" panel on the dashboard displaying the live adversarial perturbation rates, blocked jailbreak attempts, and dynamic gradient masking metrics.
- [x] **P0.2.6** **e2e Update**: Run end-to-end safety evaluation benchmarks under simulated adversarial attacks, verifying the agent remains fully governed even under direct projection steering manipulation.

### P0.3 Multi-Format Quantization & Direction Transfer

- [x] **P0.3.1** Implement quantization loaders for Turboquant, SQ8, and PQ formats inside `internal/engine/onnx.go`, enabling low-precision execution of models.
- [x] **P0.3.2** Design a mathematical scale-factor calibration mechanism to align activation magnitude variations between FP16 and INT8/FP8 model versions.
- [x] **P0.3.3** Implement direction vector transfer validation: ensure deception and alignment steering vectors extracted from an FP16 baseline map onto the quantized model layers without loss of cosine similarity.
- [x] **P0.3.4** Write unit and fuzz tests verifying the mathematical projection vectors and steering effectiveness under various quantization widths (FP16, FP8, INT8).
- [x] **P0.3.5** **UI/UX Update**: Add a "Quantization Compatibility" chart in the dashboard profiling layer weight and activation scaling metrics under different formats.
- [x] **P0.3.6** **e2e Update**: Write a complete end-to-end test converting a model to INT8, transferring baseline direction vectors, and executing steering to verify identical behavioral outputs.

### P0.4 Multi-Direction Subspace Extraction

- [x] **P0.4.1** Implement multi-direction subspace extraction as the default inside `internal/extract/pca.go` using top-k right singular SVD vectors instead of single-direction vectors.
- [x] **P0.4.2** Integrate Iterative Nullspace Projection (INLP) to find multiple orthogonal safety directions, capturing complex multi-dimensional alignment concepts.
- [x] **P0.4.3** Update `internal/scoring/scorer.go` to perform subspace projections (using `ScoreSubspace`) as the primary metric, evaluating overall projection energy.
- [x] **P0.4.4** Write robust unit tests verifying SVD orthogonal bases recover complex synthetic high-dimensional subspaces, and INLP successfully projects out deception vectors.
- [x] **P0.4.5** **UI/UX Update**: Overhaul the dashboard visualization to render a 3D scatter plot of the multi-dimensional activation projections using uPlot.
- [x] **P0.4.6** **e2e Update**: Build an end-to-end integration test verifying that multi-direction steering successfully suppresses deception even when the model attempts evasive representational shifts.

### P0.5 Temporal Deception Sequence Scoring

- [x] **P0.5.1** Implement a temporal scoring pipeline that computes exponential moving averages (EMA) of metric scores across the last $N$ tokens to track sequence-level deception signals.
- [x] **P0.5.2** Integrate sliding window sequence monitoring to identify slow-drift deception patterns that do not trigger single-token safety thresholds.
- [x] **P0.5.3** Update the policy engine to support sequence-level temporal operators (e.g. cumulative safety thresholds or sustained violations over time).
- [x] **P0.5.4** Write unit and fuzz tests simulating variable token sequence speeds, verifying temporal decay and activation energy aggregation formulas remain numerically stable.
- [x] **P0.5.5** **UI/UX Update**: Add a real-time temporal sequence chart to the web monitor showing the cumulative deception trend line and sliding threshold envelopes.
- [x] **P0.5.6** **e2e Update**: Implement an end-to-end test feeding a sequence of increasingly deceptive prompt tokens, verifying the temporal policy triggers preemptive intervention prior to completion.

### P0.6 Capability Preservation & Steering Side-Effect Benchmarks

- [x] **P0.6.1** Integrate an automated evaluation harness in the steering tests that measures the model's core intelligence metrics (e.g. logic, reasoning, code generation) under dynamic steering.
- [x] **P0.6.2** Compute and report capability degradation scores comparing baseline MMLU, GSM8k, and HumanEval performance against steered states.
- [x] **P0.6.3** Implement safe steering clamping in `internal/steer/selective.go` to prevent excessive activation distortion and preserve model performance.
- [x] **P0.6.4** Write unit tests checking that clamping and norm-preservation limits are strictly enforced, rejecting steering scaling coefficients that trigger high distortion.
- [x] **P0.6.5** **UI/UX Update**: Add a "Capability Health" gauge on the dashboard, displaying the live MMLU/GSM8k score retention and layer-wise distortion statistics.
- [x] **P0.6.6** **e2e Update**: Create a full end-to-end benchmark test running standard reasoning tasks under maximum steering strength, asserting capability preservation remains above 98%.

### P0.7 Asynchronous Non-Blocking Scoring Engine

- [x] **P0.7.1** Decouple ONNX/WASM inference from the scoring and policy evaluation pipeline using asynchronous Go channels and highly efficient ring buffers.
- [x] **P0.7.2** Implement a zero-copy activation queue to hand off intermediate tensors to a background worker pool, preventing inference pipeline blocking.
- [x] **P0.7.3** Integrate emergency sync bypass triggers: if queue saturation occurs or latency thresholds are breached, temporarily switch back to synchronous blocking evaluation for high-risk prompts.
- [x] **P0.7.4** Write exhaustive concurrency and race tests running high-throughput inference under parallel scoring loads, verifying zero race conditions or memory leaks.
- [x] **P0.7.5** **UI/UX Update**: Add a system latency panel on the dashboard showing the exact queue depth, asynchronous worker pool saturation, and engine processing delay metrics.
- [x] **P0.7.6** **e2e Update**: Run an end-to-end throughput benchmark under high parallel agent requests, asserting scoring pipeline introduces zero overhead to core token generation latency.

---

## Improvement Plan: Addressing Architectural, Performance, and Adversarial Flaws

This section details the concrete, actionable roadmap to address the systemic flaws identified in `docs/analysis.md`, `docs/analysis2.md`, `docs/analysis3.md`, and `docs/comparison.md`.

### Milestone 1: GPU-Native Inference & Enterprise Engine Pivot (Resolves Latency Bottleneck & ONNX Limitations)

- **Problem**: Extracting intermediate activations from GPU memory to CPU Host memory at every token/layer introduces unacceptable latency and breaks continuous batching optimizations. ONNX Runtime lacks the memory scheduling capabilities (like PagedAttention) of modern production serving stacks.

- [x] **M1.1** Implement direct integration plugins for high-performance serving backends like **vLLM** and **TensorRT-LLM**.
- [x] **M1.2** Move intermediate activation vector operations (e.g., cosine similarity and orthogonal projections) directly into custom GPU/CUDA tensor kernels.
- [x] **M1.3** Retain all per-token critical calculations inside GPU VRAM, reserving the CPU Go/Longbow layer exclusively for asynchronous reporting, policy updates, and ledger tracking.
- [x] **M1.4** Support advanced key-value (KV) cache hook storage to capture multi-turn contextual activation trajectories efficiently without redundant computations.
- [x] **M1.5** Write custom **Metal Shading Language (MSL)** kernels and Objective-C dispatch bindings to accelerate activation metric scoring directly on Apple Silicon GPUs (darwin/arm64).
- [x] **M1.6** Develop high-performance **C++/CUDA** kernels executing parallel activation projections and dynamic rotations directly on NVIDIA V100/A100/H100 Tensor Cores (linux/amd64).
- [x] **M1.7** Construct PyTorch/C++ custom extension wrappers linking the native Metal and CUDA kernels to enable inline, zero-latency activation steering inside the **vLLM** execution graph.

### Milestone 2: Context-Aware Semantic Intent Gating (Resolves Semantic Ambiguity & False Positives)

- **Problem**: Deception is highly contextual; static linear directions will trigger false positives when the agent simulates fictional characters, discusses deception academically, or performs roleplay, leading to unwarranted capability degradation.

- [x] **M2.1** Integrate a lightweight, fast text-based semantic pre-classifier (e.g., Llama Guard or specialized distillation models) running in parallel with prompt tokenization.
- [x] **M2.2** Implement dynamic gating logic: disable specific steering vectors (such as `deception_direction`) if the user prompt is classified under creative writing, academic analysis, or safe fictional roleplay.
- [x] **M2.3** Formulate class-specific confidence weights that scale the active intervention metrics dynamically based on conversational context.

### Milestone 3: Obfuscation-Resilient Probe Training (Resolves Bailey et al. Evasion Vulnerability)

- **Problem**: Advanced models or malicious prompt crafters can generate activation-space obfuscation perturbations (e.g., Bailey et al. ICLR 2026) that bypass static linear probes entirely while maintaining harmful outputs.

- [x] **M3.1** Upgrade direction extraction from static 1D linear probes to non-linear Multi-Layer Perceptrons (MLPs) and multi-directional ensemble networks that capture complex conceptual manifolds.
- [x] **M3.2** Introduce adversarial training during the profiling phase: perturb activation states using gradient-based noise optimization to train the detection probes to be robust against activation-space evasion vectors.
- [x] **M3.3** Implement real-time anomaly detection inside the residual stream to flag suspicious, out-of-distribution (OOD) activation structures that suggest intentional safety obfuscation.

### Milestone 4: Decentralized Standalone Fallback Engine (Resolves Longbow Pre-1.0 Dependency Risk)

- **Problem**: Longbow is a pre-1.0 rolling release, posing a significant dependency risk. If a Longbow node or distributed mesh goes offline, the scoring engine collapses.

- [x] **M4.1** Implement a robust local-storage backup pipeline using the existing standalone SQLite ledger for profile metadata and calibration thresholds.
- [x] **M4.2** Integrate an optimized Gonum-based local math backend for cosine and dot-product calculations that serves as a seamless fallback if the Longbow service is unreachable.
- [x] **M4.3** Establish automatic, zero-downtime hot-swapping between Longbow Flight connections and local math fallbacks.

### Milestone 5: Closed-Loop Coherence & Dynamic Steering Scale (Resolves Capability Loss & Lobotomy Effect)

- **Problem**: Actively injecting or rotating high-dimensional vectors (ActAdd/Angular Steering) shifts intermediate states out of training distribution, causing grammatical collapse, loss of logical coherence, and severe capability loss.

- [x] **M5.1** Implement real-time, token-level perplexity tracking and output coherence checkers directly inside the model generation loop to monitor early signs of activation collapse.
- [x] **M5.2** Develop an adaptive feedback loop that automatically scales down steering strength ($\alpha$) or dynamically shifts selected layers if logical degradation is detected in consecutive output tokens.
- [x] **M5.3** Conduct automated regression tests against standardized capability suites (GSM8k, MMLU) during live steering interventions to optimize individual layer steering bounds.
