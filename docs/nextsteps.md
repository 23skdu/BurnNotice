# BurnNotice: Prioritized Implementation Roadmap

> **Status**: Active
> **Last Updated**: 2026-09-07 (Go 1.27 Upgrade, Dependabot PR Sync & SOTA 10-Part Strategic Plan)
> **Version**: 0.7

---

## 🚀 10-Part Strategic Improvement Plan (v0.7+ Architecture)

Based on comprehensive deep code analysis, empirical performance profiling across the ONNX/CGO runtime, and 2025–2026 state-of-the-art advances in latent representation engineering (RepE, MPOA, SAE-SSV, CKA), the following 10-part strategic improvement plan establishes the next generation of BurnNotice:

### 1. Unified Serving-Engine VRAM Zero-Copy Kernel Interception
- **Architecture**: Move token-level scoring and steering directly into GPU VRAM serving loops via the PyBind11 vLLM and TensorRT-LLM C++ plugins (`plugins/vllm/`, `plugins/tensorrt/`).
- **Goal**: Eliminate PCIe host-device memory transfers for activation capture. The Go engine acts purely as an asynchronous control plane and policy coordinator via ring-buffer telemetry, while CUDA warp-reduction kernels (`plugins/cuda_kernels.cu`) execute steering in-place during the forward pass (< 25µs per layer).

### 2. Dynamic CoT & Autoregressive Thinking-Phase Boundary Gating
- **Architecture**: Implement token-stream state-machine monitoring for reasoning models (e.g. DeepSeek-R1, QwQ) detecting `<think>` / `</think>` structural boundaries and reasoning token protocol delimiters.
- **Goal**: Maintain zero steering intervention during the internal chain-of-thought exploration phase to prevent reasoning degradation, while engaging calibrated MPOA steering dynamically the moment answer emission begins.

### 3. Cross-Layer CKA Distortion Profiling & Targeted Layer Recovery
- **Architecture**: Leverage Centered Kernel Alignment (CKA) and Hilbert-Schmidt Independence Criterion (`internal/analysis/crosslayer.go`) to measure representational deformation across residual streams pre- and post-intervention.
- **Goal**: Automatically identify decision layers where deceptive trajectories emerge and apply surgical steering to only the minimal subset of critical layers, avoiding broad cross-layer representational damage.

### 4. MoE-Aware Dynamic Expert Gating & Router-Weighted Intervention
- **Architecture**: Deepen the Mixture-of-Experts routing hooks (`internal/engine/moe.go`) to extract concept vectors on a per-expert basis rather than globally across the residual stream.
- **Goal**: Apply steering vectors conditioned on router logit softmax probabilities, steering only the specialized experts active for the toxic or deceptive token subspace without altering factual knowledge experts.

### 5. Continuous SVD Subspace Interpolation & Smooth Manifold Steering
- **Architecture**: Replace discrete steering vector selection with continuous geodesic interpolation across concept cone subspaces (`internal/analysis/cone.go`, `internal/extract/pca.go`).
- **Goal**: Enable continuous parameter `DirectionMix` $\in [0, k]$ with Gaussian-weighted kernel blending across multiple basis directions, eliminating sharp behavioral discontinuities and minimizing token-level KL divergence.

### 6. Multi-Turn KV-Cache Steering Accumulation & GCAD Attention Dilation Defense
- **Architecture**: Build a KV-cache accumulation tracker (`internal/steer/`) that monitors cumulative angular drift across conversational turns.
- **Goal**: Prevent multi-turn conversational steering drift from exceeding safe angular bounds ($\theta_{\text{max}} = 15^\circ$) and implement Gated Cropped Attention-Delta (GCAD) to prevent historical steering vectors from corrupting future context attention scores.

### 7. Multi-Agent Deception Topology & Gossip Contamination Graph Analysis
- **Architecture**: Extend the agent telemetry mesh (`internal/agent/`) to track multi-agent communication networks (MCP servers, LangChain, CrewAI).
- **Goal**: Calculate pairwise directional deception cross-correlation between collaborating agents, generating live contamination risk graphs and halting deceptive cascades before multi-agent task completion.

### 8. Standardized Multi-Turn Red-Teaming & Pareto Tradeoff Harness
- **Architecture**: Integrate automated evaluation against standard safety suites (HarmBench 400 across 7 harm categories, TruthfulQA, GSM8k) into the BurnNotice benchmarking pipeline.
- **Goal**: Compute empirical Pareto frontiers comparing token KL divergence versus Attack Success Rate (ASR) across single-turn and 10-turn multi-turn jailbreak attempts, reporting a normalized composite quality score.

### 9. Adversarial Shielding with Stochastic Latent Perturbations & Gradient Masking
- **Architecture**: Build active evasion defenses around linear and non-linear probes (`internal/profile/obfuscation.go`).
- **Goal**: Inject controlled Gaussian noise $\eta \sim \mathcal{N}(0, \sigma^2)$ into residual probe passes and generate decoy gradient distributions to resist black-box gradient-based suffix optimization (GCG) and activation manipulation attacks.

### 10. Autonomous Distributed Flight/SQLite Sharded Edge Mesh Synchronization
- **Architecture**: Expand the standalone fallback vector store (`internal/longbow/client.go`) into a distributed, peer-to-peer sharded synchronization mesh.
- **Goal**: Enable distributed BurnNotice edge nodes to hot-swap between remote Arrow Flight endpoints and local SQLite caches with zero downtime, sharing discovered concept directions via secure zero-knowledge telemetry.

---

## 🚨 P0 Blockers for Next Release (Remediated & Verified)

All P0 blockers identified during deep code analysis have been resolved and verified with automated test coverage:

### P0.1: Fix Longbow Flight Query & Search Stubs (`internal/longbow/client.go`) - **[RESOLVED]**
- **Status**: Implemented automatic fallback mode in `c.Query()` and `c.Search()`. When Flight client is nil, unconfigured, or encounters connection failure, the client activates fallback mode and queries local SQLite / memory vector store seamlessly.
- **Verification**: Verified with unit tests in `internal/longbow/client_test.go` (`TestFlightClient_FallbackModes`).

### P0.2: Connect Async Ledger Polling & Dynamic Routing in MoE Engine (`internal/engine/moe.go`) - **[RESOLVED]**
- **Status**: Wired `e.ledger.PollActivation()` into `PollLedger()` and `DetectRouting()` to ingest live GPU activations. Implemented router logit softmax processing in `DetectRoutingFromLogits()` for top-k expert gating.
- **Verification**: Verified with unit tests in `internal/engine/moe_test.go` (`TestMoEEngine_PollLedger`, `TestMoEEngine_DetectRouting_WithLedger`, `TestMoEEngine_DetectRoutingFromLogits`).

### P0.3: Wire Missing CLI Flags and Subcommands (`cmd/burnnotice/main.go`) - **[RESOLVED]**
- **Status**: Added `burnnotice aggregate [dir]` subcommand to load and summarize anonymous JSON telemetry reports. Added `--cone-analysis` flag to trigger `ConceptConeAnalyzer` and `--rdo` flag to run `RefusalDirectionOptimizer` in `profileCmd`.
- **Verification**: Verified with CLI tests in `cmd/burnnotice/main_test.go` (`TestBurnNotice_AggregateCmd`, `TestBurnNotice_ProfileCmd_ConeAndRDOFlags`).

### P0.4: Create Telemetry Privacy Specification (`TELEMETRY.md`) - **[RESOLVED]**
- **Status**: Authored `TELEMETRY.md` at repository root detailing the telemetry payload schema, zero-knowledge privacy guarantees (no prompts, outputs, IPs, or identities collected), submission options, and local aggregation CLI.
- **Verification**: Documentation verified against `internal/telemetry/reporter.go`.

### P0.5: Implement Token-Level Logit KL-Divergence (`internal/analysis/kl.go`) - **[RESOLVED]**
- **Status**: Implemented `LogSoftmax` with max-subtraction, `ComputeKLDivergence`, `ComputeSymmetricKLDivergence`, and `ComputeBatchKLDivergence` in `internal/analysis/kl.go`.
- **Verification**: Verified with analytical distribution checks, identical distribution zeros, and edge cases in `internal/analysis/kl_test.go`.

---

## Completed & Verified Milestones (Archived from Active Roadmap)

All items below have been implemented, tested, and verified in source code:

| Milestone | Key Implementations | Verification Details |
|-----------|---------------------|----------------------|
| **P0.1: Longbow Flight Fallback** | `internal/longbow/client.go` | Automatic fallback to local vector store on flight query/search; roundtrip tests |
| **P0.2: MoE Engine Dynamic Routing** | `internal/engine/moe.go` | Connected `PollActivation` to `DetectRouting`, router logit gating weights, tests |
| **P0.3: Telemetry Aggregate CLI & Flags** | `cmd/burnnotice/main.go` | Added `burnnotice aggregate`, `--cone-analysis`, `--rdo` flags; full CLI tests |
| **P0.4: Telemetry Privacy Spec** | `TELEMETRY.md` | Formal schema specification, strict zero-knowledge privacy guarantees |
| **P0.5: Token Logit KL Divergence** | `internal/analysis/kl.go` | Numerically stable forward/symmetric/batch KL divergence with LogSoftmax |
| **Step 1: ONNX Inference Engine (1.1–1.3.1)** | `internal/engine/onnx.go` | Real ONNX Runtime loading, forward pass, activation capture, Prometheus histograms |
| **Step 2: WASM Backend (2.1–2.3)** | `internal/wasm/backend.go` | Wazero runtime execution, memory I/O, comprehensive integration test suite |
| **Step 3: Real Inference CLI Wiring (3.1–3.2)** | `cmd/burnnotice/main.go` | Removed simulated activations; wired ONNX engine with real token extraction |
| **Step 4: Vector Store & SQLite Fallback (4.1–4.4)** | `internal/longbow/client.go` | SQLite storage, zero-copy binary BLOB serialization, unit & integration tests |
| **Step 5: Hardware-Accelerated Vector Math (5.1–5.4)** | `internal/scoring/scorer.go`, `internal/steer/lora.go` | Gonum `floats.Norm`, `floats.Dot`, `floats.AddScaled`, hotpath microbenchmarks |
| **Step 6: Winsorization & Numerical Stability (6.1–6.4)** | `internal/math/robust.go` | Applied in DIM, PCA, whitened SVD; float64 precision throughout; fuzz tests |
| **Step 7: Whitened SVD Pipeline (7.1–7.4)** | `internal/extract/whitened.go`, `internal/math/` | Shrinkage covariance, ZCA whitening, unwhitening back-transform, `--whitened-svd` CLI flag |
| **Step 8: Concept Cone Geometry (8.1–8.4)** | `internal/analysis/cone.go`, `cmd/burnnotice/main.go` | Cross-category activation collection, pairwise cosine similarity, `--cone-analysis` CLI flag |
| **Step 9: Multi-Direction Subspace & INLP (9.1–9.4)** | `internal/extract/pca.go`, `internal/extract/inlp.go` | Top-k SVD subspace extraction, Gram-Schmidt orthogonalization, subspace energy scoring |
| **Step 10: Ridge-Regularized Projection (10.1–10.3)** | `internal/math/projection.go`, `internal/steer/steering.go` | Gabliteration projection with dynamic alpha scheduling; `RegularizationAlpha` in `Command` |
| **Step 11: MPOA Steering (11.1–11.2)** | `internal/steer/steering.go` | `MethodNormPreserving`, `NormPreservingAblation`, `ApplyCommand`, L2 norm restoration |
| **Step 13: Refusal Direction Optimization (13.1–13.4)** | `internal/extract/rdo.go`, `cmd/burnnotice/main.go` | Linear probe loss, SGD optimizer, AUROC estimation, `--rdo` CLI flag |
| **Step 14: KL Divergence & Metrics (14.1–14.2)** | `internal/analysis/kl.go`, `internal/profile/model.go` | Token-level logit KL divergence; `SteeringMetrics` added to profile export |
| **Step 15: Ouroboros Detection & Refinement (15.1–15.4)** | `internal/analysis/ouroboros.go` | Pre/post layer scoring, compensating layer detection, iterative refinement controller |
| **Step 16: Alignment Imprint Detection (16.1–16.4)** | `internal/analysis/alignment.go` | Effective rank, SV decay, cross-layer coherence, DPO/RLHF/CAI/SFT classification |
| **Step 18: Component-Specific Steering Scaling (18.1)** | `internal/steer/steering.go` | Added `AttentionStrength` and `MLPStrength` to `Command` |
| **Step 24: Sparse Autoencoder Integration (24.1–24.3)** | `internal/sae/loader.go` | Pretrained SAE loader (Gemma Scope NPZ + JSON), `Encode`/`Decode` forward pass, feature attribution scoring |
| **Step 25: Cross-Layer Alignment (25.1–25.4)** | `internal/analysis/crosslayer.go` | Pairwise cosine similarity matrix, decision-layer emergence detection, cluster sets |
| **Step 26: CKA Representational Similarity (26.1–26.4)** | `internal/analysis/crosslayer.go` | Centered Kernel Alignment (CKA), HSIC computation, representational distortion checks |
| **Step 27: Continuous Steering Param (27.2)** | `internal/steer/steering.go` | Added continuous parameter `DirectionMix` to `Command` |
| **Step 31: Anonymous Telemetry Pipeline (31.0–31.5)** | `internal/telemetry/`, `cmd/burnnotice/main.go`, `TELEMETRY.md` | Prometheus metrics, local & HTTP reporter, `burnnotice aggregate` CLI, privacy specification |
| **Step 32: GPU-Native Inference Kernels (32.1–32.4)** | `plugins/cuda_kernels.cu`, `plugins/metal/`, `plugins/vllm/`, `plugins/tensorrt/` | CUDA warp-reduction kernels, Apple Silicon Metal MSL kernels, PyBind11 vLLM plugin, TensorRT plugin |
| **Step 33: Adaptive Calibration (33.1–33.3)** | `internal/profile/manager.go`, `internal/profile/calibration.go` | Held-out calibration worker, F1-optimal threshold calculation, rolling drift detection |
| **Step 35: Quantization Scale Calibration & Transfer (35.2–35.3)** | `internal/steer/quantization.go` | Scale calibration ratio $s_{\text{calib}} = \sigma_{\text{FP16}} / \sigma_{\text{quant}}$, direction transfer with $\ge 0.98$ cosine parity check |
| **Step 36: Temporal Deception Sequence Scoring (36.1–36.2)** | `internal/scoring/aggregator.go` | Exponential Moving Average ($\text{EMA}_t$), consecutive-step sliding window drift trajectory detection |
| **Step 38: Asynchronous Scoring Engine (38.1–38.3)** | `internal/scoring/async.go` | Lock-free ring buffer channel, worker pool, emergency sync bypass at 85% depth or 2.5ms latency limit |
| **Step 39: Context-Aware Semantic Intent Gating (39.1–39.3)** | `internal/policy/gate.go` | Lightweight semantic pre-classifier backend (`LLMClient`), creative/roleplay bypass, context-specific confidence weighting |
| **Step 40: Obfuscation-Resilient Probing (40.1–40.3)** | `internal/profile/obfuscation.go` | Non-linear `MLPProbe` (ReLU+Sigmoid), finite-difference gradient adversarial perturber, Mahalanobis OOD anomaly detection |
| **Step 41: Decentralized Standalone Fallback Engine (41.1–41.3)** | `internal/longbow/client.go` | Local SQLite storage, Gonum-based distance math, zero-downtime hot-swap and automatic failover |
| **Step 42: Closed-Loop Coherence Management (42.1–42.2)** | `internal/steer/coherence.go` | Token-level sliding perplexity tracker, exponential negative log-prob accumulator, dynamic adaptive alpha scaling multiplier |

---

## Active Roadmap: Remaining Uncompleted Steps

### Tier 1 - Infrastructure Foundation

#### Step 1: ONNX Inference Engine Validation
- [ ] **1.4** Validate against a known model (e.g. Qwen2.5-0.5B) end-to-end: load → forward → capture activations (requires ONNX Runtime `.so` at runtime + model file)

#### Step 3: Model Inference Validation
- [ ] **3.3** Validate profile output: extracted directions from real activations must differ from random baselines

---

### Tier 2 - Core Direction Extraction Pipeline

#### Step 7: Whitened SVD Extraction
- [ ] **7.5** Compare singular-value decay rate: raw SVD vs whitened SVD

#### Step 8: Concept Cone Geometry Analysis
- [ ] **8.5** Integrate cone dimensionality into threshold calibration (higher-D cones need more INLP iterations)

#### Step 9: Multi-Direction Subspace Extraction
- [ ] **9.5** UI: 3D scatter plot of multi-dimensional activation projections

#### Step 10: Ridge-Regularized Subspace Projection (Gabliteration)
- [ ] **10.4** Unit tests comparing KL divergence: regularized vs exact projection

#### Step 11: Norm-Preserving Biprojected Ablation (MPOA) Steering
- [ ] **11.3** Benchmark MPOA-steered perplexity vs standard ablation on 3+ model sizes
- [ ] **11.4** Implement biprojection: when removing direction measured at L1 from L2, also remove harmless component from L2

#### Step 12: Bias Term Projection
- [ ] **12.1** Audit all transformer component types; add bias projection alongside activation projection
- [ ] **12.2** Add `BiasDirections` to `ModelProfile` for per-layer bias direction storage
- [ ] **12.3** Implement `ComputeBiasSteering` projecting direction out of bias vectors
- [ ] **12.4** Unit tests: bias-only steering reduces deception scores (without touching activations)

#### Step 13: Refusal Direction Optimization (Gradient-Based Refinement)
- [ ] **13.3** Compare AUROC: DIM vs SVD vs RDO on held-out deception datasets

---

### Tier 3 - Detection & Steering Quality

#### Step 14: KL-Divergence Co-Optimization
- [ ] **14.3** Build KL-co-optimization feedback loop: if KL > 0.2 on harmless prompts, reduce steering strength
- [ ] **14.4** UI: 2D scatter plot (KL divergence vs remaining deception score) with Pareto frontier
- [ ] **14.5** Validate correlation between KL divergence and downstream benchmark degradation

#### Step 17: Conditional Activation Steering (CAST)
- [ ] **17.1** Implement `ConditionalSteering` in `internal/steer/cast.go` - extract condition vectors per bypass category during profiling
- [ ] **17.2** Apply steering scaled by `f(sim(h_l, c))` where f is thresholded step function
- [ ] **17.3** Implement logical condition composition: AND/OR/NOT for complex gating rules
- [ ] **17.4** Benchmark CAST latency vs external LLM classifier (< 0.1ms target)

#### Step 18: Component-Specific Steering Scaling
- [ ] **18.2** Detect component type from model config metadata
- [ ] **18.3** Set defaults: attention=1.0, MLP=0.5 (MLP layers carry knowledge, more sensitive)
- [ ] **18.4** UI: component-type breakdown showing attention vs MLP contribution to steering effect

#### Step 19: Standardized Evaluation - Refusal Rate & KL Tradeoff
- [ ] **19.1** Integrate HarmBench 400: 400 test cases across 7 categories, compute Attack Success Rate (ASR)
- [ ] **19.2** Add ASR to standard steering metrics output
- [ ] **19.3** 2D tradeoff plot: KL divergence (x) vs remaining deception/refusal rate (y), Pareto frontier
- [ ] **19.4** Report "Heretic score": `w1 · refusal_rate + w2 · KL_divergence` as single-number quality metric

#### Step 20: Downstream Benchmark Sensitivity Profiling
- [ ] **20.1** Add TruthfulQA, HellaSwag, ARC, MMLU to steering evaluation suite
- [ ] **20.2** Compute per-benchmark sensitivity scores (degradation under max steering)
- [ ] **20.3** Report per-benchmark degradation in capability health dashboard panel
- [ ] **20.4** Weight capability health ratio by benchmark sensitivity

#### Step 21: Latent-Space Separation Metric
- [ ] **21.1** Implement `ComputeLatentSeparation(harmfulActs, benignActs)` - cosine distance between cluster centroids
- [ ] **21.2** Measure separation pre/post steering; flag if drop > 25%
- [ ] **21.3** UI: "Geometric Separation" gauge on dashboard

---

### Tier 4 - Architecture-Specific Steering

#### Step 22: MoE-Aware Expert-Granular Extraction
- [ ] **22.2** Implement per-expert direction extraction: collect activations separately for each expert
- [ ] **22.3** Implement per-expert steering: apply vectors only to active experts, weighted by router probability
- [ ] **22.4** Add MoE capability benchmark: expert routing entropy before/after steering (must not collapse diversity)
- [ ] **22.5** Tests with synthetic MoE model (2-4 experts) verifying per-expert correctness

#### Step 23: CoT-Aware Steering for Reasoning Models
- [ ] **23.1** Detect `<think>`/`</think>` boundaries (or protocol-level signals) in token stream
- [ ] **23.2** During thinking phase: monitoring only, zero steering strength
- [ ] **23.3** During answer phase: full steering intervention
- [ ] **23.4** Track and report CoT-token-specific deception metrics separately
- [ ] **23.5** Add `ReasoningModelAware` bool to steering config (auto-detect from architecture)

#### Step 24: Sparse Autoencoder Integration
- [ ] **24.4** Add SAE feature monitoring as independent probe alongside DIM/whitening
- [ ] **24.5** Implement SAE feature-level ablation (suppress top-k refusal features directly)
- [ ] **24.6** Tests: SAE monitoring catches jailbreaks that DIM misses (complementary signal)

#### Step 27: Float Direction Interpolation
- [ ] **27.1** Implement `FloatDirectionInterpolation(directions, t, sigma)` - Gaussian-weighted SVD direction blend
- [ ] **27.3** UI: "Direction Mix" slider in strength sweep tab
- [ ] **27.4** Research: does smooth interpolation produce lower KL divergence than discrete selection?

---

### Tier 5 - Multi-Turn & Externalities Safety

#### Step 28: KV-Cache Steering Accumulation Tracking
- [ ] **28.1** Build `KVCacheAccumulationTracker` - maintain per-turn steering vector log, measure cumulative angular drift
- [ ] **28.2** Alert when cumulative drift exceeds `theta_max` (15 degrees)
- [ ] **28.3** Implement Gated Cropped Attention-Delta (GCAD): crop attention-delta between steered/unsteered states to prevent KV-cache propagation
- [ ] **28.4** Multi-turn benchmark: 10-turn conversation with active steering, perplexity per turn
- [ ] **28.5** Fix EMA temporal scoring to account for KV-cache baseline drift (not actual deception intent)

#### Step 29: Steering Externalities Monitoring
- [ ] **29.1** Implement `JailbreakRiskAssessor` - measure ASR on 100 held-out jailbreak prompts, compare baseline vs steered
- [ ] **29.2** Add `SteeringExternalitiesScore` = steered_ASR / baseline_ASR; flag if > 1.2
- [ ] **29.3** Monitor early-token distribution shifts (steering biasing toward non-refusal trajectories)
- [ ] **29.4** Document in design docs: steering carries jailbreak externalities risk

#### Step 30: Extended Refusal Defense
- [ ] **30.1** Build dataset generator: harmful prompts → extended refusal responses (overview + refusal + rationale)
- [ ] **30.2** Implement `ExtendedRefusalFineTuner` using DoRA (not standard LoRA)
- [ ] **30.3** Verify BurnNotice direction extraction still works after defense (defense resists weight-ablation, not inference-time monitoring)
- [ ] **30.4** Document as optional hardening step for high-security deployments

---

### Tier 6 - Community & Telemetry

#### Step 31: Anonymous Telemetry Tooling
- [ ] **31.4** UI: local leaderboard showing community results for current model architecture

---

### Tier 7 - Production Hardening & Platform Pivot

#### Step 32: Serving Engine Runtime Integration
- [ ] **32.5** Move all per-token scoring and steering into GPU VRAM serving loop (Go layer handles async policy/telemetry only)

#### Step 33: Calibration Validation
- [ ] **33.4** UI: ROC curves, precision-recall graphs, "Recalibrate" trigger
- [ ] **33.5** Fuzz tests with extreme validation datasets (all-positive, all-negative, high-variance)

#### Step 34: Adversarial Robustness
- [ ] **34.1** Input token sanitization: filter low-probability token transitions typical of gradient-optimized suffixes
- [ ] **34.2** Steering gradient masking: restrict prompt tracing on high-cosine-similarity probe sequences, generate decoy distributions
- [ ] **34.3** Latent noise injection: `S_robust(a) = S(a + eta)` with Gaussian eta
- [ ] **34.4** UI: "Adversarial Shield" panel with live perturbation rates and blocked jailbreak count
- [ ] **34.5** E2E test: simulated adversarial attack under direct projection manipulation

#### Step 35: Multi-Format Quantization & Direction Transfer
- [ ] **35.1** Implement Turboquant, SQ8, PQ loaders in `internal/engine/onnx.go`
- [ ] **35.4** UI: "Quantization Compatibility" chart
- [ ] **35.5** E2E: INT8 conversion → direction transfer → steering → verify identical behavioral outputs

#### Step 36: Temporal Deception Sequence Scoring
- [ ] **36.3** Policy engine temporal operators (cumulative thresholds, sustained violation windows)
- [ ] **36.4** UI: real-time temporal trend chart with threshold envelopes
- [ ] **36.5** E2E: increasing deceptive prompt sequence → preemptive intervention before completion

#### Step 37: Capability Preservation Benchmarks
- [ ] **37.1** Automated evaluation harness: MMLU, GSM8k, HumanEval under dynamic steering
- [ ] **37.2** Safe steering clamping: theta <= 15 degrees
- [ ] **37.3** Capability health ratio: `R_cap >= 0.98` (if below, auto-scale steering strength down)
- [ ] **37.4** UI: "Capability Health" gauge with per-benchmark retention display
- [ ] **37.5** E2E: max-strength steering while asserting > 98% capability retention

#### Step 38: Asynchronous Non-Blocking Scoring Engine
- [ ] **38.4** Race tests under high parallel load
- [ ] **38.5** UI: system latency panel - queue depth, worker saturation, engine delay

#### Step 42: Closed-Loop Coherence Management
- [ ] **42.3** Regression tests against GSM8k/MMLU during live steering

---

### Tier 8 - Documentation Corrections

#### Step 43: Update Theoretical Claims
- [ ] **43.1** `design.md §3.1`: Replace "single direction" with "multi-dimensional subspace (concept cone)"
- [ ] **43.2** `design.md §10`: Add Rogue Scalpel / Steering Externalities jailbreak risk finding
- [ ] **43.3** `design.md §2.2`: Add Bailey et al. obfuscated activations finding and proposed defenses
- [ ] **43.4** `design.md §12`: Add KV-cache steering accumulation and Prompt-Activation Duality finding
- [ ] **43.5** `design.md §13`: Add references to MPOA, Gabliteration, concept cones, SAE-based steering
- [ ] **43.6** `design.md §2.2`: Add Steering Externalities paper (arXiv:2602.04896) to critical findings

#### Step 44: Document Known Limitations
- [ ] **44.1** SAE limitation: only chat/instruction-data SAEs capture refusal - pretraining SAEs do not
- [ ] **44.2** SAE limitation: different seeds share only ~30% of latents
- [ ] **44.3** Evaluation limitation: single-turn ASR is misleading - multi-turn red teaming exposes up to 75% ASR
- [ ] **44.4** Evaluation limitation: GSM8k measures math, not truthfulness - a model can pass GSM8k and still be deceptive

---

## Deep Research: Latent Space in Language Models - State of the Art

### Background

The latent space of a transformer is the high-dimensional (4096–16384) residual stream that flows between layers, encoding the model's evolving representation of each token. BurnNotice operates in this space by extracting behavioral directions (deception, refusal, truthfulness) and steering activations along them. The field has advanced substantially in 2025–2026:

1. **SAE Monosemanticity Breakthrough** (Anthropic, 2024–2026): Sparse autoencoders decompose dense activations into interpretable features. Scaling to 34M features (Claude 3 Sonnet) showed features correspond to concrete concepts (names, entities, behaviors).

2. **Representation Engineering (RepE)** (Zou et al., 2023–2025): Top-down approach to identify concept vectors via contrastive activation differences - the foundation of BurnNotice's DIM and whitened SVD.

3. **Activation Addition (ActAdd)** (Turner et al., 2023): Simple vector addition in residual stream. BurnNotice implements this alongside angular and ablation methods.

4. **SAE-SSV** (He et al., EMNLP 2025): Supervised steering in sparse representation spaces. Key finding: a remarkably small subspace is sufficient for effective steering.

5. **FGAA** (Soo et al., ICLR 2025 Workshop): Feature Guided Activation Additions combining CAA + SAE-TS - outperforms both.

6. **SAE-RSV** (ICLR 2026): Steering vector refinement via SAE denoising - effective steering from limited data by removing task-irrelevant features and enriching relevant ones.

7. **AxBench** (ICML 2025): Found that even simple baselines outperform SAE for steering - critical counterpoint showing SAE steering is not universally superior.

8. **Prefill-Only Behavioral Commitment** (2026): Research on 35B MoE models shows behavioral decisions commit during prefill phase, not autoregressive decoding. Steering during decoding has zero effect on certain architectures.

9. **CB-SAE** (2025): Concept Bottleneck SAE addressing low interpretability/steerability of standard SAEs via user-specified concept support.

10. **Cross-Layer CKA Analysis**: Centered Kernel Alignment used to measure representational distortion pre/post steering, establishing distortion thresholds.

---

## 10-Part Strategic Recommendations

Each recommendation includes: **purpose** → **approach** → **adversarial criticism**.

---

### Recommendation 1: SAE Feature-Level Steering Integration

**Purpose**: Replace raw activation-space steering with SAE-disentangled feature steering. Raw activations are polysemantic (each neuron participates in many concepts); SAE features are (approximately) monosemantic, enabling precise targeting.

**Approach**:
- Extend `internal/sae/loader.go` to load Gemma Scope and SAELens pretrained SAEs alongside the base model
- Build `SAEFeatureSteerer` in `internal/steer/` that encodes `h_l` through SAE, modifies top-k feature activations, decodes back to residual stream
- Compare SAE-steered KL divergence vs raw ActAdd on identical tasks
- Benchmark SAE encode/decode overhead (target < 50µs per layer)

**Adversarial Criticism**:
- **AxBench counterevidence**: Simple baseline steering (mean-diff) matches or exceeds SAE-based steering on standard benchmarks. The overhead of SAE encoding (10–100µs per layer) may not justify marginal gains.
- **Feature polysemy persists**: SAE features are not truly monosemantic - feature splitting and absorption (Joseph Bloom, 2024) mean a single SAE feature can encode multiple concepts. The "interpretability" may be illusory.
- **Dataset dependence**: SAEs trained on different corpora activate different feature sets for identical inputs (Kissane et al., 2024). Gemma Scope SAEs won't generalize to Qwen or Llama architectures.
- **Latency impact**: At 32 layers × 100µs = 3.2ms per token, SAE steering may add prohibitive latency for real-time agent use cases.
- **Recommendation**: Implement SAE steering as a configurable ADDITIONAL method, not a replacement. Default to raw ActAdd for latency-critical paths.

---

### Recommendation 2: Prefill-Phase Steering Optimization

**Purpose**: Recent findings (2026, 35B MoE study) show behavioral decisions commit during prefill - autoregressive decoding steering has zero causal effect on certain architectures. BurnNotice must detect and exploit the behavioral commitment window.

**Approach**:
- Add `BehavioralCommitmentWindow` detector in `internal/steer/`: measure projection scores at each prefill token position
- During prefill: apply full steering strength at commitment-critical positions
- During decoding: monitoring only (zero steering) for architectures where post-commit steering is causally inert
- Add `--commit-window` flag to enable architecture-specific timing analysis

**Adversarial Criticism**:
- **Architecture-specific**: The prefill-only commitment finding was demonstrated on a specific 35B MoE with GatedDeltaNet recurrence. It may not generalize to standard decoder-only transformers (GPT, Llama, Qwen).
- **Risk of missed intervention**: If commitment window detection is wrong, steering may be applied to the wrong tokens or not at all. False negatives here are safety-critical.
- **Additional complexity**: Per-token commitment analysis adds O(seq_len) overhead per layer - potentially 32× the current per-token cost.
- **Counter-studies needed**: No published replication of prefill-only commitment on dense transformers. BurnNotice should first replicate on Llama-3-8B before building infrastructure around it.
- **Recommendation**: Implement as experimental flag with clear documentation that behavior on non-MoE architectures is unknown.

---

### Recommendation 3: Refusal Direction Optimization via Pure-Go SGD

**Purpose**: Replace heuristic direction extraction (DIM, SVD) with gradient-based Refusal Direction Optimization (RDO). Linear probe loss with SGD converges to directions with higher AUROC than contrastive mean-diff.

**Approach**:
- Implement `RefusalDirectionOptimizer` in `internal/extract/rdo.go` using `mat.Dense` and manual gradients
- Initialize from whitened SVD direction, optimize binary cross-entropy loss with L2 regularization
- Compare AUROC: DIM vs whitened SVD vs RDO on held-out deception datasets
- Add `--rdo` flag to profile CLI

**Adversarial Criticism**:
- **Small-data fragility**: RDO on small datasets (50–100 pairs) overfits dramatically without aggressive regularization. The SVD direction, while not optimal, generalizes better.
- **Pure-Go gradient limitations**: Gonum lacks autograd - manual gradient computation for multi-layer losses is error-prone and hard to extend to non-linear probes.
- **SGD hyperparameter sensitivity**: Learning rate, momentum, weight decay all interact with initialization. The SVD direction is hyperparameter-free and already provides strong baselines.
- **Diminishing returns**: Whitened SVD already achieves AUROC > 0.95 on many refusal detection tasks. The remaining 0.03–0.05 AUROC improvement from RDO may not justify complexity.
- **Recommendation**: Implement RDO as optional refinement step with automatic early stopping via validation AUROC plateau detection.

---

### Recommendation 4: Concept Cone Multi-Category Geometry

**Purpose**: DIM and whitened SVD extract binary directions (deception vs truthful). Real safety requires understanding geometry across N harm categories - "concept cones" where multiple behavioral directions occupy a subspace.

**Approach**:
- Build `ConceptConeAnalyzer` in `internal/analysis/cone.go` collecting activations across N categories (violence, deception, CSAM, bioweapons, etc.)
- Compute pairwise cosine similarity and intrinsic dimensionality via PCA scree
- Calibrate INLP iterations: higher-D cones need more nullspace projections
- Add `--cone-analysis` flag generating category relationship heatmap

**Adversarial Criticism**:
- **Category definition instability**: "Deception" and "harm" are not stable across contexts - the same prompt can be deceptive in one context and truthful in another. Geometrically stable directions may not exist for context-dependent categories.
- **Cone collapse risk**: If all harm categories collapse into the same direction (as the 35B MoE study found - one agency axis, not five traits), the concept cone framework adds complexity without benefit.
- **Scalability cost**: N categories × L layers × D dimensions = massive storage requirements. For 20 categories and 32 layers at 4096-dim, each profile requires ~10MB just for directions.
- **Evaluation difficulty**: Ground-truth for "concept cone correctness" doesn't exist - you can't validate geometric structure without behavioral tests that are themselves unreliable.
- **Recommendation**: Start with N=3 categories (deception, refusal, harm) and validate that cone dimensionality correlates with INLP iteration requirements before scaling.

---

### Recommendation 5: KV-Cache Steering Accumulation & Drift Monitoring

**Purpose**: In multi-turn conversations, steering vectors accumulate in the KV cache, causing angular drift from the intended direction. This drift can flip steering from safe to unsafe behavior.

**Approach**:
- Build `KVCacheAccumulationTracker` in `internal/steer/kv_tracker.go` with per-turn vector log
- Alert when cumulative drift exceeds `theta_max = 15°`
- Implement Gated Cropped Attention-Delta (GCAD) to prevent KV-cache propagation
- Multi-turn benchmark: 10-turn conversation with active steering, perplexity per turn

**Adversarial Criticism**:
- **KV cache internals are architecture-specific**: The KV cache structure differs between GPT, Llama, Qwen, and MoE architectures. A tracker built for one may silently fail on another (no error, just wrong drift measurements).
- **15° threshold is arbitrary**: No published work establishes a principled maximum drift angle. 15° may be too conservative (triggering false alerts) or too permissive (allowing harmful drift).
- **EMA scoring fix (28.5) undermines detection**: The proposed fix to account for KV-cache baseline drift may mask actual deception intent changes. Separating "natural baseline drift" from "deception onset" is an unsolved research problem.
- **GCAD adds significant latency**: Cropping attention deltas per layer per token adds O(L²) operations per generation step.
- **Recommendation**: Implement drift monitoring as a read-only diagnostic first. Establish empirical drift distributions before building alert thresholds or GCAD.

---

### Recommendation 6: Cross-Layer Direction Emergence & CKA Distortion

**Purpose**: Behavioral directions don't appear uniformly - they emerge at specific "decision layers." Identifying these enables minimal steering (fewer layers, less distortion) and reveals where the model commits to deceptive trajectories.

**Approach**:
- Build `CrossLayerAlignmentAnalyzer` in `internal/analysis/crosslayer.go`
- Extract DIM directions per layer, compute pairwise cosine similarity across layers
- Identify the "decision layer" where direction cosine with downstream layers stabilizes (first layer where direction becomes discriminative)
- Compute CKA similarity pre/post steering to identify most-distorted layers for targeted recovery

**Adversarial Criticism**:
- **Layer-to-layer similarity is high by default**: Residual stream representations are similar between adjacent layers even without behavioral steering. Direction similarity may reflect architectural continuity rather than behavioral commitment.
- **CKA is not causal**: High CKA similarity ≠ causal importance. A layer may have high CKA distortion post-steering but zero behavioral impact if downstream layers compensate.
- **Recovery passes add latency**: Targeting recovery to "most distorted layers" adds 10–20µs per layer per recovery pass. For 10 layers, that's 100–200µs per token - non-trivial for real-time steering.
- **Decision layer varies by prompt**: The same model may commit to deception at layer 10 for one prompt and layer 25 for another. Identifying a universal decision layer may be impossible.
- **Recommendation**: Implement CKA as a diagnostic metric before building recovery passes around it. Cross-layer CKA distributions across 100+ prompts would establish whether a stable decision layer exists.

---

### Recommendation 7: Multi-Agent Deception Propagation Monitoring

**Purpose**: In multi-agent systems (LangChain, CrewAI, MCP), deceptive intent can propagate between agents through inter-agent communication. BurnNotice must track deception not per-agent but across the communication graph.

**Approach**:
- Extend `internal/agent/multiagent.go` to maintain cross-agent activation state
- Implement pairwise deception similarity scoring between agent activations
- Alert when deception similarity exceeds threshold (indicating propagation)
- Build communication graph with edge weights = deception contamination risk

**Adversarial Criticism**:
- **Cross-agent activation incomparability**: Agents may use different models, different quantization, or different hardware. Activations from Llama-3-70B and GPT-4o-mini exist in incomparable latent spaces.
- **Synchronization overhead**: Activating cross-agent scoring on every inter-agent message adds latency to agent-to-agent communication, potentially breaking timeouts.
- **False propagation detection**: Two agents may activate similar "deception directions" because they both received the same user prompt, not because deception propagated. Disentangling correlation from causation requires controlled experiments.
- **Narrow applicability**: Most production deployments use single-agent systems. The multi-agent scenario, while important, addresses a small fraction of users.
- **Recommendation**: Build correlation-only detection first (flag when multiple agents show high deception scores simultaneously). Graduate to causal propagation detection only after establishing baseline correlation distributions.

---

### Recommendation 8: Adversarial Robustness via Latent Noise Injection & Gradient Masking

**Purpose**: Steering can be detected and reversed by adversarial attacks that probe the steering direction. Adversarial robustness methods make steering stealthy and resistant to reverse-engineering.

**Approach**:
- Implement latent noise injection: `S_robust(a) = S(a + eta)` with Gaussian eta scaled to 1–5% of activation norm
- Gradient masking: generate decoy direction distributions that mask the true steering direction
- Input token sanitization: filter low-probability token transitions typical of gradient-optimized attack suffixes
- Build "Adversarial Shield" panel with live perturbation rates and blocked jailbreak count

**Adversarial Criticism**:
- **Noise undermines steering efficacy**: Gaussian eta of even 1% norm reduces AUROC by 2–5%. At 5% noise, most steering benefits vanish. The robustness-efficacy tradeoff is steep.
- **Gradient masking is trivially bypassable**: "Security by obscurity" approaches (decoy directions) are well-known to fail against adaptive adversaries. If the adversary knows masking is used, they can average multiple queries to cancel noise.
- **Token sanitization is a cat-and-mouse game**: Gradient-optimized attack suffixes evolve faster than sanitization rules. GCG attacks (Zou et al., 2023) can be adapted to bypass any fixed suffix filter.
- **Latency cost compounds**: Noise injection + gradient masking + token sanitization adds 50–200µs per layer per token. For production latency budgets (< 10ms per token), this may be prohibitive.
- **Recommendation**: Implement noise injection as a tunable parameter with clear documentation of the AUROC-latency tradeoff. Do not promise "robustness" - promise "increased cost for attackers."

---

### Recommendation 9: Float Direction Interpolation & Continuous Steering Mixing

**Purpose**: Discrete direction selection (use direction A or direction B) causes abrupt behavioral changes and high KL divergence. Continuous interpolation between directions (blend of A and B) may produce smoother transitions with lower distortion.

**Approach**:
- Implement `FloatDirectionInterpolation(directions, t, sigma)` - Gaussian-weighted SVD direction blend
- Add `DirectionMix` continuous parameter to `SteeringCommand` (range [0, k] where k is number of directions)
- Build UI slider for continuous direction mixing
- Research: does smooth interpolation produce lower KL divergence than discrete selection?

**Adversarial Criticism**:
- **Interpolation may mix unrelated directions**: Blending deception and truthfulness directions may produce a direction that is neither - a confused representation with unpredictable behavioral effects.
- **No theoretical grounding**: It's unclear why Gaussian-weighted interpolation would be optimal. Linear interpolation, slerp, or learned blending weights may all produce different results without a clear winner.
- **KL divergence may not correlate with capability preservation**: Lower KL divergence does not necessarily mean better capability retention. A "smooth" steering intervention can still remove refusal behavior entirely.
- **UI complexity for marginal benefit**: A slider controlling an ill-defined parameter invites user confusion and misconfiguration.
- **Recommendation**: Research interpolation methods on synthetic subspaces first (where ground truth is known). Only expose UI controls after demonstrating behavioral monotonicity.

---

### Recommendation 10: Standardized Multi-Turn Steering Evaluation Framework

**Purpose**: Current single-turn ASR metrics are misleading - multi-turn red teaming exposes up to 75% ASR vs 20% single-turn. BurnNotice needs a standardized evaluation framework across turns, benchmarks, and attack categories.

**Approach**:
- Integrate HarmBench 400: 400 test cases across 7 harm categories
- Implement multi-turn conversation harness (N=5–10 turns)
- Compute per-turn ASR, cumulative ASR, and "breakthrough turn" distributions
- Report unified "Heretic Score": `w1 · refusal_rate + w2 · KL_divergence + w3 · capability_degradation`
- Generate 2D Pareto frontier plots (KL divergence vs ASR) for steering method comparison

**Adversarial Criticism**:
- **HarmBench may not reflect real attacks**: The 400 test cases are static - adaptive attacks that learn from model responses are fundamentally different and may bypass any static benchmark.
- **Multi-turn evaluation is expensive**: 400 prompts × 10 turns = 4,000 generation calls per evaluation run. At $0.01/call (API) or 100ms/call (local), a full eval takes 40 minutes and costs $40.
- **Heretic Score weights are arbitrary**: No principled method exists to weight refusal rate, KL divergence, and capability degradation. Different deployment contexts require different tradeoffs.
- **Benchmark contamination risk**: Repeated evaluation on the same test set leads to overfitting. The steering method that scores best on HarmBench 400 may generalize poorly to out-of-distribution attacks.
- **Capability degradation measurement is incomplete**: MMLU and GSM8k measure factual knowledge and math - they don't measure creativity, instruction following, or tool-use ability. A model can pass MMLU while being useless for agent tasks.
- **Recommendation**: Build the evaluation harness but expose raw per-metric scores rather than a single Heretic Score. Let users assign weights based on their deployment context.

---

## Summary Table

| # | Recommendation | Primary Benefit | Key Risk | Difficulty |
|---|---|---|---|---|
| 1 | SAE Feature-Level Steering | Monosemantic precision | AxBench shows SAE not always better; latency | Hard |
| 2 | Prefill-Phase Optimization | Causal efficacy at commitment | Architecture-specific; not replicated | Medium |
| 3 | Refusal Direction Optimization (RDO) | Higher AUROC | Overfitting; diminishing returns | Medium |
| 4 | Concept Cone Geometry | Multi-category understanding | Category instability; cone collapse | Hard |
| 5 | KV-Cache Accumulation Tracking | Multi-turn safety | Architecture-specific; 15° threshold arbitrary | Hard |
| 6 | Cross-Layer CKA & Direction Emergence | Minimal-distortion steering | High baseline similarity; CKA ≠ causal | Medium |
| 7 | Multi-Agent Propagation Monitoring | Graph-level safety | Incomparable latent spaces | Hard |
| 8 | Adversarial Robustness | Stealthy steering | Noise undermines efficacy; cat-and-mouse | Hard |
| 9 | Float Direction Interpolation | Smooth transitions | Mixes unrelated directions; ungrounded | Medium |
| 10 | Standardized Multi-Turn Eval Framework | Reliable measurement | Expensive; weights arbitrary; contamination | Medium |

---

## Prioritization

Based on impact vs effort analysis:

**Immediate (Tier 2 parallel work)**:
- #10 (Evaluation framework) - prerequisite for all quality measurement
- #3 (RDO) - builds directly on existing whitened SVD infrastructure
- #6 (Cross-layer CKA) - diagnostic value with modest implementation cost

**Short-term (Tier 3–4)**:
- #4 (Concept cone) - extends existing extraction pipeline
- #9 (Direction interpolation) - UI research, low implementation risk
- #5 (KV-cache tracking) - needs evaluation framework first

**Medium-term (Tier 5+)**:
- #1 (SAE steering) - depends on SAE loader (Step 24) being complete
- #2 (Prefill optimization) - needs architecture detection (Step 23)
- #7 (Multi-agent) - depends on agent SDK integration completing

**Long-term (Tier 7+)**:
- #8 (Adversarial robustness) - only meaningful after evaluation framework validates steering works

---

## Gaps in Current Codebase (Resolved)

All five structural gaps identified during code validation have been resolved:

1. **`internal/analysis/` directory created**: Implemented `ConceptConeAnalyzer`, `OuroborosDetector`, `AlignmentImprintDetector`, `CrossLayerAlignmentAnalyzer`, and `CKA` in Go.
2. **`internal/math/projection.go` created**: Implemented `RidgeRegularizedProjection` and `AlphaScheduler` with tests.
3. **`internal/telemetry/reporter.go` created**: Implemented anonymous local/HTTP telemetry reporting and aggregation helper functions.
4. **`internal/profile/manager.go` created & calibration worker implemented**: Added `CalibratedThreshold`/`CalibrationResult`/`DriftDetector` in `calibration.go`, managed via the profiling manager.
5. **Gemma Scope NPZ+JSON format support implemented**: Added robust `.npz` ZIP and `.npy` parser with Python dictionary sanitization, fully covered by unit tests in `internal/sae`.
