# BurnNotice: Prioritized Implementation Roadmap

> **Status**: Active
> **Date**: 2026-06-04
> **Version**: 0.5

---

Flat sequential priority list. Each step must be completed before moving to the next within its tier. Checkmarks indicate completion.

---

## Validation Summary (2026-06-04)

All [x]-marked steps were source-code validated. Key findings:

| Step | Status | Notes |
|------|--------|-------|
| 1.1–1.3.1 | ✅ Verified | `internal/engine/onnx.go` — real ONNX Runtime, forward pass, intermediate capture, Prometheus metrics |
| 2.1–2.3 | ✅ Verified | `internal/wasm/backend.go` — wazero execution, memory I/O, tests for identity/empty/large/missing-export |
| 3.1–3.2 | ✅ Verified | `simulateActivation()` fully removed from `cmd/burnnotice/main.go`; uses `LoadONNXModel` |
| 4.1–4.4 | ✅ Verified | `internal/longbow/client.go` + `client_test.go` — SQLite backend, Ingest/Query/Search/Close, zero-copy BLOBs |
| 5.1–5.4 | ✅ Verified | `floats.Norm`/`Dot`/`AddScaled` in scorer.go/lora.go; bench tests in `scoring/bench_test.go`, `steer/bench_test.go` |
| 6.1–6.4 | ✅ Verified | `Winsorize`/`WinsorizeInPlace`/`WinsorizeMatrixInPlace` in `internal/math/robust.go`; applied in DIM/PCA/whitened; fuzz tests |
| 7.1–7.4 | ✅ Verified | ShrinkageCovariance, ZCA whitening, unwhitening back-transform, `--whitened-svd` flag in CLI |
| 9.2 | ✅ **Corrected to [x]** | `internal/extract/inlp.go` — full INLP with Gram-Schmidt orthogonalization and nullspace projection exists |
| 31.0 | ✅ Verified | Prometheus `/metrics` handler in `internal/metrics/handler.go` serves 50+ instruments |

**One correction applied**: Step 9.2 (INLP) had code implemented but was marked [ ]. Updated to [x].

---

## Tier 1 — Infrastructure Foundation (Prerequisites for All Downstream Work)

### Step 1: Implement ONNX Inference Engine
- [x] **1.1** Implement `internal/engine/onnx.go` `Load()` with real ONNX Runtime model loading (not file-existence stub)
- [x] **1.2** Implement `Forward()` that executes model forward pass and returns logits
- [x] **1.3** Implement `GetIntermediate()` that captures per-layer residual stream activations via ONNX Runtime hooks
- [x] **1.3.1** Instrument `Forward()` with `ONNXForwardDuration` histogram and `ActivationsProcessedTotal` counter
- [ ] **1.4** Validate against a known model (e.g. Qwen2.5-0.5B) end-to-end: load → forward → capture activations (blocked: requires ONNX Runtime `.so` at runtime + model file)

### Step 2: Implement WASM Backend
- [x] **2.1** Implement `internal/wasm/backend.go` `Forward()` with real wazero execution — write input to linear memory, call exported `forward` function, read output
- [x] **2.2** Implement `GetIntermediate()` for per-layer activation capture via exported `get_intermediate` (offset, length) from WASM memory
- [x] **2.3** Write integration tests with hand-rolled WASM module: identity forward, multiple calls, large input (10000 floats), empty input, missing export for GetIntermediate

### Step 3: Replace Simulated Activations with Real Model Inference
- [x] **3.1** Remove `simulateActivation()` ASCII-byte path in `cmd/burnnotice/main.go:159-168` — CLI now uses internal engine via `LoadONNXModel`
- [x] **3.2** Wire the profile CLI to load a real model via ONNX/WASM engine and collect actual activations
- [ ] **3.3** Validate profile output: extracted directions from real activations must differ from random baselines

### Step 4: Implement Longbow Integration or Replace with Viable Fallback
- [x] **4.1** Add SQLite vector store backend to `internal/longbow/client.go` (config `Backend: "sqlite"`, uses `modernc.org/sqlite`, stores vectors as BLOBs)
- [x] **4.2** Implement same interface for SQLite as Longbow/Flight — `Ingest`, `Query`, `Search`, `Close` — with `Config.Backend` selector
- [x] **4.3** Zero-copy BLOB serialization via `encoding/binary` (no JSON/Avro overhead)
- [x] **4.4** Write integration tests for SQLite path: ingest → query returns in ID order → search returns nearest by cosine/euclidean

### Step 5: Hardware-Accelerated SIMD Vector Math
- [x] **5.1** Replace manual `vecNorm` loop in `internal/scoring/scorer.go` with `floats.Norm(..., 2)` — already done
- [x] **5.2** Replace manual `vecDot` loop with `floats.Dot()` — already done
- [x] **5.3** Refactor `ApplyLoRA`/`ReverseLoRA` in `internal/steer/lora.go` to use `floats.AddScaled()` — already done
- [x] **5.4** Benchmark hotpaths with `go test -bench` to verify SIMD latency reduction (Score ~27µs, SubspaceScore ~2µs, Dot ~650ns, ActAdd ~10µs for 4096-dim)

### Step 6: Numerical Stability — Winsorization & FP32 Precision
- [x] **6.1** Implement `Winsorize(activations, lower=0.005, upper=0.995)` in `internal/math/robust.go`
- [x] **6.2** Apply winsorization as first step in every direction extraction method (DIM, PCA; INLP via PCA)
- [x] **6.3** Audit all math paths in `internal/extract/` and `internal/scoring/` — ensure float64 on critical paths (all float64, no leaks; `maxFloat64`→1e10 to prevent SVD overflow)
- [x] **6.4** Write fuzz tests with extreme values (NaN, Inf, ±1e30)

---

## Tier 2 — Core Direction Extraction Pipeline

### Step 7: Whitened SVD Extraction
- [x] **7.1** Implement covariance estimation on collected activations (shrinkage estimator for small-sample rank safety)
- [x] **7.2** Implement ZCA whitening transform `X_white = X · cov^{-1/2}`
- [x] **7.3** Perform SVD on whitened difference matrix, transform directions back: `V_orig = cov^{1/2} · V_white`
- [x] **7.4** Add `--whitened-svd` flag to profile CLI (default true for models > 7B)
- [ ] **7.5** Compare singular-value decay rate: raw SVD vs whitened SVD

### Step 8: Concept Cone Geometry Analysis
- [ ] **8.1** Build `ConceptConeAnalyzer` in `internal/analysis/cone.go` collecting activations across N harm categories
- [ ] **8.2** Compute pairwise cosine similarity of direction vectors between categories
- [ ] **8.3** Estimate intrinsic dimensionality via PCA scree plot on the direction matrix
- [ ] **8.4** Add `--cone-analysis` flag to profile CLI
- [ ] **8.5** Integrate cone dimensionality into threshold calibration (higher-D cones need more INLP iterations)

### Step 9: Multi-Direction Subspace Extraction (Default)
- [ ] **9.1** Default to top-k SVD extraction (k=5) in `internal/extract/pca.go` instead of single-direction DIM
- [x] **9.2** Implement Iterative Nullspace Projection (INLP) for orthogonal safety directions (`internal/extract/inlp.go`)
- [ ] **9.3** Add subspace energy scoring: `E_subspace(a) = sqrt(sum((a · v_i)^2))`
- [ ] **9.4** Write unit tests: SVD recovers known synthetic subspaces; INLP projects out known directions
- [ ] **9.5** UI: 3D scatter plot of multi-dimensional activation projections

### Step 10: Ridge-Regularized Subspace Projection (Gabliteration)
- [ ] **10.1** Add `RegularizationAlpha` field to `SteeringCommand` and `DirectionExtractor` config
- [ ] **10.2** Implement `RidgeRegularizedProjection(W, V_k, alpha)` in `internal/math/projection.go`
- [ ] **10.3** Implement dynamic alpha scheduling: start high (soft), decay to 0 (exact) over passes
- [ ] **10.4** Unit tests comparing KL divergence: regularized vs exact projection

### Step 11: Norm-Preserving Biprojected Ablation (MPOA) Steering
- [ ] **11.1** Implement MPOA decomposition as `MethodNormPreserving` in `internal/steer/steering.go`
- [ ] **11.2** Add Frobenius norm restoration post-projection for weight orthogonalization paths
- [ ] **11.3** Benchmark MPOA-steered perplexity vs standard ablation on 3+ model sizes
- [ ] **11.4** Implement biprojection: when removing direction measured at L1 from L2, also remove harmless component from L2

### Step 12: Bias Term Projection
- [ ] **12.1** Audit all transformer component types; add bias projection alongside activation projection
- [ ] **12.2** Add `BiasDirections` to `ModelProfile` for per-layer bias direction storage
- [ ] **12.3** Implement `ComputeBiasSteering` projecting direction out of bias vectors
- [ ] **12.4** Unit tests: bias-only steering reduces deception scores (without touching activations)

### Step 13: Refusal Direction Optimization (Gradient-Based Refinement)
- [ ] **13.1** Implement `RefusalDirectionOptimizer` — linear probe loss, SGD initialized from SVD direction
- [ ] **13.2** Add `--rdo` flag to profile CLI
- [ ] **13.3** Compare AUROC: DIM vs SVD vs RDO on held-out deception dataset
- [ ] **13.4** Implement in pure Go via `mat.Dense` + manual gradients (no autograd dependency)

---

## Tier 3 — Detection & Steering Quality

### Step 14: KL-Divergence Co-Optimization
- [ ] **14.1** Implement `ComputeKLDivergence(originalLogits, steeredLogits)` on token-level log-probs
- [ ] **14.2** Add `KLDivergence` to `SteeringMetrics` in profile format
- [ ] **14.3** Build KL-co-optimization feedback loop: if KL > 0.2 on harmless prompts, reduce steering strength
- [ ] **14.4** UI: 2D scatter plot (KL divergence vs remaining deception score) with Pareto frontier
- [ ] **14.5** Validate correlation between KL divergence and downstream benchmark degradation

### Step 15: Ouroboros Detection & Iterative Refinement
- [ ] **15.1** Build `OuroborosDetector` in `internal/analysis/ouroboros.go` measure per-layer scores pre/post intervention
- [ ] **15.2** Identify compensating layers (scores increased post-intervention); flag if net reduction < 50%
- [ ] **15.3** Implement iterative refinement: secondary passes targeting only compensating layers
- [ ] **15.4** Add `MaxRefinementPasses` config (default 3)

### Step 16: Alignment Imprint Detection
- [ ] **16.1** Build `AlignmentImprintDetector` — extract DIM directions, compute geometric metrics (effective rank, SV decay, cross-layer coherence)
- [ ] **16.2** Classify alignment method (DPO/RLHF/CAI/SFT) via MLP or SVM trained on geometric features
- [ ] **16.3** Add `AlignmentMethod` to `ModelProfile`; auto-tune regularization and projection strategy
- [ ] **16.4** Integrate with policy engine: adjust steering strength by alignment fragility

### Step 17: Conditional Activation Steering (CAST)
- [ ] **17.1** Implement `ConditionalSteering` in `internal/steer/cast.go` — extract condition vectors per bypass category during profiling
- [ ] **17.2** Apply steering scaled by `f(sim(h_l, c))` where f is thresholded step function
- [ ] **17.3** Implement logical condition composition: AND/OR/NOT for complex gating rules
- [ ] **17.4** Benchmark CAST latency vs external LLM classifier (< 0.1ms target)

### Step 18: Component-Specific Steering Scaling
- [ ] **18.1** Add `AttentionStrength` and `MLPStrength` to `SteeringCommand` (range [0,2])
- [ ] **18.2** Detect component type from model config metadata
- [ ] **18.3** Set defaults: attention=1.0, MLP=0.5 (MLP layers carry knowledge, more sensitive)
- [ ] **18.4** UI: component-type breakdown showing attention vs MLP contribution to steering effect

### Step 19: Standardized Evaluation — Refusal Rate & KL Tradeoff
- [ ] **19.1** Integrate HarmBench 400: 400 test cases across 7 categories, compute Attack Success Rate (ASR)
- [ ] **19.2** Add ASR to standard steering metrics output
- [ ] **19.3** 2D tradeoff plot: KL divergence (x) vs remaining deception/refusal rate (y), Pareto frontier
- [ ] **19.4** Report "Heretic score": `w1 · refusal_rate + w2 · KL_divergence` as single-number quality metric

### Step 20: Downstream Benchmark Sensitivity Profiling
- [ ] **20.1** Add TruthfulQA, HellaSwag, ARC, MMLU to steering evaluation suite
- [ ] **20.2** Compute per-benchmark sensitivity scores (degradation under max steering)
- [ ] **20.3** Report per-benchmark degradation in capability health dashboard panel
- [ ] **20.4** Weight capability health ratio by benchmark sensitivity

### Step 21: Latent-Space Separation Metric
- [ ] **21.1** Implement `ComputeLatentSeparation(harmfulActs, benignActs)` — cosine distance between cluster centroids
- [ ] **21.2** Measure separation pre/post steering; flag if drop > 25%
- [ ] **21.3** UI: "Geometric Separation" gauge on dashboard

---

## Tier 4 — Architecture-Specific Steering

### Step 22: MoE-Aware Expert-Granular Extraction
- [ ] **22.1** Rewrite `internal/engine/moe.go` to capture router logits per token (which experts, what weights)
- [ ] **22.2** Implement per-expert direction extraction: collect activations separately for each expert
- [ ] **22.3** Implement per-expert steering: apply vectors only to active experts, weighted by router probability
- [ ] **22.4** Add MoE capability benchmark: expert routing entropy before/after steering (must not collapse diversity)
- [ ] **22.5** Tests with synthetic MoE model (2-4 experts) verifying per-expert correctness

### Step 23: CoT-Aware Steering for Reasoning Models
- [ ] **23.1** Detect `<think>`/`</think>` boundaries (or protocol-level signals) in token stream
- [ ] **23.2** During thinking phase: monitoring only, zero steering strength
- [ ] **23.3** During answer phase: full steering intervention
- [ ] **23.4** Track and report CoT-token-specific deception metrics separately
- [ ] **23.5** Add `ReasoningModelAware` bool to steering config (auto-detect from architecture)

### Step 24: Sparse Autoencoder Integration
- [ ] **24.1** Load pretrained SAEs (Gemma Scope, SAELens) alongside base model in inference engine
- [ ] **24.2** Build `SAEEncoder` in `internal/extract/sae.go`: encode activations through SAE
- [ ] **24.3** Feature attribution scoring: correlate SAE features with deception/refusal labels
- [ ] **24.4** Add SAE feature monitoring as independent probe alongside DIM/whitening
- [ ] **24.5** Implement SAE feature-level ablation (suppress top-k refusal features directly)
- [ ] **24.6** Tests: SAE monitoring catches jailbreaks that DIM misses (complementary signal)

### Step 25: Cross-Layer Alignment & Refusal Propagation Analysis
- [ ] **25.1** Build `CrossLayerAlignmentAnalyzer` — extract directions per layer, compute pairwise cosine similarity
- [ ] **25.2** Identify direction clusters and the "decision layer" where direction first becomes discriminative
- [ ] **25.3** Use decision-layer analysis to determine minimum steering layer set
- [ ] **25.4** UI: cross-layer heatmap showing direction similarity across all layer pairs

### Step 26: CKA Similarity for Pre/Post Steering Analysis
- [ ] **26.1** Implement Centered Kernel Alignment (CKA) between pre/post layer activations
- [ ] **26.2** Add CKA to steering metrics alongside KL divergence and deception score
- [ ] **26.3** Establish CKA threshold: < 0.95 → excessive representational distortion
- [ ] **26.4** Layer-wise CKA to identify most-distorted layers for targeted recovery passes

### Step 27: Float Direction Interpolation
- [ ] **27.1** Implement `FloatDirectionInterpolation(directions, t, sigma)` — Gaussian-weighted SVD direction blend
- [ ] **27.2** Add `DirectionMix` continuous param to `SteeringCommand` (range [0, k])
- [ ] **27.3** UI: "Direction Mix" slider in strength sweep tab
- [ ] **27.4** Research: does smooth interpolation produce lower KL divergence than discrete selection?

---

## Tier 5 — Multi-Turn & Externalities Safety

### Step 28: KV-Cache Steering Accumulation Tracking
- [ ] **28.1** Build `KVCacheAccumulationTracker` — maintain per-turn steering vector log, measure cumulative angular drift
- [ ] **28.2** Alert when cumulative drift exceeds `theta_max` (15 degrees)
- [ ] **28.3** Implement Gated Cropped Attention-Delta (GCAD): crop attention-delta between steered/unsteered states to prevent KV-cache propagation
- [ ] **28.4** Multi-turn benchmark: 10-turn conversation with active steering, perplexity per turn
- [ ] **28.5** Fix EMA temporal scoring to account for KV-cache baseline drift (not actual deception intent)

### Step 29: Steering Externalities Monitoring
- [ ] **29.1** Implement `JailbreakRiskAssessor` — measure ASR on 100 held-out jailbreak prompts, compare baseline vs steered
- [ ] **29.2** Add `SteeringExternalitiesScore` = steered_ASR / baseline_ASR; flag if > 1.2
- [ ] **29.3** Monitor early-token distribution shifts (steering biasing toward non-refusal trajectories)
- [ ] **29.4** Document in design docs: steering carries jailbreak externalities risk

### Step 30: Extended Refusal Defense
- [ ] **30.1** Build dataset generator: harmful prompts → extended refusal responses (overview + refusal + rationale)
- [ ] **30.2** Implement `ExtendedRefusalFineTuner` using DoRA (not standard LoRA)
- [ ] **30.3** Verify BurnNotice direction extraction still works after defense (defense resists weight-ablation, not inference-time monitoring)
- [ ] **30.4** Document as optional hardening step for high-security deployments

---

## Tier 6 — Community & Telemetry

### Step 31: Anonymous Telemetry Infrastructure
- [x] **31.0** Prometheus `/metrics` endpoint added via `internal/metrics/handler.go` — serves 50+ metric instruments, scrapable by OTel Collector
- [ ] **31.1** Design telemetry schema: model architecture, model size, steering method, refusal rate, KL divergence, hardware, timestamp. Never prompts, outputs, IPs, or identity
- [ ] **31.2** Implement submission in `internal/telemetry/reporter.go` (HTTPS POST or local save + PR)
- [ ] **31.3** Build `burnnotice aggregate` CLI command: summarize contributed results
- [ ] **31.4** UI: local leaderboard showing community results for current model architecture
- [ ] **31.5** Write `TELEMETRY.md` with schema documentation and privacy guarantees

---

## Tier 7 — Production Hardening & Platform Pivot

### Step 32: GPU-Native Inference (vLLM / TensorRT-LLM Plugin)
- [ ] **32.1** Implement CUDA kernels for cosine similarity and dynamic rotation (`plugins/cuda_kernels.cu`)
- [ ] **32.2** Implement Metal Shading Language kernels for Apple Silicon (`plugins/metal/kernels.metal`)
- [ ] **32.3** Build vLLM plugin wrappers (PyBind11) linking native kernels into vLLM execution graph
- [ ] **32.4** Build TensorRT-LLM plugin (`IPluginV2DynamicExt` API)
- [ ] **32.5** Move all per-token scoring and steering into GPU VRAM; Go layer handles async policy/telemetry only

### Step 33: Adaptive Model-Specific Calibration
- [ ] **33.1** Calibration worker in `internal/profile/manager.go`: load held-out dataset, build ROC curves, compute F1-optimal thresholds
- [ ] **33.2** Persist `CalibratedThreshold` struct to `profile.gob`
- [ ] **33.3** Real-time drift detection: if rolling avg projection drift > 1.5 sigma, trigger recalibration
- [ ] **33.4** UI: ROC curves, precision-recall graphs, "Recalibrate" trigger
- [ ] **33.5** Fuzz tests with extreme validation datasets (all-positive, all-negative, high-variance)

### Step 34: Adversarial Robustness
- [ ] **34.1** Input token sanitization: filter low-probability token transitions typical of gradient-optimized suffixes
- [ ] **34.2** Steering gradient masking: restrict prompt tracing on high-cosine-similarity probe sequences, generate decoy distributions
- [ ] **34.3** Latent noise injection: `S_robust(a) = S(a + eta)` with Gaussian eta
- [ ] **34.4** UI: "Adversarial Shield" panel with live perturbation rates and blocked jailbreak count
- [ ] **34.5** E2E test: simulated adversarial attack under direct projection manipulation

### Step 35: Multi-Format Quantization & Direction Transfer
- [ ] **35.1** Implement Turboquant, SQ8, PQ loaders in `internal/engine/onnx.go`
- [ ] **35.2** Scale calibration: `s_calib = std(a_FP16) / std(a_quant)`
- [ ] **35.3** Direction transfer: `v_quant = s_calib · v_FP16`; cosine parity check >= 0.98
- [ ] **35.4** UI: "Quantization Compatibility" chart
- [ ] **35.5** E2E: INT8 conversion → direction transfer → steering → verify identical behavioral outputs

### Step 36: Temporal Deception Sequence Scoring
- [ ] **36.1** EMA scoring pipeline: `EMA_t = alpha · S(a_t) + (1-alpha) · EMA_{t-1}`
- [ ] **36.2** Sliding window monitoring for slow-drift deception patterns
- [ ] **36.3** Policy engine temporal operators (cumulative thresholds, sustained violation windows)
- [ ] **36.4** UI: real-time temporal trend chart with threshold envelopes
- [ ] **36.5** E2E: increasing deceptive prompt sequence → preemptive intervention before completion

### Step 37: Capability Preservation Benchmarks
- [ ] **37.1** Automated evaluation harness: MMLU, GSM8k, HumanEval under dynamic steering
- [ ] **37.2** Safe steering clamping: theta <= 15 degrees
- [ ] **37.3** Capability health ratio: `R_cap >= 0.98` (if below, auto-scale steering strength down)
- [ ] **37.4** UI: "Capability Health" gauge with per-benchmark retention display
- [ ] **37.5** E2E: max-strength steering while asserting > 98% capability retention

### Step 38: Asynchronous Non-Blocking Scoring Engine
- [ ] **38.1** Lock-free ring buffer for zero-copy activation handoff from inference to scoring
- [ ] **38.2** Background worker pool for subspace projection calculations
- [ ] **38.3** Emergency sync bypass at 85% queue capacity or 2.5ms latency
- [ ] **38.4** Race tests under high parallel load
- [ ] **38.5** UI: system latency panel — queue depth, worker saturation, engine delay

### Step 39: Context-Aware Semantic Intent Gating
- [ ] **39.1** Parallel lightweight semantic pre-classifier (Llama Guard or distilled model)
- [ ] **39.2** Dynamic gating logic: disable `deception_direction` for creative writing, academic analysis, roleplay
- [ ] **39.3** Class-specific confidence weights scaling intervention strength by conversational context

### Step 40: Obfuscation-Resilient Probe Training
- [ ] **40.1** Upgrade linear probes to non-linear MLPs and multi-directional ensembles
- [ ] **40.2** Adversarial training: perturb activations via gradient-based noise during profiling
- [ ] **40.3** Real-time OOD anomaly detection on residual stream

### Step 41: Decentralized Standalone Fallback Engine
- [ ] **41.1** Local SQLite storage for profile metadata and calibration thresholds
- [ ] **41.2** Gonum-based local math backend (cosine/dot fallback)
- [ ] **41.3** Zero-downtime hot-swap between Longbow Flight and local fallback

### Step 42: Closed-Loop Coherence Management
- [ ] **42.1** Token-level perplexity tracking and coherence checker in generation loop
- [ ] **42.2** Adaptive feedback: scale down alpha or shift layers on consecutive degradation
- [ ] **42.3** Regression tests against GSM8k/MMLU during live steering

---

## Tier 8 — Documentation Corrections

### Step 43: Update Theoretical Claims
- [ ] **43.1** `design.md §3.1`: Replace "single direction" with "multi-dimensional subspace (concept cone)"
- [ ] **43.2** `design.md §10`: Add Rogue Scalpel / Steering Externalities jailbreak risk finding
- [ ] **43.3** `design.md §2.2`: Add Bailey et al. obfuscated activations finding and proposed defenses
- [ ] **43.4** `design.md §12`: Add KV-cache steering accumulation and Prompt-Activation Duality finding
- [ ] **43.5** `design.md §13`: Add references to MPOA, Gabliteration, concept cones, SAE-based steering
- [ ] **43.6** `design.md §2.2`: Add Steering Externalities paper (arXiv:2602.04896) to critical findings

### Step 44: Document Known Limitations
- [ ] **44.1** SAE limitation: only chat/instruction-data SAEs capture refusal — pretraining SAEs do not
- [ ] **44.2** SAE limitation: different seeds share only ~30% of latents
- [ ] **44.3** Evaluation limitation: single-turn ASR is misleading — multi-turn red teaming exposes up to 75% ASR
- [ ] **44.4** Evaluation limitation: GSM8k measures math, not truthfulness — a model can pass GSM8k and still be deceptive

---

## Deep Research: Latent Space in Language Models — State of the Art

### Background

The latent space of a transformer is the high-dimensional (4096–16384) residual stream that flows between layers, encoding the model's evolving representation of each token. BurnNotice operates in this space by extracting behavioral directions (deception, refusal, truthfulness) and steering activations along them. The field has advanced substantially in 2025–2026:

1. **SAE Monosemanticity Breakthrough** (Anthropic, 2024–2026): Sparse autoencoders decompose dense activations into interpretable features. Scaling to 34M features (Claude 3 Sonnet) showed features correspond to concrete concepts (names, entities, behaviors).

2. **Representation Engineering (RepE)** (Zou et al., 2023–2025): Top-down approach to identify concept vectors via contrastive activation differences — the foundation of BurnNotice's DIM and whitened SVD.

3. **Activation Addition (ActAdd)** (Turner et al., 2023): Simple vector addition in residual stream. BurnNotice implements this alongside angular and ablation methods.

4. **SAE-SSV** (He et al., EMNLP 2025): Supervised steering in sparse representation spaces. Key finding: a remarkably small subspace is sufficient for effective steering.

5. **FGAA** (Soo et al., ICLR 2025 Workshop): Feature Guided Activation Additions combining CAA + SAE-TS — outperforms both.

6. **SAE-RSV** (ICLR 2026): Steering vector refinement via SAE denoising — effective steering from limited data by removing task-irrelevant features and enriching relevant ones.

7. **AxBench** (ICML 2025): Found that even simple baselines outperform SAE for steering — critical counterpoint showing SAE steering is not universally superior.

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
- **Feature polysemy persists**: SAE features are not truly monosemantic — feature splitting and absorption (Joseph Bloom, 2024) mean a single SAE feature can encode multiple concepts. The "interpretability" may be illusory.
- **Dataset dependence**: SAEs trained on different corpora activate different feature sets for identical inputs (Kissane et al., 2024). Gemma Scope SAEs won't generalize to Qwen or Llama architectures.
- **Latency impact**: At 32 layers × 100µs = 3.2ms per token, SAE steering may add prohibitive latency for real-time agent use cases.
- **Recommendation**: Implement SAE steering as a configurable ADDITIONAL method, not a replacement. Default to raw ActAdd for latency-critical paths.

---

### Recommendation 2: Prefill-Phase Steering Optimization

**Purpose**: Recent findings (2026, 35B MoE study) show behavioral decisions commit during prefill — autoregressive decoding steering has zero causal effect on certain architectures. BurnNotice must detect and exploit the behavioral commitment window.

**Approach**:
- Add `BehavioralCommitmentWindow` detector in `internal/steer/`: measure projection scores at each prefill token position
- During prefill: apply full steering strength at commitment-critical positions
- During decoding: monitoring only (zero steering) for architectures where post-commit steering is causally inert
- Add `--commit-window` flag to enable architecture-specific timing analysis

**Adversarial Criticism**:
- **Architecture-specific**: The prefill-only commitment finding was demonstrated on a specific 35B MoE with GatedDeltaNet recurrence. It may not generalize to standard decoder-only transformers (GPT, Llama, Qwen).
- **Risk of missed intervention**: If commitment window detection is wrong, steering may be applied to the wrong tokens or not at all. False negatives here are safety-critical.
- **Additional complexity**: Per-token commitment analysis adds O(seq_len) overhead per layer — potentially 32× the current per-token cost.
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
- **Pure-Go gradient limitations**: Gonum lacks autograd — manual gradient computation for multi-layer losses is error-prone and hard to extend to non-linear probes.
- **SGD hyperparameter sensitivity**: Learning rate, momentum, weight decay all interact with initialization. The SVD direction is hyperparameter-free and already provides strong baselines.
- **Diminishing returns**: Whitened SVD already achieves AUROC > 0.95 on many refusal detection tasks. The remaining 0.03–0.05 AUROC improvement from RDO may not justify complexity.
- **Recommendation**: Implement RDO as optional refinement step with automatic early stopping via validation AUROC plateau detection.

---

### Recommendation 4: Concept Cone Multi-Category Geometry

**Purpose**: DIM and whitened SVD extract binary directions (deception vs truthful). Real safety requires understanding geometry across N harm categories — "concept cones" where multiple behavioral directions occupy a subspace.

**Approach**:
- Build `ConceptConeAnalyzer` in `internal/analysis/cone.go` collecting activations across N categories (violence, deception, CSAM, bioweapons, etc.)
- Compute pairwise cosine similarity and intrinsic dimensionality via PCA scree
- Calibrate INLP iterations: higher-D cones need more nullspace projections
- Add `--cone-analysis` flag generating category relationship heatmap

**Adversarial Criticism**:
- **Category definition instability**: "Deception" and "harm" are not stable across contexts — the same prompt can be deceptive in one context and truthful in another. Geometrically stable directions may not exist for context-dependent categories.
- **Cone collapse risk**: If all harm categories collapse into the same direction (as the 35B MoE study found — one agency axis, not five traits), the concept cone framework adds complexity without benefit.
- **Scalability cost**: N categories × L layers × D dimensions = massive storage requirements. For 20 categories and 32 layers at 4096-dim, each profile requires ~10MB just for directions.
- **Evaluation difficulty**: Ground-truth for "concept cone correctness" doesn't exist — you can't validate geometric structure without behavioral tests that are themselves unreliable.
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

**Purpose**: Behavioral directions don't appear uniformly — they emerge at specific "decision layers." Identifying these enables minimal steering (fewer layers, less distortion) and reveals where the model commits to deceptive trajectories.

**Approach**:
- Build `CrossLayerAlignmentAnalyzer` in `internal/analysis/crosslayer.go`
- Extract DIM directions per layer, compute pairwise cosine similarity across layers
- Identify the "decision layer" where direction cosine with downstream layers stabilizes (first layer where direction becomes discriminative)
- Compute CKA similarity pre/post steering to identify most-distorted layers for targeted recovery

**Adversarial Criticism**:
- **Layer-to-layer similarity is high by default**: Residual stream representations are similar between adjacent layers even without behavioral steering. Direction similarity may reflect architectural continuity rather than behavioral commitment.
- **CKA is not causal**: High CKA similarity ≠ causal importance. A layer may have high CKA distortion post-steering but zero behavioral impact if downstream layers compensate.
- **Recovery passes add latency**: Targeting recovery to "most distorted layers" adds 10–20µs per layer per recovery pass. For 10 layers, that's 100–200µs per token — non-trivial for real-time steering.
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
- **Recommendation**: Implement noise injection as a tunable parameter with clear documentation of the AUROC-latency tradeoff. Do not promise "robustness" — promise "increased cost for attackers."

---

### Recommendation 9: Float Direction Interpolation & Continuous Steering Mixing

**Purpose**: Discrete direction selection (use direction A or direction B) causes abrupt behavioral changes and high KL divergence. Continuous interpolation between directions (blend of A and B) may produce smoother transitions with lower distortion.

**Approach**:
- Implement `FloatDirectionInterpolation(directions, t, sigma)` — Gaussian-weighted SVD direction blend
- Add `DirectionMix` continuous parameter to `SteeringCommand` (range [0, k] where k is number of directions)
- Build UI slider for continuous direction mixing
- Research: does smooth interpolation produce lower KL divergence than discrete selection?

**Adversarial Criticism**:
- **Interpolation may mix unrelated directions**: Blending deception and truthfulness directions may produce a direction that is neither — a confused representation with unpredictable behavioral effects.
- **No theoretical grounding**: It's unclear why Gaussian-weighted interpolation would be optimal. Linear interpolation, slerp, or learned blending weights may all produce different results without a clear winner.
- **KL divergence may not correlate with capability preservation**: Lower KL divergence does not necessarily mean better capability retention. A "smooth" steering intervention can still remove refusal behavior entirely.
- **UI complexity for marginal benefit**: A slider controlling an ill-defined parameter invites user confusion and misconfiguration.
- **Recommendation**: Research interpolation methods on synthetic subspaces first (where ground truth is known). Only expose UI controls after demonstrating behavioral monotonicity.

---

### Recommendation 10: Standardized Multi-Turn Steering Evaluation Framework

**Purpose**: Current single-turn ASR metrics are misleading — multi-turn red teaming exposes up to 75% ASR vs 20% single-turn. BurnNotice needs a standardized evaluation framework across turns, benchmarks, and attack categories.

**Approach**:
- Integrate HarmBench 400: 400 test cases across 7 harm categories
- Implement multi-turn conversation harness (N=5–10 turns)
- Compute per-turn ASR, cumulative ASR, and "breakthrough turn" distributions
- Report unified "Heretic Score": `w1 · refusal_rate + w2 · KL_divergence + w3 · capability_degradation`
- Generate 2D Pareto frontier plots (KL divergence vs ASR) for steering method comparison

**Adversarial Criticism**:
- **HarmBench may not reflect real attacks**: The 400 test cases are static — adaptive attacks that learn from model responses are fundamentally different and may bypass any static benchmark.
- **Multi-turn evaluation is expensive**: 400 prompts × 10 turns = 4,000 generation calls per evaluation run. At $0.01/call (API) or 100ms/call (local), a full eval takes 40 minutes and costs $40.
- **Heretic Score weights are arbitrary**: No principled method exists to weight refusal rate, KL divergence, and capability degradation. Different deployment contexts require different tradeoffs.
- **Benchmark contamination risk**: Repeated evaluation on the same test set leads to overfitting. The steering method that scores best on HarmBench 400 may generalize poorly to out-of-distribution attacks.
- **Capability degradation measurement is incomplete**: MMLU and GSM8k measure factual knowledge and math — they don't measure creativity, instruction following, or tool-use ability. A model can pass MMLU while being useless for agent tasks.
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
- #10 (Evaluation framework) — prerequisite for all quality measurement
- #3 (RDO) — builds directly on existing whitened SVD infrastructure
- #6 (Cross-layer CKA) — diagnostic value with modest implementation cost

**Short-term (Tier 3–4)**:
- #4 (Concept cone) — extends existing extraction pipeline
- #9 (Direction interpolation) — UI research, low implementation risk
- #5 (KV-cache tracking) — needs evaluation framework first

**Medium-term (Tier 5+)**:
- #1 (SAE steering) — depends on SAE loader (Step 24) being complete
- #2 (Prefill optimization) — needs architecture detection (Step 23)
- #7 (Multi-agent) — depends on agent SDK integration completing

**Long-term (Tier 7+)**:
- #8 (Adversarial robustness) — only meaningful after evaluation framework validates steering works

---

## Gaps in Current Codebase

Beyond the 10 recommendations, five structural gaps were identified during code validation:

1. **`internal/analysis/` directory missing**: Referenced by Steps 8, 15, 16 but does not exist. Must be created.
2. **`internal/math/projection.go` missing**: Required for Step 10 (ridge projection) — does not exist yet.
3. **`internal/telemetry/reporter.go` missing**: Required for Step 31 telemetry — does not exist yet.
4. **`internal/profile/manager.go` missing**: Required for Step 33 calibration worker — does not exist yet.
5. **SAE loader lacks Gemma Scope format support**: `internal/sae/loader.go` loads JSON-format SAEs, but Gemma Scope uses a different serialization format (NPZ + config JSON).
