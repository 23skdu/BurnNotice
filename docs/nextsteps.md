# BurnNotice: Prioritized Implementation Roadmap

> **Status**: Active
> **Date**: 2026-05-27
> **Version**: 0.4

---

Flat sequential priority list. Each step must be completed before moving to the next within its tier. Checkmarks indicate completion.

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
- [ ] **9.2** Implement Iterative Nullspace Projection (INLP) for orthogonal safety directions
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
