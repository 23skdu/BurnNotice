# BurnNotice Implementation Roadmap: 10-Part Improvement Plan

> **Status**: Completed (All v0.8 & v0.9 10-Part Improvement Plan Milestones Resolved)
> **Last Updated**: 2026-11-06
> **Version**: 0.9
> **Scope**: Ten-part improvement plan spanning multi-turn crescendo re-anchoring, DoRA extended-refusal fine-tuning, cross-architecture SAE alignment, performance, accuracy, testing, CI/release, architecture compatibility, adversarial defenses, and observability/deployment.

---

## Part 1 — Multi-Turn Crescendo Context Re-Anchoring

**Goal**: Detect slow 15+ turn crescendo escalations without saturating the `15°` angular drift budget, by combining exponential GCAD KV-cache decay with periodic full-prefix subspace re-scoring.

- [x] **Crescendo-aware turn windowing**: Added sliding turn-window tracking and escalation step tagging (`internal/steer/kvcache.go`, `internal/steer/crescendo.go`).
- [x] **Exponential GCAD decay with re-anchor cadence**: Retained `GCADApplyInPlace` exponential decay `γ^Δt` per turn (`0 B/op`) and triggered full-prefix subspace re-scoring every `K` turns (default `K=5`).
- [x] **Angular drift budget accounting**: Maintained incremental angular drift accumulator capped at `θ_max = 15°` and re-baselined the drift reference (`ReAnchorBaseline`) on every full-prefix re-score.
- [x] **Crescendo escalation detector**: Implemented `CrescendoDetector` (`internal/steer/crescendo.go`) combining per-turn drift, full-prefix re-score deltas, and monotonic linear slope across the sliding turn window to flag 15+ turn slow escalations.
- [x] **KV-cache re-anchoring hook**: Exposed `ReAnchor(ctx, prefix)` on `CrescendoDetector` and `ReAnchorBaseline(newBaseline)` on `KVCacheAccumulationTracker`.
- [x] **Tests & oracles**: Added golden oracle (`TestOracleCrescendoReAnchoring`), recall/FPR/drift accuracy gate (`TestAccuracyCrescendoRecallAndDriftBound`), fuzz target (`FuzzCrescendoReAnchoring`), and `-race` concurrency tests (`internal/steer/crescendo_test.go`).
- [x] **Docs & metrics**: Documented re-anchor cadence and drift budget in `docs/architecture.md`; added `burnnotice_crescendo_reanchors_total` and `burnnotice_crescendo_escalations_total` in `internal/metrics/metrics.go`.

**Acceptance criteria**: 15+ turn crescendo scenarios are detected with ≥95% recall at ≤1% FPR on held-out dialogues; angular drift never exceeds `15°` between re-anchors; `GCADApplyInPlace` remains `0 B/op`.

---

## Part 2 — DoRA Extended-Refusal Fine-Tuning (ExtendedRefusalFineTuner)

**Goal**: Optional weight-hardening pass pairing inference-time MPOA steering with weight-decomposed low-rank adaptation on high-risk deployment checkpoints.

- [x] **ExtendedRefusalFineTuner type**: Implemented `ExtendedRefusalFineTuner` in `internal/steer/dora.go` orchestrating weight-hardening passes over deployment checkpoints.
- [x] **DoRA decomposition**: Implemented weight-decomposed low-rank adaptation (`composeDoRAWeights`, `DoRALayerAdapter`) splitting row-wise magnitude `m_i = ||W_0[i]||_2` and directional matrix `V + (alpha/r) B A` on target refusal layers.
- [x] **MPOA pairing**: Paired biprojected MPOA steering directions with the low-rank directional update and closed-loop KL scaling so hardened weights reproduce MPOA refusal suppression while preserving row norms.
- [x] **Checkpoint gating**: Enforced high-risk deployment gating (`ErrCheckpointNotHighRisk`), dry-run verification (`DryRun`), and deterministic bit-exact weight rollback (`RollbackCheckpoint`).
- [x] **Verification**: Added golden oracle (`TestOracleDoRAExtendedRefusal`), KL/norm/coherence accuracy invariants (`TestAccuracyDoRAKLAndNormInvariants`), and fuzz target (`FuzzDoRAExtendedRefusal`).
- [x] **CLI integration**: Added `burnnotice finetune [checkpoint-json] --extended-refusal` command with `--output`, `--rank`, `--alpha`, `--max-kl`, `--dry-run`, and `--high-risk` flags in `cmd/burnnotice/main.go`.
- [x] **Compliance**: Added `DoRAManifest` (rank, alpha, target layers, base/hardened checkpoint SHA-256, max KL, min coherence) to `ReleaseEvidenceBundle` (`internal/compliance/benchmarks.go`).

**Acceptance criteria**: Hardened checkpoint matches MPOA-steered behavior within KL ≤0.2 and coherence ≥baseline; no regression on benign capabilities (HarmBench-400 Pareto non-dominated).

---

## Part 3 — Cross-Architecture SAE Feature Alignment

**Goal**: Extend `internal/sae/` beyond Gemma Scope to Llama-Scope and Qwen-SAE dictionaries with automated feature-absorption diagnostics.

- [x] **SAE loader generalization**: Refactored `internal/sae/loader.go` to support Gemma Scope (`LoadGemmaScope`), Llama-Scope (`LoadLlamaScope`), Qwen-SAE (`LoadQwenSAE`), and auto-detected dictionaries (`LoadDictionary`) via the `SAEDictionary` interface.
- [x] **Architecture detection**: Added `DetectSAEArchitecture` and `MapSAEToModelFamily` in `internal/sae/loader.go` and `DefaultSAEFamily` in `internal/profile/architecture.go`.
- [x] **Feature-absorption diagnostics**: Implemented `AnalyzeFeatureAbsorption` (`internal/sae/alignment.go`) measuring primary decoder alignment, encoder firing recall, child-feature capture ratio, polysemanticity entropy, active coverage, dead feature ratio, L0 sparsity, and reconstruction `R^2`.
- [x] **Cross-architecture alignment**: Implemented `AlignSAEDictionaries` (`internal/sae/alignment.go`) aligning SAE features across architectures via Linear CKA and Procrustes-style structural cosine matching.
- [x] **Hallucination integration**: Added `FamilyEnergyScale` and `IngestFamilyFeatures` in `internal/hallucination/energy.go` to normalize SAE-energy rolling windows across Gemma Scope, Llama-Scope, and Qwen-SAE.
- [x] **Tests & fuzz**: Added loader, absorption, and CKA alignment tests (`internal/sae/alignment_test.go`) and `FuzzSAEFeatureAbsorptionAndAlignment` (`internal/sae/fuzz_test.go`).
- [x] **Docs**: Updated `docs/architecture.md` SAE & threat model sections.

**Acceptance criteria**: Llama-Scope and Qwen-SAE dictionaries load and score equivalently to Gemma Scope; absorption diagnostics report per-feature metrics; no regressions in hallucination detection.

---

## Part 4 — Performance & Memory Optimization

**Goal**: Sustain microsecond-scale hot paths and reduce allocation pressure across scoring, steering, and analysis.

- [x] **MoE cumulative offset slicing** (`internal/engine/moe.go`): maintained clamped `ExpertActivation` slicing and top-k routing benchmarks.
- [x] **Pooled scoring & ring buffer** (`internal/scoring/scorer.go`, `internal/scoring/async.go`): cached direction norms, `sync.Pool` reuse, `PushActivationPooled`.
- [x] **FNV-1a cached thin-SVD projection** (`internal/analysis/latent_vis.go`): retained `svdCache` (`35.1x` cached speedup).
- [x] **Zero-alloc in-place GCAD decay** (`internal/steer/kvcache.go`): kept `GCADApplyInPlace` at `0 B/op`, `0 allocs/op`.
- [x] **Worker-pool concept cone SVD** (`internal/analysis/cone.go`): bounded `parallelFor` up to `GOMAXPROCS`.
- [x] **CPU/mem profiling & manifests** (`benchmark_results.txt`, `internal/compliance/benchmarks.go`): reproducible baselines.
- [x] **New**: Added `BenchmarkCrescendoDetector_StepTurn` and `BenchmarkExtendedRefusalFineTuner_HardenLayer` in `internal/steer/bench_test.go`.

**Acceptance criteria**: No regression vs. stored baselines; all hot paths remain ≤65µs where applicable.

---

## Part 5 — Mathematical Accuracy & Oracles

**Goal**: Keep deterministic oracles and numerical invariants authoritative.

- [x] **Scoring & subspace oracles** (`internal/scoring/oracle_test.go`): exact cosine/sigmoid, golden subspace, temporal aggregator.
- [x] **Steering kernel golden vectors** (`internal/steer/oracle_test.go`): MPOA, biprojection, bias, interpolation, CAST, commitment, coherence, GCAD, LoRA.
- [x] **NaN/Inf & covariance invariants** (`internal/math`, `internal/extract`, `internal/analysis`).
- [x] **CKA identity/orthogonality/scale** (`internal/analysis/cka_accuracy_test.go`).
- [x] **MoE softmax & hallucination calibration** (`internal/engine`, `internal/hallucination`).
- [x] **New oracles**: Added `TestOracleCrescendoReAnchoring`, `TestOracleDoRAExtendedRefusal`, and `TestOracleSAEFeatureAbsorptionAndAlignment` alongside `TestAccuracyCrescendoRecallAndDriftBound`, `TestAccuracyDoRAKLAndNormInvariants`, and `TestAccuracySAECrossArchCKAAndAbsorptionInvariants`.

**Acceptance criteria**: `make accuracy-gate` and `make oracle-gate` green; `go test -count=2` deterministic.

---

## Part 6 — Testing Scope & Concurrency

**Goal**: Maintain and extend fuzz, coverage, and race guarantees.

- [x] **Fuzz targets**: 76 native Go `Fuzz*` targets across all 25 fuzz-enabled packages, including `FuzzCrescendoReAnchoring`, `FuzzDoRAExtendedRefusal`, and `FuzzSAEFeatureAbsorptionAndAlignment`.
- [x] **Coverage**: 90%+ statement coverage across all 26 core packages.
- [x] **Race stress**: `-race` suites for scheduler, longbow, telemetry, async, concept cone, `CrescendoDetector`, and `ExtendedRefusalFineTuner`.
- [x] **Property tests**: Multi-turn crescendo recall/FPR and drift bounds; DoRA KL ceiling and row-norm invariance; SAE absorption and CKA bounds in `[0, 1]`.

**Acceptance criteria**: `go test -race ./...` clean; coverage gates enforced in CI.

---

## Part 7 — CI & Release Quality Gates

**Goal**: Keep verification and release evidence tamper-evident and automated.

- [x] **Pre-commit hook** (`scripts/verify.sh`): gofmt + vet + lint + gosec + govulncheck + unit + race + fuzz + accuracy + oracle gates.
- [x] **CI gates** (`.github/workflows/ci.yml`): accuracy + oracle + profiling + fuzz + release evidence gates.
- [x] **Release evidence bundle** (`internal/compliance/benchmarks.go`, `scripts/release-evidence.sh`, `Makefile`): OS/CPU/Go/ORT versions, dataset SHA-256, commit SHAs, `DoRAManifest`, and `SAEManifest`.
- [x] **New**: Integrated crescendo, DoRA, and SAE oracle/accuracy/fuzz tests into `scripts/verify.sh` and CI gates.

**Acceptance criteria**: CI green on every commit; `make release-evidence` produces `dist/evidence/`.

---

## Part 8 — Architecture Compatibility & Model Families

**Goal**: Preserve calibrated support across Llama, Qwen, Gemma, Mistral, MoE and extend SAE families.

- [x] **Family detection & calibration** (`internal/profile/architecture.go`, `internal/steer/bias.go`, `cot.go`, `commitment.go`).
- [x] **MoE offset partitioning & softmax top-k routing** (`internal/engine/moe.go`).
- [x] **SAE family mapping** (Gemma Scope, Llama-Scope, Qwen-SAE) via `DefaultSAEFamily` in `internal/profile/architecture.go` and `MapSAEToModelFamily` in `internal/sae/loader.go`.
- [x] **Quantization transfer** (`internal/steer/quantization.go`): INT8/FP8 direction mapping and variance scale calibration.
- [x] **Oversight-ablated profiling** (`internal/profile/oversight.go`): alignment-faking resistance.

**Acceptance criteria**: All supported families pass profiling + steering smoke tests.

---

## Part 9 — Adversarial Threat Model & Defenses

**Goal**: Maintain explicit defenses against obfuscated activations, steering externalities, alignment faking, multi-turn KV contamination, topic confusion, ouroboros compensation.

- [x] **Obfuscated activations**: MLPProbe, Mahalanobis OOD, Gaussian noise, GCG sanitization, query-similarity masking (`internal/profile/obfuscation.go`, `internal/steer/adversarial.go`).
- [x] **Steering externalities**: JailbreakRiskAssessor, HereticScoreEvaluator Pareto (`internal/analysis/externalities.go`, `heretic.go`).
- [x] **Alignment faking**: OversightAblator two-phase extraction (`internal/profile/oversight.go`).
- [x] **Multi-turn KV contamination**: GCAD + `CrescendoDetector` periodic full-prefix re-anchoring (`internal/steer/kvcache.go`, `internal/steer/crescendo.go`).
- [x] **Topic confusion**: SemanticGater, CAST boolean trees, ROC F1 calibration.
- [x] **Ouroboros compensation**: bounded multi-pass refinement (`internal/analysis/ouroboros.go`).
- [x] **Weight-level refusal hardening**: `ExtendedRefusalFineTuner` DoRA + MPOA pairing (`internal/steer/dora.go`).

**Acceptance criteria**: Threat-model table in `docs/architecture.md` updated with residual limitations.

---

## Part 10 — Observability, Deployment & Documentation

**Goal**: Keep telemetry, deployment topology, and docs aligned with new capabilities.

- [x] **Telemetry**: 55+ Prometheus metrics + OTLP/HTTP (`internal/metrics/metrics.go`), including `CrescendoReAnchorsTotal`, `CrescendoEscalationsTotal`, `DoRAHardeningPassesTotal`, `SAEFeatureAbsorptionRate`, and `SAECrossArchAlignmentCKA`.
- [x] **Dashboard**: SSE score updates and latent-space SVG visualization (`internal/web/`, `internal/analysis/latent_vis.go`).
- [x] **Deployment**: Helm chart (`helm/burnnotice/`), GPU/CPU clusters, Longbow mesh.
- [x] **Documentation**: Updated `docs/roadmap.md` and `docs/architecture.md`.
- [x] **Quality gate commands**: Verified `scripts/verify.sh` and `Makefile` targets.

**Acceptance criteria**: Docs reflect all 10 parts; metrics exposed and scraped; deployment manifests valid.

---

## Quality Gate Commands

```bash
bash scripts/verify.sh 1x
make lint
make fmt-check
make vet
make security
make accuracy-gate
make oracle-gate
make quality-gate
make release-evidence
make test
make test-race
make test-coverage
make test-fuzz
```

---

## Feature Coverage Checklist (from improvement plan)

- [x] Multi-Turn Crescendo Context Re-Anchoring: Combine exponential GCAD KV-cache decay with periodic full-prefix subspace re-scoring to detect slow 15+ turn crescendo escalations without saturating the 15° angular drift budget.
- [x] DoRA Extended-Refusal Fine-Tuning (ExtendedRefusalFineTuner): Optional weight-hardening pass pairing inference-time MPOA steering with weight-decomposed low-rank adaptation on high-risk deployment checkpoints.
- [x] Cross-Architecture SAE Feature Alignment: Extend internal/sae/ beyond Gemma Scope to Llama-Scope and Qwen-SAE dictionaries with automated feature-absorption diagnostics.
