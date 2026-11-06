# BurnNotice Roadmap: Performance, Accuracy, and Quality Gates

> **Status**: Completed (All 20 Items Resolved)
> **Last Updated**: 2026-11-05
> **Scope**: 20 prioritized improvements for performance, accuracy, unit-test coverage, fuzz coverage, and CI quality gates.

---

## 0. Baseline & Resolution Summary

BurnNotice now maintains a fully formatted (`gofmt -s`), lint-clean (`golangci-lint run ./...` -> **0 issues**), vulnerability-free (`gosec` -> **0 issues**, `govulncheck` -> **0 vulnerabilities**), and race-clean (`go test -race ./...`) codebase:

- **Static Analysis**: All 69 pre-existing `golangci-lint` findings (`errcheck`, `govet`, `ineffassign`, `staticcheck`, `unused`) resolved to **0 issues**.
- **Fuzz Coverage**: Expanded to **72 `Fuzz*` targets across 25 packages**, covering all `internal/*` and `cmd/*` packages.
- **Unit Test Coverage**: Core packages raised to **90%–100%** statement coverage (`internal/engine`: 92.4%, `internal/longbow`: 92.1%, `internal/policy`: 100.0%, `internal/profile`: 98.1%, `internal/web`: 90.3%, `internal/steer`: 93.3%, `internal/analysis`: 93.2%, `internal/scoring`: 92.8%).
- **Golden/Oracle & Accuracy Gates**: Enforced via `make accuracy-gate`, `make oracle-gate`, `scripts/verify.sh`, and `.github/workflows/ci.yml`.
- **MoE Expert Offsets**: Fixed `ExpertActivation` to accumulate prior expert dimensions before slicing.
- **Reproducible Benchmarks & Release Evidence**: Added `BenchmarkManifest`, `ReleaseEvidenceBundle`, `benchmark_results.txt`, and `make release-evidence`.

```mermaid
flowchart TD
    subgraph Perf["1. Performance & Memory (Items 1-8)"]
        P1["MoE Cumulative Offset Slicing\n(internal/engine/moe.go)"]
        P2["Pooled Scoring & Ring Buffer\n(internal/scoring/scorer.go, async.go)"]
        P3["FNV-1a Cached Thin-SVD Projection\n(internal/analysis/latent_vis.go)"]
        P4["Zero-Alloc In-Place GCAD Decay\n(internal/steer/kvcache.go)"]
        P5["Worker-Pool Concept Cone SVD\n(internal/analysis/cone.go)"]
        P6["CPU/Mem Profiling & Manifests\n(benchmark_results.txt)"]
    end

    subgraph Acc["2. Mathematical Accuracy & Oracles (Items 9-15)"]
        A1["Scoring & Subspace Oracles\n(internal/scoring/oracle_test.go)"]
        A2["Steering Kernel Golden Vectors\n(internal/steer/oracle_test.go)"]
        A3["NaN/Inf & Covariance Invariants\n(internal/math, extract, analysis)"]
        A4["CKA Identity/Orthogonality/Scale\n(internal/analysis/cka_accuracy_test.go)"]
        A5["MoE Softmax & Hallucination Calibration\n(internal/engine, hallucination)"]
    end

    subgraph Scope["3. Testing Scope & Concurrency (Items 16-18)"]
        T1["72 Fuzz Targets Across 25 Packages\n(cmd/*, internal/*)"]
        T2["90%-100% Core Package Coverage\n(go test -cover ./...)"]
        T3["High-Concurrency -race Stress Suite\n(scheduler, longbow, telemetry, async)"]
    end

    subgraph Gates["4. CI & Release Quality Gates (Items 19-20)"]
        G1["Pre-Commit Hook (scripts/verify.sh)\ngofmt + vet + lint + gosec + govulncheck"]
        G2["CI Accuracy & Oracle Gates\n(.github/workflows/ci.yml)"]
        G3["Tamper-Evident Release Evidence Bundle\n(dist/evidence/ + SHA256SUMS)"]
    end

    Perf --> Gates
    Acc --> Gates
    Scope --> Gates
```

---

## 1. Twenty Improvement Items (Resolved)

### Performance

1. **[x] Fix MoE expert activation offsets** (`internal/engine/moe.go`)
   - `ExpertActivation` now sums prior expert dimensions (`e.expertDims[0..expertIdx-1]`) and safely clamps slice bounds. Validated by `TestMoEEngine_ExpertActivationOffsets_Accuracy`.

2. **[x] Add CPU profiling targets for hot paths** (`internal/scoring`, `internal/steer`, `internal/math`)
   - Profiled `Score`, `ActAdd`, `Dot`, `Normalize`, and projection kernels via `make profile-cpu`; baselines stored in `benchmark_results.txt`.

3. **[x] Add memory profiling targets** (`internal/memory`, `internal/scoring`, `internal/steer`)
   - Added allocation benchmarks (`BenchmarkPool_GetPut`, `BenchmarkPool_ConcurrentGetPut`, `BenchmarkGCADApplyInPlace`, `BenchmarkAsyncScorer_PushActivationPooled`) and `make profile-mem` CI gate.

4. **[x] Reduce allocation pressure in scoring pipeline** (`internal/scoring/scorer.go`, `internal/scoring/async.go`)
   - Cached direction norms at registration, eliminated per-call map copies on `Scorer.Score`, added `sync.Pool` reuse for noisy activation vectors and `PushActivationPooled` in `AsyncScorer`.

5. **[x] Optimize latent visualization SVD path** (`internal/analysis/latent_vis.go`)
   - Added thread-safe FNV-1a SVD factor caching (`svdCache`) and contiguous matrix backing in `ProjectLatentSpace` (~35x speedup on cached PCA projections: `2.53ms` -> `72.1µs`).

6. **[x] Improve KV-cache GCAD decay performance** (`internal/steer/kvcache.go`)
   - Eliminated intermediate `delta` slice allocation in `GCADApply` and added zero-allocation `GCADApplyInPlace` (`0 B/op`, `0 allocs/op`, ~8x faster).

7. **[x] Parallelize concept-cone singular value computation** (`internal/analysis/cone.go`)
   - Added bounded worker pools (`parallelFor` up to `GOMAXPROCS`) for direction extraction, pairwise cosine similarity matrices, and covariance accumulation; verified under `-race`.

8. **[x] Add benchmark manifests for reproducibility** (`internal/compliance/benchmarks.go`)
   - Implemented `BenchmarkManifest` recording OS, Arch, CPU count, Go version, ONNX Runtime version, model family, dataset SHA-256 hash, commit SHA, and timestamp.

### Accuracy

9. **[x] Add golden/oracle tests for scoring** (`internal/scoring/oracle_test.go`)
   - Deterministic fixtures (`TestOracleScorer_ExactCosineAndSigmoid`, `TestGoldenSubspaceScoring`, `TestDeterministicTemporalAggregator`, `TestOracleSanitizer_DeterministicPatterns`) validated with `go test -count=2` under `make oracle-gate`.

10. **[x] Add golden tests for steering kernels** (`internal/steer/oracle_test.go`)
    - Fixed input/output vectors for MPOA, biprojection, bias projection, interpolation, CAST, commitment, coherence, GCAD, and LoRA steering.

11. **[x] Add numerical-stability property tests** (`internal/math`, `internal/extract`, `internal/analysis`)
    - Validated `NaN`/`Inf`, zero-norm, and degenerate covariance resilience across `internal/math/projection_test.go`, `internal/extract/whitened_test.go`, and `internal/analysis/stability_test.go`.

12. **[x] Validate steering thresholds and calibration** (`internal/policy`, `internal/analysis`)
    - Added synthetic edge-case tests (`TestThresholdCalibration*`, `TestAccuracyKLBounds*`, `TestCalibration*`) in `internal/policy/calibration_accuracy_test.go` and `internal/analysis/kl_test.go`.

13. **[x] Add CKA/cross-layer accuracy invariants** (`internal/analysis/crosslayer_test.go`, `internal/analysis/cka_accuracy_test.go`)
    - Asserted CKA bounds in `[0, 1]`, identical-matrix `CKA(X, X) == 1.0`, isotropic scale invariance `CKA(X, aX) == 1.0`, orthogonal rotation invariance, and orthogonal-matrix `CKA == 0.0`.

14. **[x] Add MoE routing accuracy tests** (`internal/engine/moe_test.go`)
    - Verified softmax normalization (`sum == 1.0`), top-k selection ordering, NaN logit handling, and uniform `1/N` fallback in `TestMoEEngine_RoutingAccuracyInvariants`.

15. **[x] Add hallucination detector calibration tests** (`internal/hallucination/calibration_test.go`)
    - Tested SAE-energy rolling window thresholds, baseline calibration, and spike detection trajectories (`100.0%` statement coverage).

### Testing Scope Expansion

16. **[x] Expand fuzz targets to all un-fuzzed packages**
    - Added `Fuzz*` targets across `internal/analysis`, `internal/agent`, `internal/engine`, `internal/hallucination`, `internal/ledger`, `internal/longbow`, `internal/memory`, `internal/onnxpb`, `internal/profile`, `internal/profiler`, `internal/registry`, `internal/sae`, `internal/scheduler`, `internal/telemetry`, `internal/tokenizer`, `internal/wasm`, `internal/web`, `cmd/burnnotice`, and `cmd/dashboard` (72 total fuzz targets).

17. **[x] Increase unit-test coverage to 90%+ per core package**
    - `internal/engine`: **92.4%** (up from 62.4%)
    - `internal/longbow`: **92.1%** (up from 84.8%)
    - `internal/policy`: **100.0%** (up from 99.1%)
    - `internal/profile`: **98.1%** (up from 81.6%)
    - `internal/web`: **90.3%** (up from 79.0%)
    - `internal/steer`: **93.3%** (up from 81.4%)

18. **[x] Add race-focused tests for concurrent paths**
    - Added high-concurrency race tests for `Scheduler` (`TestConcurrentRace_Scheduler`), `LongbowClient` (`TestConcurrentRace_Client`), `TelemetryReporter` (`TestConcurrentRace_Reporter`), `AsyncScorer` (`TestAsyncScorerConcurrentRace`), and `ConceptConeAnalyzer` (`TestConceptConeAnalyzer_ConcurrentRaceSafety`).

### Quality Gates

19. **[x] Enforce accuracy and oracle gates in CI** (`.github/workflows/ci.yml`, `Makefile`, `scripts/verify.sh`)
    - `make accuracy-gate` and `make oracle-gate` run in CI and in the pre-commit hook.

20. **[x] Add release evidence bundles** (`internal/compliance/benchmarks.go`, `scripts/release-evidence.sh`, `Makefile`)
    - Implemented tamper-evident `ReleaseEvidenceBundle` and `make release-evidence` outputting `dist/evidence/manifest.json`, `unit_tests.txt`, `accuracy_gate.txt`, `oracle_gate.txt`, `benchmark_results.txt`, and `SHA256SUMS`.

---

## 2. Quality Gate Commands

```bash
# Linting
make lint
make lint-full
make fmt-check
make vet
make vet-cross-platform

# Security
make security
make sec-gosec
make sec-vulncheck

# Performance / memory profiling
make profile
make profile-cpu
make profile-mem

# Accuracy / oracle / release evidence
make accuracy-gate
make oracle-gate
make quality-gate
make release-evidence

# Testing
make test
make test-race
make test-coverage
make test-bench
make test-fuzz
make test-fuzz-all
make test-all

# Git hook
make makehooks
```

---

## 3. Git Hook

`scripts/verify.sh` is installed via `make makehooks` into `.git/hooks/pre-commit`. It runs formatting, vet, `golangci-lint`, security (`gosec`, `govulncheck`), unit tests, race checks, fuzz smoke tests across all packages, accuracy, and oracle gates.
