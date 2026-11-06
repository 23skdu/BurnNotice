# BurnNotice Roadmap: Performance, Accuracy, and Quality Gates

> **Status**: Active
> **Last Updated**: 2026-11-05
> **Scope**: 20 prioritized improvements for performance, accuracy, unit-test coverage, fuzz coverage, and CI quality gates.

---

## 0. Current Baseline (Deep Code Analysis)

BurnNotice has a strong, well-factored Go codebase with broad package-level tests, fuzz targets, benchmarks, and a race-clean suite (`go test -race ./...` passes). Static analysis currently reports zero `gosec` issues and zero `govulncheck` vulnerabilities. The most important remaining gaps are:

- Inconsistent formatting across many files (resolved via `gofmt -s -w .`).
- `golangci-lint` reports pre-existing findings (errcheck/staticcheck/unused) that should be driven to zero.
- Several performance- and accuracy-critical packages have no fuzz coverage (`internal/analysis`, `internal/agent`, `internal/hallucination`, `internal/ledger`, `internal/longbow`, `internal/memory`, `internal/onnxpb`, `internal/profile`, `internal/profiler`, `internal/registry`, `internal/sae`, `internal/scheduler`, `internal/telemetry`, `internal/tokenizer`, `internal/wasm`, `internal/web`, `cmd/*`).
- No golden/oracle tests exist for deterministic scoring, steering, and math kernels.
- MoE expert extraction has an offset bug: `ExpertActivation` always reads from the start of the activation vector instead of the expert's partition.
- Benchmark claims are not tied to reproducible environment/dataset manifests.

---

## 1. Twenty Improvement Items

### Performance

1. **Fix MoE expert activation offsets** (`internal/engine/moe.go`)
   - `ExpertActivation` must sum prior expert dimensions before slicing. Add unit tests for expert index boundaries and multi-expert offsets.

2. **Add CPU profiling targets for hot paths** (`internal/scoring`, `internal/steer`, `internal/math`)
   - Profile `Score`, `ActAdd`, `Dot`, `Normalize`, and projection kernels with `make profile-cpu`; store baselines in `benchmark_results.txt`.

3. **Add memory profiling targets** (`internal/memory`, `internal/scoring`, `internal/steer`)
   - Track allocations in activation buffer pools and steering application; gate regressions in CI.

4. **Reduce allocation pressure in scoring pipeline**
   - Reuse pooled buffers in `internal/scoring` async ring-buffer paths; add `-benchmem` comparisons.

5. **Optimize latent visualization SVD path**
   - Cache intermediate SVD factors in `internal/analysis/latent_vis.go`; benchmark Thin SVD projection.

6. **Improve KV-cache GCAD decay performance**
   - Replace repeated slice copies with in-place decay in `internal/steer/kvcache.go`; add benchmarks.

7. **Parallelize concept-cone singular value computation**
   - Use worker pools for pairwise cosine matrices in `internal/analysis/cone.go`; verify race safety.

8. **Add benchmark manifests for reproducibility**
   - Record OS, CPU, ONNX Runtime, model family, dataset hash, and commit SHA for every published benchmark.

### Accuracy

9. **Add golden/oracle tests for scoring**
   - Create deterministic fixtures in `internal/scoring` validated with `go test -count=2` under `make oracle-gate`.

10. **Add golden tests for steering kernels**
    - Fixed input/output vectors for MPOA, bias projection, interpolation, and LoRA steering in `internal/steer`.

11. **Add numerical-stability property tests**
    - NaN/Inf, zero-norm, and degenerate covariance handling in `internal/math`, `internal/extract`, and `internal/analysis`.

12. **Validate steering thresholds and calibration**
    - Synthetic edge cases for threshold, KL, and angular-drift bounds in `internal/policy` and `internal/analysis`.

13. **Add CKA/cross-layer accuracy invariants**
    - Assert CKA ranges, identical-matrix CKA = 1.0, orthogonal-matrix CKA = 0 in `internal/analysis`.

14. **Add MoE routing accuracy tests**
    - Softmax normalization, top-k selection, and uniform fallback in `internal/engine/moe.go`.

15. **Add hallucination detector calibration tests**
    - SAE-energy rolling window thresholds and spike detection in `internal/hallucination`.

### Testing Scope Expansion

16. **Expand fuzz targets to all un-fuzzed packages**
    - Add `Fuzz*` targets for `internal/analysis`, `internal/agent`, `internal/hallucination`, `internal/ledger`, `internal/longbow`, `internal/memory`, `internal/onnxpb`, `internal/profile`, `internal/profiler`, `internal/registry`, `internal/sae`, `internal/scheduler`, `internal/telemetry`, `internal/tokenizer`, `internal/wasm`, `internal/web`, and `cmd/*`.

17. **Increase unit-test coverage to 90%+ per core package**
    - Focus on `internal/engine`, `internal/longbow`, `internal/policy`, `internal/profile`, and `internal/web`.

18. **Add race-focused tests for concurrent paths**
    - Scheduler, longbow client, telemetry reporter, and async scoring ring buffer.

### Quality Gates

19. **Enforce accuracy and oracle gates in CI**
    - `make accuracy-gate` and `make oracle-gate` must run on every change touching `internal/`.

20. **Add release evidence bundles**
    - Ship commit SHA, dataset hash, environment manifest, benchmark artifacts, and test results with each release.

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

# Accuracy / oracle
make accuracy-gate
make oracle-gate
make quality-gate

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

`scripts/verify.sh` is installed via `make makehooks` into `.git/hooks/pre-commit`. It runs formatting, vet, security, tests, race checks, fuzz smoke tests, accuracy, and oracle gates.

---

## 4. Tracking

Each item above should be tracked as an individual issue/PR with:
- Reproducible benchmark or test artifact
- Before/after metrics where applicable
- Updated documentation in `docs/testing.md`
