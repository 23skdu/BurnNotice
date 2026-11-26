# BurnNotice Implementation Roadmap: 10-Step Improvement Plan

> **Status**: In Progress — Improvement Phase
> **Last Updated**: 2026-11-26
> **Version**: 0.9.2
> **Scope**: Ten improvement steps identified through gap analysis of the completed v0.8/v0.9 improvement plan and formal verification work.

---

## Gap Analysis Summary (2026-11-26)

| # | Area | Current State | Gap |
|---|------|---------------|-----|
| 1 | Lean proof completeness | 2 `sorry` remain (KL, DoRA now done); CKA/Covariance/Biprojection are stubs | Substantive proofs needed for CKA bounds, covariance PSD construction, biprojection idempotence |
| 2 | Test coverage | Internal packages 90-95%; cmd/burnnotice 85.8% | Push internal packages to 95%+; improve cmd coverage |
| 3 | E2E testing | 8 Playwright spec files exist | Never verified running; may have stale selectors or broken flows |
| 4 | Performance regression | 50% timing tolerance, 10% memory tolerance | Tolerance too loose to catch real regressions; noisy single-run methodology |
| 5 | Documentation accuracy | Fuzz count says 72 (actual 76); metrics says 50+ (actual 55) | Minor inaccuracies in README/testing.md |
| 6 | Security hygiene | 56 `#nosec` annotations, some with minimal justification | Reduce suppressions by fixing underlying patterns where possible |
| 7 | Observability | 55 metrics; no per-layer latency breakdown | Missing granular profiling metrics for steering operations |
| 8 | Error handling | CLI errors are basic; some paths lack recovery | Improve error messages, add retry logic, better user feedback |
| 9 | Cross-platform | Metal/CUDA kernels exist but untested in CI | No CI coverage for GPU kernel compilation |
| 10 | Code quality | 76 fuzz targets; some packages have thin fuzz coverage | Expand fuzz targets for edge cases in scoring, steering, and analysis |

---

## Step 1 — Complete Lean Proof Placeholders

**Goal**: Eliminate all `sorry` placeholders and strengthen stub proofs into substantive theorems.

- [x] `KL.lean`: Complete Gibbs' inequality proof via `log(x) ≤ x - 1`.
- [x] `DoRA.lean`: Complete row-norm preservation via scaling lemma.
- [ ] `CKA.lean`: Prove CKA ∈ [0,1] from Cauchy-Schwarz and centering properties (not just restate hypothesis).
- [ ] `Covariance.lean`: Prove covariance matrix is symmetric and PSD from definition (not just restate).
- [ ] `Biprojection.lean`: Prove biprojection idempotence from orthogonal projection properties.
- [ ] `Softmax.lean`: Complete hdenom positivity proof (sum of exponentials > 0).
- [ ] Add proofs for GCAD exponential decay monotonicity (already partially done, tighten).
- [ ] Add proofs for CKA (Centered Kernel Alignment) tightness: CKA = 1 iff linearly dependent.
- [ ] Add proofs for Procrustes alignment optimality.
- [ ] Verify `lake build` passes with zero `sorry` and zero warnings.

**Acceptance criteria**: `lake build` reports 0 `sorry` warnings; all theorems are substantive (not tautological).

---

## Step 2 — Push Test Coverage to 95%+

**Goal**: Raise internal package coverage from ~90-94% to 95%+ by covering remaining error branches.

- [ ] `internal/steer` (93.9%): cover error paths in MPOA, biprojection, GCAD decay, and crescendo detector.
- [ ] `internal/analysis` (93.1%): cover CKA computation, concept cone SVD, and ouroboros edge cases.
- [ ] `internal/scoring` (92.8%): cover async scorer error paths, pooled buffer reuse, and noise injection.
- [ ] `internal/sae` (92.4%): cover loader error paths, absorption diagnostics edge cases.
- [ ] `internal/engine` (91.8%): cover MoE routing edge cases, ONNX session fallback.
- [ ] `internal/extract` (92.3%): cover transfer learning and direction extraction error paths.
- [ ] `internal/agent` (91.8%): cover middleware, governor, and MCP integration error paths.
- [ ] `internal/wasm` (92.1%): cover WASM policy evaluation error paths.
- [ ] `internal/longbow` (91.8%): cover fallback mode, query, and search error paths.
- [ ] `internal/policy` (90.1%): cover OPA evaluation, semantic gating, and kill switch error paths.
- [ ] `internal/profiler` (90.4%): cover manager edge cases, job cancellation, and oversight profiling.
- [ ] `internal/web` (91.7%): cover remaining handler error paths and SSE edge cases.
- [ ] `internal/zjson` (90.0%): cover token parser edge cases and encode error paths.
- [ ] `cmd/burnnotice` (85.8%): cover CLI command error branches and flag validation.
- [ ] Raise coverage gate thresholds to 95% for internal packages.

**Acceptance criteria**: All internal packages ≥95%; cmd/burnnotice ≥90%; coverage gate updated.

---

## Step 3 — E2E Test Verification and Repair

**Goal**: Run Playwright E2E tests, fix broken selectors/flows, and add coverage for new features.

- [ ] Install Playwright browsers and dependencies (`npm ci` in `e2e/`).
- [ ] Run `npx playwright test` and identify failures.
- [ ] Fix broken selectors in `dashboard.spec.ts` (SSE updates, latent space SVG).
- [ ] Fix broken selectors in `profiler.spec.ts` (job submission, cancellation).
- [ ] Fix broken selectors in `monitor.spec.ts` (real-time score updates).
- [ ] Fix broken selectors in `alerts.spec.ts` (CSV export, alert filtering).
- [ ] Fix broken selectors in `policies.spec.ts` (threshold updates, OPA rules).
- [ ] Fix broken selectors in `visualizations.spec.ts` (SVG rendering, heatmap).
- [ ] Fix broken selectors in `navigation.spec.ts` (routing, deep links).
- [ ] Fix broken selectors in `model-profile.spec.ts` (profile detail, latent space).
- [ ] Add E2E tests for DoRA fine-tuning CLI output.
- [ ] Add E2E tests for crescendo detection dashboard indicators.
- [ ] Add E2E tests for SAE feature absorption visualization.
- [ ] Add E2E test step to CI workflow (with browser caching).
- [ ] Document E2E test setup in `docs/testing.md`.

**Acceptance criteria**: All 8+ spec files pass; E2E gate in CI; no flaky tests.

---

## Step 4 — Performance Regression Tightening

**Goal**: Reduce false positives/negatives in benchmark regression detection.

- [ ] Run benchmarks 3x and take median instead of single run.
- [ ] Reduce timing tolerance from 50% to 15% using median of 3 runs.
- [ ] Add allocation count regression as hard gate (already done, verify).
- [ ] Add memory (B/op) regression as hard gate with 10% tolerance.
- [ ] Pin CPU governor to performance mode in CI for stable benchmarks.
- [ ] Add benchmark result comparison to PR comments (GitHub Actions).
- [ ] Document rebaseline procedure in `docs/testing.md`.
- [ ] Add `make bench-regression` target.
- [ ] Verify no false positives over 5 consecutive runs.

**Acceptance criteria**: Bench-regression passes consistently; tolerance tightened to 15%.

---

## Step 5 — Documentation Accuracy

**Goal**: Fix all documentation inaccuracies and keep docs in sync with code.

- [ ] Update README.md fuzz count: 72 → 76 targets, 25 → 27 packages.
- [ ] Update README.md metrics count: 50+ → 55+.
- [ ] Update testing.md mermaid diagram fuzz count.
- [ ] Update testing.md coverage claims to reflect actual per-package numbers.
- [ ] Add `docs/formalproof.md` reference to README documentation table (done).
- [ ] Verify all file paths referenced in docs exist.
- [ ] Verify all CLI flags documented match `cmd/burnnotice/main.go`.
- [ ] Verify all metrics documented match `internal/metrics/metrics.go`.
- [ ] Add changelog entry for v0.9.2 improvements.
- [ ] Run markdownlint on all docs and fix violations.

**Acceptance criteria**: All documentation accurate; markdownlint clean (except pre-existing).

---

## Step 6 — Security Hardening

**Goal**: Reduce `#nosec` suppressions by fixing underlying patterns.

- [ ] Audit all 56 `#nosec` annotations for necessity.
- [ ] Replace `math/rand` with `math/rand/v2` in scoring (already v2, verify usage).
- [ ] Add input validation for G115 conversion sites (longbow, converter, native).
- [ ] Add path traversal checks for G304 file read sites.
- [ ] Add integer overflow checks for G115 tensor dimension conversions.
- [ ] Reduce G103 (unsafe) suppressions by using safe alternatives where possible.
- [ ] Add `#nosec` justification comments where suppression is truly necessary.
- [ ] Run gosec with `--confidence=high` and fix all high-confidence issues.
- [ ] Add security-focused fuzz targets for file I/O and path handling.
- [ ] Document security model in `docs/architecture.md`.

**Acceptance criteria**: gosec reports 0 issues; nosec count reduced by ≥25%.

---

## Step 7 — Observability Enhancement

**Goal**: Add granular profiling metrics for steering and scoring operations.

- [ ] Add per-layer scoring latency histogram (`burnnotice_scoring_layer_duration_seconds`).
- [ ] Add per-layer steering intervention latency histogram.
- [ ] Add GCAD decay computation latency metric.
- [ ] Add crescendo detector step latency metric.
- [ ] Add DoRA hardening pass duration metric.
- [ ] Add SAE feature extraction latency metric.
- [ ] Add KV-cache re-anchor latency metric.
- [ ] Add metric for number of active SSE connections (already exists, verify).
- [ ] Add metric for profile manager queue depth.
- [ ] Update Grafana dashboard with new metrics.
- [ ] Update `docs/metrics.md` with new metric descriptions.

**Acceptance criteria**: 65+ metrics; Grafana dashboard updated.

---

## Step 8 — Error Handling and Resilience

**Goal**: Improve CLI error messages, add retry logic, and better user feedback.

- [ ] Add retry logic with exponential backoff for network operations (downloader).
- [ ] Add retry logic for ONNX Runtime initialization failures.
- [ ] Improve CLI error messages with actionable suggestions.
- [ ] Add `--verbose` flag for detailed error output.
- [ ] Add graceful degradation when OPA sidecar is unavailable.
- [ ] Add graceful degradation when Longbow mesh is unreachable.
- [ ] Add timeout configuration for all network operations.
- [ ] Add circuit breaker for repeated failures.
- [ ] Add structured error types with error codes.
- [ ] Add error recovery documentation in `docs/cli.md`.

**Acceptance criteria**: All network operations have retry + timeout; error messages are actionable.

---

## Step 9 — Cross-Platform GPU Support

**Goal**: Improve Metal/CUDA kernel compilation and testing in CI.

- [ ] Add macOS (Metal) build target to CI (GitHub Actions macOS runner).
- [ ] Add CUDA kernel compilation check (nvcc) in CI (with GPU runner or mock).
- [ ] Add Metal shader compilation check (xcrun) in CI.
- [ ] Add cross-platform test matrix (linux/amd64, linux/arm64, darwin/amd64, darwin/arm64, windows/amd64).
- [ ] Add GPU kernel unit tests (CPU fallback verification).
- [ ] Document GPU setup requirements in `docs/architecture.md`.
- [ ] Add Docker image for CUDA (already exists, verify).
- [ ] Add Docker image for Metal (already exists, verify).
- [ ] Test WASM execution on all supported platforms.
- [ ] Add platform-specific performance baselines.

**Acceptance criteria**: CI tests all 5 platform targets; GPU kernels compile on all platforms.

---

## Step 10 — Code Quality and Fuzz Expansion

**Goal**: Expand fuzz targets and improve code quality.

- [ ] Add fuzz targets for `internal/scoring` (scorer, async scorer, normalizer).
- [ ] Add fuzz targets for `internal/steer` (biprojection, CAST, commitment, coherence).
- [ ] Add fuzz targets for `internal/analysis` (CKA, concept cone, latent vis).
- [ ] Add fuzz targets for `internal/engine` (MoE routing, ONNX parsing).
- [ ] Add fuzz targets for `internal/hallucination` (energy, family features).
- [ ] Add property-based tests for cosine similarity bounds.
- [ ] Add property-based tests for DoRA KL ceiling.
- [ ] Add property-based tests for SAE metric ranges.
- [ ] Add property-based tests for angular drift bound.
- [ ] Run `go test -shuffle=on` in CI to catch order-dependent tests.
- [ ] Run `golangci-lint` with additional linters (gocyclo, gocognit, funlen).
- [ ] Add code complexity thresholds to CI.

**Acceptance criteria**: 90+ fuzz targets; complexity thresholds enforced.

---

## Quality Gate Commands

```bash
bash scripts/verify.sh 1x
make lint
make fmt-check
make vet
make vet-cross-platform
make security
make accuracy-gate
make oracle-gate
make quality-gate
make release-evidence
make test
make test-race
make test-coverage
make test-coverage-gate
make test-fuzz
make bench-regression
cd lean && lake build
```
