# BurnNotice Implementation Roadmap: 10-Step Improvement Plan

> **Status**: In Progress — Improvement Phase
> **Last Updated**: 2026-11-27
> **Version**: 0.9.3
> **Scope**: Ten improvement steps identified through deep gap analysis of the completed v0.8/v0.9 improvement plan, formal verification work, and full quality gate validation.

---

## Deep Gap Analysis (2026-11-27)

### Validation Results

| Gate | Status | Details |
|------|--------|---------|
| `go build ./...` | ✅ Pass | All packages compile |
| `go vet ./...` | ✅ Pass | No vet issues |
| `go test ./...` | ✅ Pass | All 34 packages pass |
| `go test -race ./...` | ✅ Pass | No race conditions |
| `gosec ./...` | ✅ 0 issues | 56 nosec annotations (7 G103, 18 G115, 1 G202, 25 G304, 2 G404, 1 G704, 2 G706) |
| `govulncheck ./...` | ✅ Pass | No vulnerabilities |
| `golangci-lint ./...` | ✅ 0 issues | All linters pass |
| `make accuracy-gate` | ✅ Pass | All accuracy invariants hold |
| `make oracle-gate` | ✅ Pass | All golden oracles deterministic |
| `make test-coverage-gate` | ✅ Pass | All packages meet floors |
| `make test-fuzz` | ✅ Pass | 76 targets across 27 packages |
| `make vet-cross-platform` | ✅ Pass | All 4 CGO_ENABLED=0 targets |
| `helm lint` | ✅ Pass | 0 failures |
| `make release-evidence` | ✅ Pass | Bundle generated |
| `cd lean && lake build` | ✅ Pass | 15 modules, 0 errors |
| Playwright E2E | ⚠️ Untested | 8 spec files exist, never verified running |

### Identified Gaps

| # | Area | Current State | Gap | Severity |
|---|------|---------------|-----|----------|
| 1 | Lean proof completeness | 0 `sorry` in project; CKA/Covariance/Biprojection are stubs (restate hypotheses) | Substantive proofs needed | Medium |
| 2 | Test coverage | Internal 90-95%; cmd/burnnotice 85.8% | Push to 95%+; 6 packages lack fuzz targets | Medium |
| 3 | E2E testing | 8 Playwright specs exist | Never verified running; likely stale selectors | High |
| 4 | Performance regression | 50% timing tolerance, single-run methodology | Too loose; noisy; allocation-only hard gate | Medium |
| 5 | Documentation accuracy | README says 72-73 fuzz (actual 76); testing.md says 72 (actual 76) | Minor inaccuracies | Low |
| 6 | Security hygiene | 56 nosec annotations (25 G304, 18 G115, 7 G103) | Could reduce by ~25% with input validation | Medium |
| 7 | Observability | 55 metrics; no per-layer latency breakdown | Missing granular profiling metrics | Low |
| 8 | Error handling | CLI errors are basic; no retry logic for network ops | Improve UX and resilience | Medium |
| 9 | Cross-platform | Metal/CUDA kernels untested in CI | No CI coverage for GPU compilation | Low |
| 10 | Code quality | 76 fuzz targets; 6 packages have none | Expand fuzz coverage; add complexity thresholds | Medium |

### Packages Without Fuzz Targets
- `internal/checkpoint`
- `internal/compliance`
- `internal/datasets`
- `internal/downloader`
- `internal/engine/native`
- `internal/policydetect`

### Coverage Distribution
- **95%+**: checkpoint (100), compliance (94.0), datasets, hallucination (100), ledger (96.0), math (96.6), memory (100), metrics (96.3), policydetect (98.1), profile (98.1), scheduler (100), tokenizer (100), converter (96.1), registry (100)
- **90-95%**: agent (91.8), analysis (93.1), downloader (91.0), engine (91.8), extract (92.3), longbow (91.8), policy (90.1), profiler (90.4), sae (92.4), scoring (92.8), steer (93.9), telemetry (94.4), wasm (92.1), web (91.7), zjson (90.0)
- **Below 90%**: cmd/burnnotice (85.8%)

---

## Step 1 — Complete Lean Proof Substantiation

**Goal**: Eliminate stub proofs and strengthen all theorems into substantive, non-tautological results.

- [x] `KL.lean`: Complete Gibbs' inequality proof via `log(x) ≤ x - 1`.
- [x] `DoRA.lean`: Complete row-norm preservation via scaling lemma.
- [x] `Cosine.lean`: Complete cosine similarity bounds via Cauchy-Schwarz.
- [x] `Softmax.lean`: Complete softmax sum-to-1 proof.
- [x] `Drift.lean`: Complete angular drift bound and re-anchor reset.
- [x] `GCAD.lean`: Complete exponential decay monotonicity.
- [ ] `CKA.lean`: Prove CKA ∈ [0,1] from Cauchy-Schwarz and centering (not just restate hypothesis).
- [ ] `Covariance.lean`: Prove covariance matrix is symmetric and PSD from definition.
- [ ] `Biprojection.lean`: Prove biprojection idempotence from orthogonal projection properties.
- [ ] Add proofs for CKA tightness (CKA = 1 iff linearly dependent).
- [ ] Add proofs for Procrustes alignment optimality.
- [ ] Add proofs for covariance matrix symmetry and PSD construction.
- [ ] Add proofs for biprojection idempotence property.
- [ ] Verify `lake build` passes with zero `sorry` and zero warnings.
- [ ] Add CI step (`lake build`) that fails on proof errors.

**Acceptance criteria**: `lake build` reports 0 `sorry` and 0 warnings; all theorems substantive.

---

## Step 2 — Push Test Coverage to 95%+

**Goal**: Raise all internal packages from ~90-94% to 95%+ by covering remaining error branches and edge cases.

- [ ] `internal/steer` (93.9%): cover MPOA error paths, biprojection edge cases, GCAD decay boundaries, crescendo detector edge cases.
- [ ] `internal/analysis` (93.1%): cover CKA computation edge cases, concept cone SVD degenerate inputs, ouroboros boundary conditions.
- [ ] `internal/scoring` (92.8%): cover async scorer error paths, pooled buffer exhaustion, noise injection edge cases.
- [ ] `internal/sae` (92.4%): cover loader error paths, absorption diagnostics with empty features, alignment edge cases.
- [ ] `internal/engine` (91.8%): cover MoE routing with zero experts, ONNX session fallback, native kernel edge cases.
- [ ] `internal/extract` (92.3%): cover transfer learning with mismatched dimensions, direction extraction failures.
- [ ] `internal/agent` (91.8%): cover middleware error paths, governor edge cases, MCP integration failures.
- [ ] `internal/wasm` (92.1%): cover WASM policy evaluation with malformed bundles, OPA sidecar failures.
- [ ] `internal/longbow` (91.8%): cover fallback mode transitions, query with empty store, search dimension mismatches.
- [ ] `internal/policy` (90.1%): cover OPA evaluation with invalid rules, semantic gating edge cases, kill switch activation.
- [ ] `internal/profiler` (90.4%): cover manager with zero workers, job cancellation races, oversight profiling failures.
- [ ] `internal/web` (91.7%): cover remaining handler error paths, SSE client disconnect, SVG generation failures.
- [ ] `internal/zjson` (90.0%): cover token parser with malformed input, encode error paths, buffer pool exhaustion.
- [ ] `cmd/burnnotice` (85.8%): cover CLI command error branches, flag validation, rollback paths.
- [ ] Raise coverage gate thresholds: internal 95%, cmd 90%.
- [ ] Add coverage trend tracking to CI (fail on regression).

**Acceptance criteria**: All internal packages ≥95%; cmd/burnnotice ≥90%; coverage gate updated and enforced.

---

## Step 3 — E2E Test Verification and Repair

**Goal**: Run Playwright E2E tests, fix broken selectors/flows, and add coverage for new features.

- [ ] Install Playwright browsers and dependencies (`npm ci` in `e2e/`).
- [ ] Run `npx playwright test` and catalog all failures.
- [ ] Fix broken selectors in `dashboard.spec.ts` (SSE updates, latent space SVG, metric cards).
- [ ] Fix broken selectors in `profiler.spec.ts` (job submission, cancellation, progress).
- [ ] Fix broken selectors in `monitor.spec.ts` (real-time score updates, drift indicators).
- [ ] Fix broken selectors in `alerts.spec.ts` (CSV export, alert filtering, pagination).
- [ ] Fix broken selectors in `policies.spec.ts` (threshold updates, OPA rules, kill switch).
- [ ] Fix broken selectors in `visualizations.spec.ts` (SVG rendering, heatmap, latent space).
- [ ] Fix broken selectors in `navigation.spec.ts` (routing, deep links, responsive layout).
- [ ] Fix broken selectors in `model-profile.spec.ts` (profile detail, SAE features, latent space).
- [ ] Add E2E tests for DoRA fine-tuning CLI output verification.
- [ ] Add E2E tests for crescendo detection dashboard indicators.
- [ ] Add E2E tests for SAE feature absorption visualization.
- [ ] Add E2E test step to CI workflow with browser caching.
- [ ] Document E2E test setup and troubleshooting in `docs/testing.md`.
- [ ] Fix any flaky tests (add retries, stabilize selectors).

**Acceptance criteria**: All 8+ spec files pass; E2E gate in CI; zero flaky tests over 5 runs.

---

## Step 4 — Performance Regression Tightening

**Goal**: Reduce false positives/negatives in benchmark regression detection.

- [ ] Run benchmarks 3x and take median instead of single run.
- [ ] Reduce timing tolerance from 50% to 15% using median of 3 runs.
- [ ] Add allocation count regression as hard gate (verify existing).
- [ ] Add memory (B/op) regression as hard gate with 10% tolerance.
- [ ] Pin CPU governor to performance mode in CI for stable benchmarks.
- [ ] Add benchmark result comparison to PR comments (GitHub Actions).
- [ ] Document rebaseline procedure in `docs/testing.md`.
- [ ] Add `make bench-regression` target (verify existing).
- [ ] Verify no false positives over 5 consecutive runs.
- [ ] Add per-benchmark timeout to prevent hung benchmarks.

**Acceptance criteria**: Bench-regression passes consistently; tolerance tightened to 15%; zero false positives.

---

## Step 5 — Documentation Accuracy

**Goal**: Fix all documentation inaccuracies and keep docs in sync with code.

- [ ] Update README.md fuzz count: 72 → 76 targets, 25 → 27 packages.
- [ ] Update README.md metrics count: 50+ → 55+.
- [ ] Update testing.md mermaid diagram fuzz count: 72 → 76.
- [ ] Update testing.md coverage claims to reflect actual per-package numbers.
- [ ] Verify all file paths referenced in docs exist.
- [ ] Verify all CLI flags documented match `cmd/burnnotice/main.go`.
- [ ] Verify all metrics documented match `internal/metrics/metrics.go`.
- [ ] Add changelog entry for v0.9.3 improvements.
- [ ] Run markdownlint on all docs and fix violations.
- [ ] Add `docs/formalproof.md` reference to README documentation table (done).
- [ ] Verify architecture.md threat model table is current.
- [ ] Update `docs/cli.md` with all current commands and flags.

**Acceptance criteria**: All documentation accurate; markdownlint clean (except pre-existing MD013).

---

## Step 6 — Security Hardening

**Goal**: Reduce `#nosec` suppressions by fixing underlying patterns.

- [ ] Audit all 56 `#nosec` annotations for necessity and justification quality.
- [ ] Replace `math/rand` with `math/rand/v2` where applicable (scoring already uses v2).
- [ ] Add input validation for G115 conversion sites (longbow Arrow schema, converter tensor dims, native kernel dims).
- [ ] Add path traversal checks for G304 file read sites (SAE paths, dataset paths, OPA bundles).
- [ ] Add integer overflow checks for G115 tensor dimension conversions.
- [ ] Reduce G103 (unsafe) suppressions by using safe alternatives where possible.
- [ ] Add `#nosec` justification comments where suppression is truly necessary.
- [ ] Run gosec with `--confidence=high` and fix all high-confidence issues.
- [ ] Add security-focused fuzz targets for file I/O and path handling.
- [ ] Document security model and threat mitigations in `docs/architecture.md`.
- [ ] Add dependency vulnerability scanning to CI (govulncheck already in verify.sh, add to CI).

**Acceptance criteria**: gosec reports 0 issues; nosec count reduced by ≥25% (target ≤42); all high-confidence issues resolved.

---

## Step 7 — Observability Enhancement

**Goal**: Add granular profiling metrics for steering and scoring operations.

- [ ] Add per-layer scoring latency histogram (`burnnotice_scoring_layer_duration_seconds`).
- [ ] Add per-layer steering intervention latency histogram (`burnnotice_steering_layer_duration_seconds`).
- [ ] Add GCAD decay computation latency metric.
- [ ] Add crescendo detector step latency metric.
- [ ] Add DoRA hardening pass duration metric.
- [ ] Add SAE feature extraction latency metric.
- [ ] Add KV-cache re-anchor latency metric.
- [ ] Add metric for number of active SSE connections (verify existing).
- [ ] Add metric for profile manager queue depth.
- [ ] Add metric for scheduler available memory (verify existing).
- [ ] Update Grafana dashboard with new metrics and panels.
- [ ] Update `docs/metrics.md` with new metric descriptions and labels.
- [ ] Add OTLP trace spans for steering operations.

**Acceptance criteria**: 65+ metrics; Grafana dashboard updated; OTLP traces for key operations.

---

## Step 8 — Error Handling and Resilience

**Goal**: Improve CLI error messages, add retry logic, and better user feedback.

- [ ] Add retry logic with exponential backoff for network operations (downloader).
- [ ] Add retry logic for ONNX Runtime initialization failures.
- [ ] Improve CLI error messages with actionable suggestions.
- [ ] Add `--verbose` flag for detailed error output and debug logging.
- [ ] Add graceful degradation when OPA sidecar is unavailable.
- [ ] Add graceful degradation when Longbow mesh is unreachable.
- [ ] Add timeout configuration for all network operations.
- [ ] Add circuit breaker for repeated failures (downloader, OPA, Longbow).
- [ ] Add structured error types with error codes.
- [ ] Add error recovery documentation in `docs/cli.md`.
- [ ] Add `--json` output flag for machine-readable CLI output.

**Acceptance criteria**: All network operations have retry + timeout; error messages are actionable; structured errors with codes.

---

## Step 9 — Cross-Platform GPU Support

**Goal**: Improve Metal/CUDA kernel compilation and testing in CI.

- [ ] Add macOS (Metal) build target to CI (GitHub Actions macOS runner).
- [ ] Add CUDA kernel compilation check (nvcc) in CI.
- [ ] Add Metal shader compilation check (xcrun metal) in CI.
- [ ] Add cross-platform test matrix (linux/amd64, linux/arm64, darwin/amd64, darwin/arm64, windows/amd64).
- [ ] Add GPU kernel unit tests (CPU fallback verification).
- [ ] Document GPU setup requirements in `docs/architecture.md`.
- [ ] Verify Docker image for CUDA (already exists).
- [ ] Verify Docker image for Metal (already exists).
- [ ] Test WASM execution on all supported platforms.
- [ ] Add platform-specific performance baselines.
- [ ] Add GPU memory usage metrics.

**Acceptance criteria**: CI tests all 5 platform targets; GPU kernels compile; platform baselines committed.

---

## Step 10 — Code Quality and Fuzz Expansion

**Goal**: Expand fuzz targets and improve code quality.

- [ ] Add fuzz targets for `internal/scoring` (scorer, async scorer, normalizer, vecDot, vecNorm).
- [ ] Add fuzz targets for `internal/steer` (biprojection, CAST, commitment, coherence, quantization).
- [ ] Add fuzz targets for `internal/analysis` (CKA, concept cone, latent vis, externalities).
- [ ] Add fuzz targets for `internal/engine` (MoE routing, ONNX parsing, top-k selection).
- [ ] Add fuzz targets for `internal/hallucination` (energy, family features, calibration).
- [ ] Add fuzz targets for `internal/checkpoint`, `internal/compliance`, `internal/datasets`, `internal/downloader`, `internal/policydetect`.
- [ ] Add property-based tests for cosine similarity bounds.
- [ ] Add property-based tests for DoRA KL ceiling.
- [ ] Add property-based tests for SAE metric ranges.
- [ ] Add property-based tests for angular drift bound.
- [ ] Run `go test -shuffle=on` in CI to catch order-dependent tests.
- [ ] Run `golangci-lint` with additional linters (gocyclo, gocognit, funlen).
- [ ] Add code complexity thresholds to CI (max cyclomatic complexity 15).
- [ ] Target: 90+ fuzz targets across 34 packages.

**Acceptance criteria**: 90+ fuzz targets; complexity thresholds enforced; shuffle tests in CI.

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
EOF
echo "written"