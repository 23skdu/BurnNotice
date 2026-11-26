# BurnNotice Implementation Roadmap: Gap Resolution Plan

> **Status**: In Progress — Gap Resolution Phase
> **Last Updated**: 2026-11-26
> **Version**: 0.9.1
> **Scope**: Resolve verification gaps discovered during the 2026-11-26 audit of the completed v0.8/v0.9 ten-part improvement plan.

---

## Completed Work (Verified & Removed)

The ten-part improvement plan (multi-turn crescendo re-anchoring, DoRA extended-refusal fine-tuning, cross-architecture SAE alignment, performance, accuracy, testing, CI/release, architecture compatibility, adversarial defenses, observability/deployment) was audited on 2026-11-26 and verified fully complete. Its checklist items have been removed from this roadmap. Audit evidence:

| Part | Verified Scope | Evidence |
| ---- | -------------- | -------- |
| 1 | Crescendo context re-anchoring | `internal/steer/crescendo.go`, `kvcache.go`; oracle/accuracy/fuzz/race tests green; `GCADApplyInPlace` 0 B/op, 0 allocs/op (~3.1µs) |
| 2 | DoRA extended-refusal fine-tuning | `internal/steer/dora.go`; `burnnotice finetune --extended-refusal` CLI with all documented flags; `DoRAManifest` in `ReleaseEvidenceBundle` |
| 3 | Cross-architecture SAE alignment | `internal/sae/loader.go`, `alignment.go`; Gemma Scope / Llama-Scope / Qwen-SAE loaders, absorption diagnostics, CKA/Procrustes alignment; `FamilyEnergyScale` + `IngestFamilyFeatures` integration |
| 4 | Performance & memory | Scoring hot paths ≤65µs (max observed 35.4µs); pooled scoring; cached thin-SVD; zero-alloc GCAD decay |
| 5 | Mathematical accuracy & oracles | `make accuracy-gate` and `make oracle-gate` (`-count=2`) green |
| 6 | Testing scope & concurrency | 76 native fuzz targets across 27 packages; `go test -race ./...` clean |
| 7 | CI & release quality gates | `scripts/verify.sh` pre-commit gate; `.github/workflows/ci.yml` accuracy/oracle/profiling/fuzz/evidence gates; `make release-evidence` produces `dist/evidence/` |
| 8 | Architecture compatibility | Family detection/calibration, MoE top-k routing, INT8/FP8 quantization transfer, oversight-ablated profiling |
| 9 | Adversarial threat model & defenses | Obfuscation probes, steering externalities, alignment-faking, KV contamination, topic confusion, ouroboros compensation, DoRA weight hardening; threat-model table in `docs/architecture.md` |
| 10 | Observability & deployment | 55 Prometheus metrics + OTLP/HTTP; SSE dashboard + latent-space SVG; `helm lint` clean; scrape config in `helm/burnnotice/values.yaml` |

Defects found and **fixed during the audit** (verified green, no open items):

- **DoRA KL-ceiling enforcement**: `evaluateAndScaleDoRA` could return weights violating `MaxKLDivergence` after 6 proportional scaling attempts (fuzz-discovered). Added a guaranteed geometric fallback and made `EffectiveScale` report the applied scale (`internal/steer/dora.go`); failing input retained as a regression seed in `internal/steer/testdata/fuzz/FuzzDoRAExtendedRefusal/`.
- **Cross-platform builds**: `make vet-cross-platform` (`CGO_ENABLED=0`) failed because `internal/engine/native` and the ORT bindings were cgo-only. Added a pure-Go kernel fallback (`dispatch_fallback.go`), split ORT behind build tags (`onnx_ort.go` / `onnx_noort.go`) with the existing protobuf-session fallback, and bounds-checked kernel dimensions. All four cross-vet targets now pass.
- **gosec suppression hygiene**: added explicit bounds checks where `#nosec G115` was unjustified (longbow Arrow schema dimension, ONNX tensor dimensions, native kernel dimensions) and documented the non-cryptographic `G404` sites in `internal/scoring/scorer.go`. gosec reports 0 issues across 112 files.
- **Job cancellation race**: `profile.Job.Cancel()` silently dropped cancellations issued before the worker lazily created the job context, hanging `Manager.Stop()` and `TestHandlers_WiredRuntimeEndpoints` on fresh (uncached) runs. Cancellation is now sticky (`internal/profile/manager.go`) with deterministic regression tests; cached test results had masked the hang.

---

## Gap Analysis (2026-11-26)

| # | Gap | Evidence | Impact |
| - | --- | -------- | ------ |
| G1 | Statement coverage below the claimed 90% floor | Per-package coverage (generated code excluded): `internal/converter` 82.4%, `internal/registry` 84.6%, `internal/zjson` 85.0%, `cmd/burnnotice` 86.8%, `internal/downloader` 87.2%, `internal/web` 87.0%; repo total 86.6% | Part 6 claim "90%+ across all 26 core packages" was not satisfied; untested error paths in six packages |
| G2 | No coverage gate in CI | `.github/workflows/ci.yml` has no coverage step or threshold | Part 6 acceptance criterion "coverage gates enforced in CI" unmet; coverage can regress silently |
| G3 | Performance baselines not reproducible from the repo | `benchmark_results.txt` is gitignored and absent from the checkout; CI uploads it as an artifact but nothing committed exists to compare against | Part 4 acceptance criterion "no regression vs. stored baselines" cannot be evaluated |
| G4 | Module graph not portable | `go.mod` contains `replace github.com/23skdu/longbow => ../longbow`; CI has no sibling checkout; the local sibling is 268 commits ahead of the pinned pseudo-version `v0.2.1-rc3.0.20261106033529-8c74c7d8ee00` | Fresh clones and CI cannot resolve the module; CI build gates are broken for anyone without the sibling repo |
| G5 | Security/lint scanning only runs locally | gosec, govulncheck, and golangci-lint run in `scripts/verify.sh` (pre-commit) but not in `ci.yml` | Gates are bypassable with `git commit --no-verify`; no server-side enforcement |

---

## Step 1 — Restore ≥90% Coverage in Lagging Packages (resolves G1)

- [ ] `internal/converter` (82.4%): test ONNX serialization/validation error paths (including the new int64 dimension bounds check), safetensors parse edge cases, and `Convert` failure branches.
- [ ] `internal/registry` (84.6%): cover registration, lookup, duplicate/conflict, and persistence error paths.
- [ ] `internal/zjson` (85.0%): cover malformed-input, truncation, and zero-copy parsing edge paths.
- [ ] `cmd/burnnotice` (86.8%): cover `finetune` gating/rollback flags and ops-command error branches.
- [ ] `internal/downloader` (87.2%): cover retry, checksum-mismatch, and partial-download failure paths.
- [ ] `internal/web` (87.0% excl. generated templ code): cover SSE hub broadcast/disconnect and handler error paths.
- [ ] Re-run `go test -coverprofile` and confirm every non-generated package ≥90% (generated `internal/onnxpb` and `templates` are exempt).

## Step 2 — Enforce Coverage Gates in CI (resolves G2)

- [ ] Add a `make test-coverage-gate` target computing per-package statement coverage with a ≥90% floor, excluding generated packages (`internal/onnxpb`, `templates`).
- [ ] Wire the coverage gate into `scripts/verify.sh`.
- [ ] Add a coverage step to `.github/workflows/ci.yml` and upload the coverage profile as an artifact.
- [ ] Document the gate and exemptions in `docs/testing.md`.

## Step 3 — Commit Reproducible Performance Baselines (resolves G3)

- [ ] Generate fresh baselines via `make profile` on reference hardware; store a committed copy (e.g. `docs/benchmarks/baseline.txt`) annotated with CPU model, Go version, and commit SHA.
- [ ] Update `.gitignore` and the Makefile `profile-cpu`/`profile-mem` targets so the committed baseline path is tracked while local scratch outputs remain ignored.
- [ ] Add `scripts/bench-regression.sh` comparing a current benchmark run against the committed baseline with a documented tolerance (e.g. ±15% ns/op, allocation counts must not increase).
- [ ] Wire the regression check into the CI profiling gate and document the rebaseline procedure in `docs/testing.md`.

## Step 4 — Make the Module Graph Portable (resolves G4)

- [ ] Decide strategy: publish/tag `23skdu/longbow` at the commit BurnNotice requires, or keep the local `replace` and teach CI to check out the sibling.
- [ ] If publishing: bump the `go.mod` requirement to the published version, drop the `replace`, run `go mod tidy`, and verify `go build ./...` and `go test ./...` against the published module.
- [ ] If keeping the `replace`: add an `actions/checkout` step for `23skdu/longbow` at a pinned SHA into `../longbow` in `ci.yml`, and document the sibling-repo requirement in `README.md`.
- [ ] Verify a fresh `git clone` into a clean directory builds without a pre-existing `../longbow`.

## Step 5 — Run Security & Lint Gates in CI (resolves G5)

- [ ] Add a `golangci-lint` step to `.github/workflows/ci.yml`.
- [ ] Add a `gosec` step that fails on any reported issue; keep `#nosec` suppressions justified and bounds-checked.
- [ ] Add a `govulncheck` step against the Go vulnerability database.
- [ ] Confirm the full CI workflow is green end-to-end on a pull request.

## Step 6 — Sustained Fuzz Hardening (post-fix verification)

- [ ] Run extended fuzz campaigns (≥5 min/target) on `FuzzDoRAExtendedRefusal`, `FuzzCrescendoReAnchoring`, and `FuzzSAEFeatureAbsorptionAndAlignment`.
- [ ] Retain the DoRA KL-ceiling regression seed (`internal/steer/testdata/fuzz/FuzzDoRAExtendedRefusal/2e473a76537f3881`) and commit any newly discovered failing seeds as regressions.
- [ ] Fix any invariant violations surfaced and re-run the accuracy/oracle gates.

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
make test-fuzz
```
