# BurnNotice Test Suite & Security Reference

BurnNotice employs a rigorous testing suite and static analysis configuration to ensure absolute reliability, mathematical precision, thread safety, and zero security vulnerabilities.

---

## 1. Testing Strategy

The test suite prioritizes high-coverage unit and integration tests for all handwritten logic, alongside offline validation of external formats (like ONNX models and Gemma Scope NPZ archives).

### Programmatically Generated Mock Objects
To allow fully offline and lightweight testing without needing massive model files, the test suite generates minimal, mathematically valid mock inputs:
1. **Dynamic ONNX Models**: Built programmatically using `google.golang.org/protobuf` and `onnxpb`. The generator constructs a valid ONNX graph featuring a single `Gather` operator behaving as an embedding layer. This allows verifying `Load`, `Forward`, and `GetIntermediate` in unit tests, skipping gracefully when the native `onnxruntime.so` is absent.
2. **NumPy v1/v2 NPY Arrays**: Generates raw byte buffers with Python-style dictionary descriptors (representing `shape`, `descr`, and `fortran_order`) to validate the parser.
3. **NPZ Zip Archives**: Packages mock `.npy` files inside a zip writer to test `LoadGemmaScope` without actual binary assets.

---

## 2. Package Coverage Summary

Unit testing targets ~90%+ statement coverage on all core packages. Below is a breakdown of what is validated per component:

### `internal/math` (96.6% Coverage)
- **Winsorization**: Extreme quantile bounds (e.g. quantiles $\le 0$, $\ge 1$) to verify indexing clamps inside `Winsorize` and `WinsorizeInPlace`.
- **Covariance**: Unbiased Bessel-corrected covariance, Ledoit-Wolf optimal shrinkage, all-zero inputs, and empty/ragged arrays.
- **Whitening**: Eigenvalue floor calculations, ZCA whitening matrices, unwhitening reconstruction, asymmetric input rejections, and infinite/NaN value handling.
- **Projection**: Orthogonal projection and rejection onto unit-norm vectors, with degenerate (zero-norm) vector handling.

### `internal/sae` (92.8% Coverage)
- **NPZ Archiver**: Successful parsing of both standard names (`encoder_weights`, `decoder_weights`) and alternative schemas (`W_enc`, `W_dec`).
- **NPY Format**: NumPy version 1 and version 2 headers, data truncation boundaries, non-float data-types, and non-2D/non-1D dimension checks.
- **Errors**: Nonexistent files, corrupt JSON configurations, and missing zip archive entries.

### `internal/zjson` (85.0% Coverage)
- **Encoder**: Proper `uint64` and float bounds, omission of empty values (`omitempty`), and escape hex characters.
- **Decoder**: Underflow/overflow tokenization, unexpected syntax character errors, and invalid null/bool formats.
- **Headers**: Zero-allocation reflection castings for `SliceHeader` and `StringHeader`.

### `internal/longbow` & `internal/telemetry` (90.8% / 94.5% Coverage)
- **Fallback Mode**: Handles connection failures to Flight servers by automatically hot-swapping to local memory vector lists.
- **SQLite Engine**: Verifies database constraints, WAL journal initialization, query parameterized placeholders, and handling of database close states.
- **Telemetry**: Network timeout simulations, HTTP status error codes, and local file storage fallbacks on write failures.

### `internal/analysis` (≥85% Coverage)
- **AlignmentImprintDetector**: Empty map guard (returns `unknown`), single-layer guard (returns 0 for all metrics), multi-layer metric computation, and all four `ClassifyAlignment` branches.
- **ConceptConeAnalyzer**: Minimum-2-category guard, `nil`-direction category skipping in `computeSingularValues`, pairwise cosine accuracy, and intrinsic dimensionality estimation.
- **CrossLayerAlignmentAnalyzer**: Pairwise similarity matrix correctness, `DecisionLayer` with `threshold ≤ 0` defaulting to 0.8, `DirectionClusters` with `threshold ≤ 0` defaulting to 0.85, and empty-map guard.
- **OuroborosDetector**: `CompensatingLayers` detection, `NetReduction` edge cases (zero pre-scores), `RequiresRefinement` thresholding, and `RefinementPasses` capping.
- **CKA (standalone)**: Mismatched-length guard (returns 0), identical-matrix CKA = 1.0, orthogonal-matrix CKA = 0.

### `internal/hallucination` (≥88% Coverage)
- **HallucinationDetector**: `ComputeEnergy`, `DetectHallucination` (with and without explicit threshold), `DetectSpike` (window not full, window full, spike detected), `RiskScore` (below/above threshold), `UpdateHistory`, and `Validate` error paths.

### `internal/metrics` (≥80% Coverage)
- **Prometheus Handler**: Verifies the `/metrics` HTTP endpoint returns `200 OK` with `text/plain` content-type.
- **OTel Exporter**: `OTelConfigFromEnv` with and without environment variables, no-op initialization (empty endpoint), URL parse error handling, and `ShutdownOTel` with un-initialized state.
- **Sampler**: Rate limiter behavior and pass-through semantics.

### `internal/onnxpb` (≥75% Coverage)
- **Getter Functions**: `GetOpType`, `GetInputs`, `GetOutputs`, `GetInitializers` against programmatically constructed `ModelProto` messages.
- **Edge Cases**: Nil node guards, empty graph guards.

### `cmd/burnnotice` (≥80% Coverage)
- **`inference.go`**: Direction projection, scoring thresholds, and activation sanitization on synthetic activations.
- **`main.go`**: Command registration, flag parsing, and help output.

### `templates` (78.9% Coverage)
- **HTML Renderers**: Verifies layout tags, HTMX/SSE scripts, and Dashboard overview structure are parsed and output correctly.

---

## 3. Fuzz Testing

Several packages include corpus-seeded fuzz tests runnable with `go test -fuzz`:

| Package | Fuzz Target | What It Tests |
|---|---|---|
| `internal/math` | `FuzzWinsorize` | Extreme float inputs (NaN, Inf, ±1e30) to robust.go |
| `internal/math` | `FuzzCovariance`, `FuzzWhitening` | Degenerate covariance inputs |
| `internal/extract` | `FuzzDIM`, `FuzzPCA` | Extreme activations in direction extractors |
| `internal/extract` | `FuzzWhitened` | SVD numerical stability under extreme whitening |
| `internal/scoring` | `FuzzScorer` | Score computation under random activations |
| `internal/steer` | `FuzzSteering`, `FuzzLoRA` | Steering vector application under random inputs |
| `internal/metrics` | `FuzzMetrics` | Prometheus metric registration under concurrent load |

Run a single fuzz target for 30 seconds:

```bash
go test -fuzz=FuzzWinsorize -fuzztime=30s ./internal/math/
```

---

## 4. Benchmark Tests

Benchmarks live alongside unit tests in `bench_test.go` files:

| Package | Benchmark | Typical Result |
|---|---|---|
| `internal/scoring` | `BenchmarkScore` | ~27µs per 4096-dim activation |
| `internal/scoring` | `BenchmarkSubspaceScore` | ~2µs per projection |
| `internal/steer` | `BenchmarkActAdd` | ~10µs per activation addition |
| `internal/steer` | `BenchmarkDot` | ~650ns dot product |

Run all benchmarks:

```bash
go test -bench=. -benchmem ./...
```

---

## 5. Running the Tests

To run the full suite with statement coverage logging, execute:

```bash
go test -cover ./...
```

To output a detailed coverage profile for visual analysis:

```bash
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out
```

To run with the race detector (required for CI):

```bash
go test -race ./...
```

---

## 6. Security Auditing (Gosec)

BurnNotice integrates `gosec` to scan Go ASTs for security vulnerabilities. The repository maintains **0 security issues** (with 48 audited `#nosec` exclusions for necessary user file loading).

For Go 1.27.2, the scanner must be built with `golang.org/x/tools` v0.52.0 or newer so its SSA importer can read the compiler's export data. The verified scan used gosec v2.29.0 with x/tools v0.52.0; the default gosec v2.29.0 dependency on x/tools v0.49.0 reports importer errors instead of completing analysis.

Run a security audit locally using:

```bash
# Build gosec with a Go 1.27.2-compatible analyzer dependency.
tool_dir=$(mktemp -d)
(
  cd "$tool_dir"
  go mod init gosec-tools
  go get github.com/securego/gosec/v2/cmd/gosec@v2.29.0
  go get golang.org/x/tools@v0.52.0
  go install github.com/securego/gosec/v2/cmd/gosec
)

# Run the scan from the repository root.
"$(go env GOPATH)/bin/gosec" ./...
```
