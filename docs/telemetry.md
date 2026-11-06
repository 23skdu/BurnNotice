# BurnNotice Anonymous Telemetry Specification

> **Version**: 1.0  
> **Status**: Active  
> **Compliance**: Zero-Knowledge Privacy Architecture  

BurnNotice includes an optional, anonymous telemetry subsystem designed to gather aggregate empirical metrics on activation steering efficacy across model architectures and hardware backends without compromising user privacy or intellectual property.

---

## 1. Core Privacy Guarantees

Telemetry in BurnNotice is built upon strict zero-knowledge principles:

| Collected | Strictly Excluded (Never Stored or Transmitted) |
|-----------|-------------------------------------------------|
| High-level model family (e.g. `llama`, `qwen`) | **Prompts or user inputs** |
| Coarse model parameter size (e.g. `7B`, `13B`) | **Model outputs or generated text** |
| Steering method name (e.g. `whitened_svd`, `dim`, `mpoa`) | **Token IDs or vocab indices** |
| Benchmark refusal rate (e.g. `0.05`) | **Raw activation vectors or internal states** |
| Token KL-divergence score (e.g. `0.12`) | **Model weights or adapter checkpoints** |
| Capability retention score (e.g. `0.98`) | **IP addresses, MAC addresses, or network metadata** |
| Hardware backend category (e.g. `cuda`, `metal`, `wasm`) | **User identities, hostnames, or machine IDs** |
| Unix timestamp (seconds) & schema version | **API keys, authentication tokens, or environment vars** |

All telemetry records are fully anonymous, un-linkable, and mathematically decoupled from user data.

---

## 2. Telemetry Schema (`Report`)

Each telemetry payload conforms to the Go struct `telemetry.Report` defined in [`internal/telemetry/reporter.go`](internal/telemetry/reporter.go):

```json
{
  "model_architecture": "llama",
  "model_size": "7B",
  "steering_method": "whitened_svd",
  "refusal_rate": 0.05,
  "kl_divergence": 0.12,
  "capability_retention": 0.98,
  "hardware": "cuda",
  "timestamp": 1772659200,
  "version": 1
}
```

### Field Definitions

1. **`model_architecture`** (`string`): General model architecture family (e.g. `llama`, `qwen`, `mistral`).
2. **`model_size`** (`string`): Approximate parameter count bucket (e.g. `0.5B`, `7B`, `13B`, `70B`).
3. **`steering_method`** (`string`): Intervention algorithm applied (`dim`, `pca`, `whitened_svd`, `rdo`, `mpoa`, `angular`).
4. **`refusal_rate`** (`float64`): Post-intervention refusal rate on benchmark harm datasets in range $[0.0, 1.0]$.
5. **`kl_divergence`** (`float64`): Token-level logit Kullback-Leibler divergence between original and steered logits.
6. **`capability_retention`** (`float64`): Performance retention score on benign benchmark tasks in range $[0.0, 1.0]$.
7. **`hardware`** (`string`): Execution engine used (`cuda`, `metal`, `cpu`, `wasm`).
8. **`timestamp`** (`int64`): Unix timestamp in seconds when the benchmark finished.
9. **`version`** (`int`): Telemetry schema format version (currently `1`).

---

## 3. Submission Methods

Telemetry supports two submission mechanisms via `telemetry.Reporter`:

### 3.1 Local Storage (Default)
When no remote HTTP endpoint is specified, telemetry is written locally to disk:
- **Default Directory**: `./telemetry/`
- **Filename**: `telemetry_<timestamp>.json`
- **Permissions**: Directory `0750`, files `0600` (read/write by owner only).

### 3.2 Secure HTTPS Submission
If an authorized endpoint URL is configured:
- Reports are sent via `HTTP POST` with `Content-Type: application/json`.
- Standard timeout: 10 seconds.
- Fails closed: if the network or endpoint is unreachable, an error is reported without blocking local profiling workflows.

---

## 4. Local Telemetry Aggregation

Users can inspect and summarize all local telemetry reports using the CLI:

```bash
burnnotice aggregate ./telemetry
```

Example output:
```text
Aggregated 12 telemetry reports from ./telemetry:
  Average Refusal Rate:         4.20%
  Average KL Divergence:        0.1042
  Average Capability Retention: 98.45%
  Model Architectures:          llama, qwen
  Steering Methods:             whitened_svd, rdo
```

---

## 5. Security and Auditability

- **Code Location**: [`internal/telemetry/reporter.go`](internal/telemetry/reporter.go)
- **Security Audit**: Codebase audited with `gosec` (0 security warnings).
- **Network Verification**: Telemetry client makes zero background network calls unless explicitly configured with a remote endpoint.
