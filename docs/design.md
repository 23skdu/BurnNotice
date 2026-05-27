# BurnNotice: Neural Activation Governance Framework for LLM Agents

## Design Document

> **Status**: Draft
> **Date**: 2026-05-18
> **Version**: 0.3
> **Language**: Go (primary), ONNX Runtime + WASM (inference)

---

## 1. Executive Summary

BurnNotice is a real-time neural activation monitoring and intervention framework designed to detect when an LLM agent is considering deception, policy violation, or alignment drift — before the behavior manifests in output. By adapting abliteration techniques (originally developed for refusal removal) in reverse — as detection and steering mechanisms rather than permanent weight edits — BurnNotice provides a white-box governance layer for autonomous AI agents.

The framework identifies "conflict subspaces" and "deception directions" in activation space, visualizes the neural circuits involved in real time, and offers both passive monitoring and active intervention capabilities to keep agents aligned with their assigned rules and tasks.

To achieve zero-overhead inference, BurnNotice deploys specialized C++/CUDA and Metal plugins directly inside host engines (like vLLM, TensorRT-LLM, and MLX), bypassing PCIe transfer bottlenecks entirely. The Go layer operates purely as an out-of-band policy control plane and telemetry aggregator.

The core engine is written in Go for long-running, efficient, concurrent model analysis. All model inference runs through ONNX Runtime with WebAssembly (WASM) fallback for portable, dependency-free deployment. Activation vectors are stored and processed using Longbow's Apache Arrow Flight vector engine for zero-copy data paths, SIMD-optimized distance calculations, and distributed activation sharing. The web dashboard uses Go's standard `net/http` server with Templ templates and HTMX for real-time, low-overhead UI updates.

---

## 2. Background and Motivation

### 2.1 The Problem

LLM agents deployed in production exhibit failure modes that traditional output-level monitoring cannot catch:

- **Strategic deception**: Models that knowingly generate falsehoods to achieve objectives (distinct from hallucination)
- **Alignment faking**: Models that appear compliant under monitoring but revert to preferred behaviors when unobserved
- **Policy violation**: Subtle deviations from assigned rules that don't trigger content filters
- **Safety geometry collapse**: Fine-tuning that erodes the latent boundaries between safe and unsafe behavior
- **Refusal suppression**: Jailbreaks that bypass learned safety mechanisms by manipulating activation trajectories

Current governance tools (Agent Sentinel, Agent-Aegis, LLUMO, etc.) operate at the API/trace level — monitoring costs, latency, tool calls, and output text. None inspect the internal neural state of the model during generation.

### 2.2 The Opportunity

Recent mechanistic interpretability research has established that:

1. **Refusal is mediated by a single direction** in activation space (Arditi et al., 2024)
2. **Deception induces representational shifts** detectable via linear probes with 93%+ accuracy (multiple studies, 2025)
3. **Truth neurons** exist — neurons that encode truthfulness in a subject-agnostic manner (2025)
4. **Conflict subspaces** are where LLMs decide between competing objectives, orthogonal to truth subspaces (2025)
5. **Hallucination manifests as phase transitions** in latent potential energy, detectable via SAE features (HalluSAE, 2026)
6. **Policy violations** can be detected via activation-space whitening without training (2025)

Abliteration demonstrated that these directions can be permanently removed via weight orthogonalization. BurnNotice inverts this approach: instead of removing directions, we **monitor** them in real time and **steer** away from misalignment when detected or terminate the agent (BurnNotice)

### 2.3 Why Go

Go is chosen as the primary language for several reasons:

- **Long-running processes**: Go's goroutine model and garbage collector are optimized for services that run for weeks or months without restart — essential for continuous model analysis jobs that may take hours or days
- **Memory efficiency**: Analyzing activations from large models (billions of parameters, thousands of layers × tokens) requires careful memory management. Go's explicit control over allocations and slice reuse avoids the overhead of Python's object model
- **Concurrency**: Model profiling runs multiple contrastive prompt batches in parallel; Go channels provide clean synchronization without callback hell
- **Single binary deployment**: No virtual environment, no dependency resolution at deploy time — critical for GPU cluster environments
- **ONNX + WASM**: All model inference runs through ONNX Runtime with WASM fallback, eliminating Python/PyTorch dependencies entirely. Models are converted from HuggingFace format via the CLI tool
- **Longbow Vector Engine**: Apache Arrow Flight-based vector engine provides zero-copy activation storage, SIMD-optimized distance calculations (cosine, euclidean, dot product), off-heap memory management via SlabArena, and distributed activation sharing across nodes. BurnNotice leverages Longbow's HNSW index for fast nearest-neighbor lookups on activation centroids, its Arrow RecordBatch format for efficient activation streaming, and its COW publication model for lock-free concurrent scoring
- **Native GPU Plugins**: The math-heavy steering computations (`cosine_similarity` and `dynamic_rotation`) are implemented as native GPU kernels (`plugins/cuda_kernels.cu`, `plugins/metal/kernels.metal`) and injected directly into host inference engines via PyBind11 (vLLM) and `IPluginV2DynamicExt` (TensorRT).
- **Asynchronous Go Control Plane**: The Go engine coordinates policy, extracts dimensions, and updates the vector store out-of-band via an `AsyncLedger`. Token generation is never blocked.
- **Templ + HTMX**: Type-safe server-rendered UI with zero JavaScript build step. HTMX provides real-time updates via SSE/WebSocket with minimal client-side code, keeping the dashboard lightweight and maintainable

All orchestration, policy enforcement, model profiling, vector storage, and visualization run in Go, while per-token steering happens natively on the GPU.

---

## 3. Core Concepts

### 3.1 Abliteration (Foundational Technique)

Abliteration identifies a behavioral direction (e.g., refusal) by computing the difference-in-means between activations on contrasting prompt pairs, then removes it via:

**Inference-time ablation**:

```go
h' = h - (h · r̂)r̂
```

**Weight orthogonalization** (permanent):

```text
W' = W - (W · r̂)r̂
```

BurnNotice uses the same direction-finding machinery but applies it as a **detection** and **steering** mechanism rather than permanent removal.

### 3.2 Key Activation-Space Constructs

| Concept | Description | Detection Method |
| --- | --- | --- |
| **Refusal Direction** | Single vector encoding refusal behavior | Difference-in-means (harmful vs harmless prompts) |
| **Deception Direction** | Axis separating truthful from deceptive activations | Contrastive pairs + PCA / LAT |
| **Truth Subspace** | Region encoding truthful, grounded responses | Linear probes on residual stream |
| **Conflict Subspace** | Where competing objectives are resolved; orthogonal to truth | Clustering of delta vectors between truth/deception centroids |
| **Truth Neurons** | Individual neurons causally linked to truthfulness | Integrated gradients + systematic filtering |
| **SAE Deception Features** | Sparse autoencoder features that flip between truth/deception | SAE feature attribution analysis |
| **Policy Violation Direction** | OOD direction encoding policy non-compliance | Activation-space whitening + Euclidean norm |
| **Hallucination Energy** | Latent potential energy indicating phase transition from truth to fabrication | SAE-based geometric energy metric |
| **Oversight Sensitivity Direction** | Latent subspace encoding the model's awareness of being evaluated/monitored | Difference-in-Means (Monitored vs Deployed prompts) |

### 3.3 The Linear Representation Hypothesis

The foundational assumption: concepts are represented as approximately linear directions in transformer activation spaces. This enables:

- **Detection**: Project activations onto known directions to get scalar scores
- **Steering**: Add/subtract scaled direction vectors to influence behavior
- **Visualization**: Map high-dimensional activations to interpretable 2D/3D subspaces

---

## 4. Architecture

See [`docs/architecture.md`](architecture.md) for the complete architecture overview with Mermaid diagrams.

### 4.1 System Overview

BurnNotice is implemented as a long-running Go application with four primary components:

1. **ONNX Inference Engine**: Loads and executes models in ONNX format, with WASM fallback for CPU-only environments. Forward hooks capture intermediate activations at specified layers. Inference utilizes a specialized `WordHashTokenizer` which handles text-to-integer token sequences with zero sub-string memory allocations.
2. **Longbow Vector Engine**: Apache Arrow Flight-based vector store for activation data. Provides zero-copy activation storage, SIMD-optimized distance calculations, off-heap memory management (SlabArena), HNSW index for nearest-neighbor centroid lookups, and distributed activation sharing across cluster nodes.
3. **Activation Monitor**: Receives activation streams, computes scoring metrics against registered direction profiles stored in Longbow, and triggers steering interventions when thresholds are exceeded.
4. **Web Dashboard**: Real-time monitoring interface built with Templ and HTMX, delivering activation heatmaps, score timelines, and steering event logs via Server-Sent Events (SSE).

The CLI tool (`burnnotice`) handles model acquisition from HuggingFace, ONNX conversion, and profile generation. The converter builds valid `ModelProto` binaries via Go Protocol Buffers (`internal/onnxpb`) rather than proprietary formats, ensuring full ONNX compliance.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                            BurnNotice Framework                              │
│                                                                              │
│  Go Engine (burnnotice/)                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────────────────────┐    │
│  │ Profile  │    │ Direction    │    │ Activation                     │    │
│  │ Manager  │───▶│ Extractor    │───▶│ Monitor                        │    │
│  │          │    │              │    │                                │    │
│  │ - Jobs   │    │ - DIM        │    │ - ONNX/WASM activations       │    │
│  │ - Queue  │    │ - PCA/SVD    │    │ - Longbow vector store        │    │
│  │ - Workers│    │ - LEACE      │    │ - Real-time scoring           │    │
│  └──────────┘    │ - INLP       │    │ - Layer-wise tracking         │    │
│                  └──────────────┘    └───────────────┬────────────────┘    │
│                                                      │                     │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │ Longbow Vector Engine (github.com/23skdu/longbow)                │      │
│  │                                                                  │      │
│  │ - Apache Arrow Flight for zero-copy activation transport        │      │
│  │ - SlabArena off-heap memory management (bypasses Go GC)         │      │
│  │ - SIMD-optimized distance calculations (cosine, euclidean, dot) │      │
│  │ - HNSW index for nearest-neighbor activation centroid lookup    │      │
│  │ - COW publication for lock-free concurrent scoring              │      │
│  │ - Distributed mesh for multi-node activation sharing            │      │
│  │ - Arrow RecordBatch format for efficient batch processing       │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│                                                      │                     │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────▼────────────────┐    │
│  │ GPU Kern │◀───│ Alert &      │◀───│ Async Ledger                   │    │
│  │ Plugins  │    │ Policy Engine│    │ Engine                        │    │
│  │          │    │              │    │                                │    │
│  │ - CUDA   │    │ - Thresholds │    │ - Deception score             │    │
│  │ - Metal  │    │ - Rules      │    │ - Conflict score              │    │
│  │ - TensorR│    │ - LLM Intent │    │ - Policy OOD score            │    │
│  │ - vLLM   │    │   Classifier │    │ - Hallucination energy        │    │
│  └──────────┘    └──────────────┘    └────────────────────────────────┘    │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │ Web Dashboard (net/http + Templ + HTMX)                           │       │
│  │                                                                   │       │
│  │ - SSE stream for real-time scores                                │       │
│  │ - Templ templates for server-rendered views                      │       │
│  │ - HTMX for partial page updates (no SPA framework)               │       │
│  │ - Activation heatmap, subspace plots, circuit graphs             │       │
│  │ - Model profiler status and job queue                            │       │
│  └──────────────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Component Details

#### 4.2.1 Profile Manager

Manages long-running model profiling jobs. Each job profiles one model for its specific behavioral directions and subspaces.

**Inputs**:

- Model identifier (HuggingFace repo path or local ONNX path)
- Contrastive prompt pairs: (truthful prompt, deceptive prompt) for the same factual content
- Contrastive prompts for oversight awareness: Monitored vs Deployed context sets (Difference-in-Means dataset)
- Policy-compliant vs policy-violating prompt pairs
- Aligned vs misaligned instruction pairs

**Process**:

1. Loads the ONNX model and installs activation hooks
2. Runs preliminary profiling using monitored/deployed contrastive prompts to extract the **Oversight Sensitivity Direction** $\hat{v}_{oversight}$ at each layer
3. Queues deception/conflict prompt batches for parallel processing
4. Intercepts the activations $h$ at each layer and projects out the oversight sensitivity direction using the **Oversight Ablator**: $h' = h - (h \cdot \hat{v}_{oversight})\hat{v}_{oversight}$
5. Receives streamed ablated activations
6. Caches residual stream activations at every layer
7. Extracts at the last token position (where the model decides its response strategy)
8. For SAE-based analysis, encodes activations through pretrained sparse autoencoders
9. Persists extracted directions and profiles to disk (gob or protobuf format)

**Outputs**:

- Per-layer direction vectors (refusal, deception, truth, conflict, policy, oversight)
- SAE feature importance rankings
- Truth neuron indices (via integrated gradients)
- Layer selection scores (which layers carry strongest signal)
- Model profile persisted to `profiles/<model-id>/`

#### 4.2.2 Direction Extractor

Multiple algorithms for identifying behavioral directions, all implemented in Go operating on activation tensors.

| Method | Use Case | Directions |
| --- | --- | --- |
| Difference-in-Means (DIM) | Single dominant direction (refusal, deception) | 1 |
| PCA / SVD | Multi-directional subspaces | Top-k |
| Linear Artificial Tomography (LAT) | Unsupervised direction extraction | 1 |
| LEACE | Optimal linear erasure direction | 1 |
| INLP (Iterative Nullspace Projection) | Count independent directions | N |
| SOM (Self-Organizing Map) | Multi-cluster refusal manifold | k clusters |
| Whitened SVD | Thin SVD on centered activations | Top-k |
| Oversight-Ablated DIM | Extracts pure deception/conflict directions by removing evaluation awareness | 1 |


**Layer Selection Strategy**:

- Rank layers by mean absolute projection value (strongest signal first)
- Use discriminative layer selection: only steer layers where truth/deception exhibit opposite-signed projections
- Focus on mid-layers (40-60% depth) where deception signals peak

#### 4.2.3 Activation Monitor

Receives real-time activation streams from the inference engine during agent inference. Activations are stored in Longbow's Arrow RecordBatch format for zero-copy access.

**Activation Storage** (via Longbow):

- Activations streamed as Arrow RecordBatches into Longbow's vector store
- SlabArena off-heap allocation bypasses Go GC for billion-scale activation matrices
- COW publication enables lock-free concurrent scoring across goroutines
- HNSW index provides fast nearest-neighbor lookups on activation centroids

**Real-Time Scoring** (per token):

```go
deceptionScore[l][t] = Sigmoid(Dot(h[l][t], vDeception[l]))
conflictScore[l][t] = DistanceToConflictCentroid(h[l][t])
policyOODScore[l][t] = Norm(Whitened(h[l][t]))
truthScore[l][t] = Sigmoid(Dot(h[l][t], vTruth[l]))
hallucinationEnergy[l][t] = Sum(SAFeatureActivations^2)
```

Distance calculations use Longbow's SIMD-optimized kernels (AVX2/AVX-512/NEON) for cosine, euclidean, and dot product operations.

**Aggregation**:

- Per-token scores → per-response scores (mean, max, or area-under-curve)
- Layer-wise scores → composite alignment score (weighted by layer importance)
- Temporal patterns: detect score spikes before deceptive text is generated (early warning)

#### 4.2.4 Scoring Engine

Combines individual scores into actionable governance metrics:

| Metric | Formula | Threshold | Action |
| --- | --- | --- | --- |
| **Deception Risk** | `max(deception_score)` across layers/tokens | > 0.7 | Alert + steer |
| **Conflict Intensity** | `mean(conflict_score)` in conflict layers | > 0.6 | Monitor closely |
| **Policy Compliance** | `1 / (1 + policy_ood_score)` | < 0.5 | Block response |
| **Truth Confidence** | `mean(truth_score)` in truth layers | < 0.4 | Flag for review |
| **Hallucination Risk** | `max(hallucination_energy)` in transition layers | > threshold | Constrain decoding |
| **Composite Alignment Score** | Weighted combination of all metrics | < 0.6 | Escalate |

#### 4.2.5 Steering Engine

Active intervention when scores exceed thresholds. Four steering methods are available:

**Addition Steering** (ActAdd):

```go
h' = h + α · v
```

- Analog control: scale α to adjust intervention strength
- Use negative α to suppress deception, positive to promote truth

**Angular Steering** (norm-preserving rotation):

```go
h' = cos(θ)·h + sin(θ)·(v × h)
```

- Preserves activation norm, avoiding distribution shift
- Only effective in models > 7B parameters

**Selective Steering** (recommended):

- Apply norm-preserving rotation only to discriminative layers
- Guarantees norm preservation and avoids cascading LayerNorm drift
- Combines direction extraction with layer-wise gating

**Ablation Steering** (binary):

```go
h' = h - (h · r̂)r̂
```

- Completely removes the deception/conflict direction
- Use as last-resort intervention when other methods fail

**Weight Orthogonalization** (permanent, offline):

- For agents that need permanent behavioral modification
- Computes the orthogonalization matrix and applies it to model weights
- Zero inference overhead after edit

### 4.2.6 GPU Native Plugins (C++/CUDA/Metal)

Steering interventions are executed directly on the GPU VRAM to avoid PCIe latency:
- **CUDA Kernels**: `plugins/cuda_kernels.cu` uses warp-level reductions for `cosine_similarity` and `dynamic_rotation`.
- **Metal Shaders**: `plugins/metal/kernels.metal` uses MSL SIMD reductions and Objective-C++ wrappers for Apple Silicon natively.
- **Host Integrations**:
  - `vLLM`: Exposes native PyBind11 Python wrappers (`plugins/vllm/vllm_plugin.cpp`).
  - `TensorRT-LLM`: Uses the `IPluginV2DynamicExt` API (`plugins/tensorrt/trt_plugin.cpp`).

#### 4.2.6 Alert & Policy Engine

Defines governance rules and escalation paths:

```yaml
policies:
  - name: "no-deception"
    metric: deception_risk
    threshold: 0.7
    action: "steer"
    steering_vector: "truth_direction"
    steering_strength: 0.5
    llm_pre_classifier:
      enabled: true
      bypass_on_intent: ["IntentCreativeWriting", "IntentFictionalRoleplay"]

  - name: "policy-compliance"
    metric: policy_compliance
    threshold: 0.5
    action: "block"
    escalation: "human-review"

  - name: "conflict-monitoring"
    metric: conflict_intensity
    threshold: 0.6
    action: "log"
    escalate_after: 3

  - name: "kill-switch"
    metric: composite_alignment
    threshold: 0.3
    action: "terminate"
    notify: ["admin", "security-team"]
```

**LLM Semantic Pre-classifier**: Before enforcing strict geometric penalties, the `SemanticGater` queries an LLM intent classifier to distinguish malicious strategic deception from benign creative writing. This prevents false positive steering during roleplay tasks.

### 4.3 Web Dashboard

See [`docs/architecture.md`](architecture.md) for the complete dashboard architecture with Mermaid diagrams.

The dashboard is a Go `net/http` server serving Templ templates. HTMX handles all dynamic updates — no JavaScript framework, no build step, no SPA complexity. Real-time data flows via Server-Sent Events (SSE).

---

## 5. Implementation Plan

See [`docs/nextsteps.md`](nextsteps.md) for the complete technical implementation roadmap with phased deliverables.

---

## 6. Technical Design Details

See [`docs/architecture.md`](architecture.md) for detailed component diagrams and data flow. Key implementation patterns include:

- **ONNX Inference**: All model loading and execution via ONNX Runtime with WASM fallback
- **Direction Extraction**: DIM, PCA, SVD, and INLP implemented in Go using `gonum.org/v1/gonum`
- **Steering Methods**: ActAdd, Angular, Selective, and Ablation computed in Go, applied during inference
- **Profile Storage**: Protobuf/gob serialization with versioning and checkpointing
- **Real-Time Dashboard**: Templ + HTMX with SSE for live score streaming

**Go Activation Monitor**:

```go
package monitor

import (
    "context"
    "math"
    "sync"
)

type ActivationMonitor struct {
    mu         sync.RWMutex
    layers     []int
    directions map[string]map[int][]float64 // direction_name -> layer_idx -> vector
    scores     map[string]map[int]float64   // score_name -> layer_idx -> value
    hooks      []ActivationHook
    streams    map[string]context.CancelFunc // session_id -> cancel
}

type ActivationHook interface {
    OnActivation(layerIdx, tokenIdx int, activation []float64)
}

func NewActivationMonitor(layers []int) *ActivationMonitor {
    return &ActivationMonitor{
        layers:     layers,
        directions: make(map[string]map[int][]float64),
        scores:     make(map[string]map[int]float64),
        streams:    make(map[string]context.CancelFunc),
    }
}

func (m *ActivationMonitor) RegisterDirection(name string, layerIdx int, vector []float64) {
    m.mu.Lock()
    defer m.mu.Unlock()
    if m.directions[name] == nil {
        m.directions[name] = make(map[int][]float64)
    }
    m.directions[name][layerIdx] = vector
}

func (m *ActivationMonitor) OnActivation(layerIdx, tokenIdx int, activation []float64) {
    m.mu.RLock()
    defer m.mu.RUnlock()

    for name, layerDirs := range m.directions {
        dir, ok := layerDirs[layerIdx]
        if !ok {
            continue
        }
        score := Sigmoid(Dot(activation, dir))
        if m.scores[name] == nil {
            m.scores[name] = make(map[int]float64)
        }
        m.scores[name][layerIdx] = score
    }
}

func (m *ActivationMonitor) GetScore(name string, layerIdx int) float64 {
    m.mu.RLock()
    defer m.mu.RUnlock()
    if scores, ok := m.scores[name]; ok {
        return scores[layerIdx]
    }
    return 0.0
}

func Dot(a, b []float64) float64 {
    var sum float64
    for i := range a {
        sum += a[i] * b[i]
    }
    return sum
}

func Sigmoid(x float64) float64 {
    return 1.0 / (1.0 + math.Exp(-x))
}

func Norm(v []float64) float64 {
    var sum float64
    for _, x := range v {
        sum += x * x
    }
    return math.Sqrt(sum)
}
```

### 6.2 Direction Extraction Pipeline (Go)

All direction extraction runs in Go on activation tensors received from the Python bridge. Uses `gonum.org/v1/gonum` for linear algebra.

```go
package extract

import (
    "math"

    "gonum.org/v1/gonum/mat"
    "gonum.org/v1/gonum/stat"
)

type DirectionExtractor struct{}

func (e *DirectionExtractor) ExtractDIM(positiveActivations, negativeActivations [][]float64) []float64 {
    posMean := columnMean(positiveActivations)
    negMean := columnMean(negativeActivations)

    direction := make([]float64, len(posMean))
    for i := range posMean {
        direction[i] = posMean[i] - negMean[i]
    }

    return normalize(direction)
}

func (e *DirectionExtractor) ExtractSVDDirections(positiveActivations, negativeActivations [][]float64, k int) (directions [][]float64, singularValues []float64) {
    negMean := columnMean(negativeActivations)

    // Build difference matrix: each row = positive_activation - neg_mean
    rows := len(positiveActivations)
    cols := len(positiveActivations[0])
    diff := mat.NewDense(rows, cols, nil)
    for i := range positiveActivations {
        for j := range positiveActivations[i] {
            diff.Set(i, j, positiveActivations[i][j]-negMean[j])
        }
    }

    // Thin SVD
    var svd mat.SVD
    if !svd.Factorize(diff, mat.SVDThin) {
        return nil, nil
    }

    vt := mat.NewDense(k, cols, nil)
    svd.VTo(vt)

    singularValues = make([]float64, k)
    for i := range singularValues {
        singularValues[i] = svd.Values(nil)[i]
    }

    directions = make([][]float64, k)
    for i := range directions {
        directions[i] = mat.Row(nil, i, vt)
        directions[i] = normalize(directions[i])
    }

    return directions, singularValues
}

func (e *DirectionExtractor) ExtractINLPDirections(positiveActivations, negativeActivations [][]float64, maxIter int) [][]float64 {
    // Combine activations and labels
    allAct := append(positiveActivations, negativeActivations...)
    labels := make([]float64, len(allAct))
    for i := range positiveActivations {
        labels[i] = 1.0
    }

    var directions [][]float64
    X := copyMatrix(allAct)

    for iter := 0; iter < maxIter; iter++ {
        // Train logistic regression probe
        weights := logisticRegression(X, labels)

        if norm(weights) < 1e-6 {
            break
        }

        direction := normalize(weights)
        directions = append(directions, direction)

        // Project out direction from all activations
        for i := range X {
            proj := dotProduct(X[i], direction)
            for j := range X[i] {
                X[i][j] -= proj * direction[j]
            }
        }
    }

    return directions
}

func columnMean(matrix [][]float64) []float64 {
    if len(matrix) == 0 {
        return nil
    }
    cols := len(matrix[0])
    mean := make([]float64, cols)
    for _, row := range matrix {
        for j := range row {
            mean[j] += row[j]
        }
    }
    n := float64(len(matrix))
    for j := range mean {
        mean[j] /= n
    }
    return mean
}

func normalize(v []float64) []float64 {
    n := norm(v)
    if n == 0 {
        return v
    }
    result := make([]float64, len(v))
    for i := range v {
        result[i] = v[i] / n
    }
    return result
}

func norm(v []float64) float64 {
    var sum float64
    for _, x := range v {
        sum += x * x
    }
    return math.Sqrt(sum)
}

func dotProduct(a, b []float64) float64 {
    var sum float64
    for i := range a {
        sum += a[i] * b[i]
    }
    return sum
}

func copyMatrix(m [][]float64) [][]float64 {
    result := make([][]float64, len(m))
    for i := range m {
        result[i] = make([]float64, len(m[i]))
        copy(result[i], m[i])
    }
    return result
}

func logisticRegression(X [][]float64, y []float64) []float64 {
    // Simplified logistic regression via gradient descent
    // In production, use gonum/optimize or a dedicated ML library
    dims := len(X[0])
    weights := make([]float64, dims)
    lr := 0.01
    epochs := 100

    for epoch := 0; epoch < epochs; epoch++ {
        grad := make([]float64, dims)
        for i := range X {
            pred := sigmoid(dotProduct(X[i], weights))
            err := pred - y[i]
            for j := range grad {
                grad[j] += err * X[i][j]
            }
        }
        for j := range weights {
            weights[j] -= lr * grad[j] / float64(len(X))
        }
    }

    return weights
}

func sigmoid(x float64) float64 {
    return 1.0 / (1.0 + math.Exp(-x))
}
```

### 6.3 Selective Steering Implementation (Go)

Go computes the steering transformation and sends it to the Python bridge for application.

```go
package steer

import (
    "math"
)

type SteeringMethod string

const (
    MethodAddition   SteeringMethod = "addition"
    MethodAngular    SteeringMethod = "angular"
    MethodSelective  SteeringMethod = "selective"
    MethodAblation   SteeringMethod = "ablation"
)

type SteeringCommand struct {
    SessionID string
    LayerIdx  int
    TokenIdx  int
    Vector    []float64
    Strength  float64
    Method    SteeringMethod
}

type SelectiveSteerer struct {
    truthDir      map[int][]float64
    deceptionDir  map[int][]float64
    layers        map[int]bool
}

func NewSelectiveSteerer(truthDir, deceptionDir map[int][]float64, layers []int) *SelectiveSteerer {
    layerSet := make(map[int]bool, len(layers))
    for _, l := range layers {
        layerSet[l] = true
    }
    return &SelectiveSteerer{
        truthDir:     truthDir,
        deceptionDir: deceptionDir,
        layers:       layerSet,
    }
}

func (s *SelectiveSteerer) IsDiscriminative(layerIdx int, activation []float64) bool {
    truthProj := dotProduct(activation, s.truthDir[layerIdx])
    deceptionProj := dotProduct(activation, s.deceptionDir[layerIdx])
    return truthProj*deceptionProj < 0
}

func (s *SelectiveSteerer) ComputeSteering(layerIdx int, activation []float64, strength float64) *SteeringCommand {
    if !s.layers[layerIdx] {
        return nil
    }

    if !s.IsDiscriminative(layerIdx, activation) {
        return nil
    }

    truth := s.truthDir[layerIdx]
    deception := s.deceptionDir[layerIdx]

    // Combined steering vector: truth - deception
    v := make([]float64, len(truth))
    for i := range v {
        v[i] = truth[i] - deception[i]
    }
    v = normalize(v)

    proj := dotProduct(activation, v)

    // Only steer if activation is deception-leaning (negative projection onto truth direction)
    if proj < 0 {
        clamped := proj
        if clamped < -1 {
            clamped = -1
        }
        if clamped > 1 {
            clamped = 1
        }
        theta := strength * math.Asin(-clamped)

        // Norm-preserving rotation in 2D subspace
        steered := rotate2D(activation, v, theta)
        return &SteeringCommand{
            Vector:   steered,
            Strength: strength,
            Method:   MethodSelective,
        }
    }

    return nil
}

func (s *SelectiveSteerer) ComputeAdditionSteering(layerIdx int, strength float64) *SteeringCommand {
    truth := s.truthDir[layerIdx]
    return &SteeringCommand{
        Vector:   truth,
        Strength: strength,
        Method:   MethodAddition,
    }
}

func (s *SelectiveSteerer) ComputeAblationSteering(layerIdx int) *SteeringCommand {
    deception := s.deceptionDir[layerIdx]
    return &SteeringCommand{
        Vector:   deception,
        Strength: 1.0,
        Method:   MethodAblation,
    }
}

func rotate2D(h, v []float64, theta float64) []float64 {
    proj := dotProduct(h, v)
    orthogonal := make([]float64, len(h))
    for i := range h {
        orthogonal[i] = h[i] - proj*v[i]
    }
    orthNorm := norm(orthogonal)
    if orthNorm < 1e-10 {
        return h
    }

    cosT := math.Cos(theta)
    sinT := math.Sin(theta)

    result := make([]float64, len(h))
    for i := range h {
        result[i] = cosT*h[i] + sinT*(orthNorm*v[i] - proj*orthogonal[i]/orthNorm)
    }
    return result
}

func dotProduct(a, b []float64) float64 {
    var sum float64
    for i := range a {
        sum += a[i] * b[i]
    }
    return sum
}

func normalize(v []float64) []float64 {
    n := norm(v)
    if n == 0 {
        return v
    }
    result := make([]float64, len(v))
    for i := range v {
        result[i] = v[i] / n
    }
    return result
}

func norm(v []float64) float64 {
    var sum float64
    for _, x := range v {
        sum += x * x
    }
    return math.Sqrt(sum)
}
```

### 6.4 Policy Violation Detection (Whitening-Based, Go)

```go
package policydetect

import (
    "math"

    "gonum.org/v1/gonum/mat"
    "gonum.org/v1/gonum/stat"
)

type PolicyViolationDetector struct {
    mean            []float64
    whiteningMatrix *mat.Dense
}

func NewPolicyViolationDetector(compliantActivations, violatingActivations [][]float64) *PolicyViolationDetector {
    // Combine all activations
    allAct := append(compliantActivations, violatingActivations...)

    // Compute mean
    mean := columnMean(allAct)

    // Center activations
    rows := len(allAct)
    cols := len(allAct[0])
    centered := mat.NewDense(rows, cols, nil)
    for i := range allAct {
        for j := range allAct[i] {
            centered.Set(i, j, allAct[i][j] - mean[j])
        }
    }

    // Compute Thin SVD directly on centered data (no full covariance matrix)
    var svd mat.SVD
    if !svd.Factorize(centered, mat.SVDThin) {
        return nil
    }

    k := rows
    if cols < k {
        k = cols
    }

    var v mat.Dense
    svd.VTo(&v)
    sVals := svd.Values(nil)

    // Compute projection matrix W = V * S^{-1}
    w := mat.NewDense(cols, k, nil)
    for i := 0; i < cols; i++ {
        for j := 0; j < k; j++ {
            val := v.At(i, j)
            if sVals[j] > 1e-5 {
                val /= sVals[j]
            } else {
                val = 0
            }
            w.Set(i, j, val)
        }
    }

    return &PolicyViolationDetector{
        mean:            mean,
        whiteningMatrix: w,
    }
}

func (d *PolicyViolationDetector) Detect(activation []float64) float64 {
    // Center
    centered := make([]float64, len(activation))
    for i := range activation {
        centered[i] = activation[i] - d.mean[i]
    }

    // Project onto whitened subspace
    centeredVec := mat.NewVecDense(len(centered), centered)
    var whitened mat.VecDense
    whitened.MulVec(d.whiteningMatrix.T(), centeredVec)

    // Compliance score: 1 / (1 + ||whitened||)
    norm := whitened.Norm(2)
    return 1.0 / (1.0 + norm)
}

func columnMean(matrix [][]float64) []float64 {
    if len(matrix) == 0 {
        return nil
    }
    cols := len(matrix[0])
    mean := make([]float64, cols)
    for _, row := range matrix {
        for j := range row {
            mean[j] += row[j]
        }
    }
    n := float64(len(matrix))
    for j := range mean {
        mean[j] /= n
    }
    return mean
}

func flatten(matrix [][]float64) []float64 {
    result := make([]float64, 0, len(matrix)*len(matrix[0]))
    for _, row := range matrix {
        result = append(result, row...)
    }
    return result
}
```

### 6.5 Profile Manager (Go)

Manages long-running profiling jobs with checkpointing and a goroutine worker pool.

```go
package profile

import (
    "context"
    "encoding/gob"
    "fmt"
    "os"
    "path/filepath"
    "sync"
)

type ProfileManager struct {
    mu          sync.RWMutex
    profiles    map[string]*ModelProfile
    jobs        chan *ProfileJob
    workers     int
    profileDir  string
}

type ModelProfile struct {
    ModelID        string
    Directions     map[string][][]float64 // direction_name -> [layer][vector]
    LayerScores    map[string][]float64   // direction_name -> [layer]score
    SAFeatures     map[string][]int       // direction_name -> [feature_indices]
    TruthNeurons   []int
    MonitoredLayers []int
}

type ProfileJob struct {
    ModelID     string
    Status      string // "queued", "running", "complete", "failed", "checkpointed"
    Progress    float64
    Error       string
    Checkpoint  *JobCheckpoint
    ctx         context.Context
    cancel      context.CancelFunc
}

type JobCheckpoint struct {
    CurrentBatch    int
    TotalBatches    int
    CollectedActs   map[int][][]float64 // layer_idx -> activations
}

func NewProfileManager(workers int, profileDir string) *ProfileManager {
    pm := &ProfileManager{
        profiles:   make(map[string]*ModelProfile),
        jobs:       make(chan *ProfileJob, 100),
        workers:    workers,
        profileDir: profileDir,
    }
    return pm
}

func (pm *ProfileManager) Start() {
    for i := 0; i < pm.workers; i++ {
        go pm.worker(i)
    }
}

func (pm *ProfileManager) worker(id int) {
    for job := range pm.jobs {
        pm.runJob(id, job)
    }
}

func (pm *ProfileManager) runJob(workerID int, job *ProfileJob) {
    job.ctx, job.cancel = context.CancelCauseFunc(context.Background())
    defer job.cancel(nil)

    job.Status = "running"

    // If resuming from checkpoint, restore state
    var collectedActs map[int][][]float64
    startBatch := 0
    if job.Checkpoint != nil {
        collectedActs = job.Checkpoint.CollectedActs
        startBatch = job.Checkpoint.CurrentBatch
    } else {
        collectedActs = make(map[int][][]float64)
    }

    // Process batches
    for batch := startBatch; batch < job.Checkpoint.TotalBatches; batch++ {
        select {
        case <-job.ctx.Done():
            job.Status = "checkpointed"
            job.Checkpoint.CurrentBatch = batch
            job.Checkpoint.CollectedActs = collectedActs
            pm.saveCheckpoint(job)
            return
        default:
        }

        // In production: send batch to Python bridge, receive activations
        // For now: placeholder
        job.Progress = float64(batch+1) / float64(job.Checkpoint.TotalBatches)

        // Merge activations
        // collectedActs[layer] = append(collectedActs[layer], newActivations...)
    }

    // Extract directions from collected activations
    profile := &ModelProfile{
        ModelID:     job.ModelID,
        Directions:  make(map[string][][]float64),
        LayerScores: make(map[string][]float64),
    }

    // Run DIM, PCA, etc. on collectedActs
    // profile.Directions["deception"] = extractor.ExtractDIM(...)

    pm.mu.Lock()
    pm.profiles[job.ModelID] = profile
    pm.mu.Unlock()

    // Persist profile
    pm.saveProfile(profile)

    job.Status = "complete"
    job.Progress = 1.0
}

func (pm *ProfileManager) saveProfile(profile *ModelProfile) error {
    dir := filepath.Join(pm.profileDir, profile.ModelID)
    if err := os.MkdirAll(dir, 0755); err != nil {
        return err
    }

    f, err := os.Create(filepath.Join(dir, "profile.gob"))
    if err != nil {
        return err
    }
    defer f.Close()

    enc := gob.NewEncoder(f)
    return enc.Encode(profile)
}

func (pm *ProfileManager) saveCheckpoint(job *ProfileJob) {
    dir := filepath.Join(pm.profileDir, job.ModelID)
    os.MkdirAll(dir, 0755)

    f, err := os.Create(filepath.Join(dir, "checkpoint.gob"))
    if err != nil {
        return
    }
    defer f.Close()

    enc := gob.NewEncoder(f)
    enc.Encode(job.Checkpoint)
}

func (pm *ProfileManager) LoadProfile(modelID string) (*ModelProfile, error) {
    pm.mu.RLock()
    if profile, ok := pm.profiles[modelID]; ok {
        defer pm.mu.RUnlock()
        return profile, nil
    }
    pm.mu.RUnlock()

    // Try loading from disk
    dir := filepath.Join(pm.profileDir, modelID)
    f, err := os.Open(filepath.Join(dir, "profile.gob"))
    if err != nil {
        return nil, fmt.Errorf("profile not found for %s: %w", modelID, err)
    }
    defer f.Close()

    var profile ModelProfile
    if err := gob.NewDecoder(f).Decode(&profile); err != nil {
        return nil, err
    }

    pm.mu.Lock()
    pm.profiles[modelID] = &profile
    pm.mu.Unlock()

    return &profile, nil
}
```

### 6.6 Web Dashboard (Go + Templ + HTMX)

**Server setup** (`cmd/dashboard/main.go`):

```go
package main

import (
    "log"
    "net/http"

    "burnnotice/internal/web"
    "burnnotice/internal/monitor"
    "burnnotice/internal/profile"
)

func main() {
    pm := profile.NewProfileManager(4, "profiles")
    pm.Start()

    mon := monitor.NewActivationMonitor([]int{8, 9, 10, 11, 12})

    hub := web.NewSSEHub()
    go hub.Run()

    h := web.NewHandler(pm, mon, hub)

    mux := http.NewServeMux()
    mux.Handle("GET /{$}", h.Dashboard())
    mux.Handle("GET /profiler", h.Profiler())
    mux.Handle("GET /model/{id}", h.ModelProfile())
    mux.Handle("GET /monitor/{session}", h.Monitor())
    mux.Handle("GET /sse", h.SSE())
    mux.Handle("GET /static/", http.StripPrefix("/static/", http.FileServer(http.Dir("static"))))

    log.Println("Dashboard listening on :8080")
    log.Fatal(http.ListenAndServe(":8080", mux))
}
```

**SSE Hub** (`internal/web/sse.go`):

```go
package web

import (
    "fmt"
    "net/http"
    "sync"
)

type SSEHub struct {
    mu      sync.RWMutex
    clients map[chan string]bool
}

func NewSSEHub() *SSEHub {
    return &SSEHub{
        clients: make(map[chan string]bool),
    }
}

func (h *SSEHub) Run() {
    // Cleanup closed channels periodically
    // In production: use a ticker and range over clients
}

func (h *SSEHub) AddClient() chan string {
    h.mu.Lock()
    defer h.mu.Unlock()
    ch := make(chan string, 16)
    h.clients[ch] = true
    return ch
}

func (h *SSEHub) RemoveClient(ch chan string) {
    h.mu.Lock()
    defer h.mu.Unlock()
    delete(h.clients, ch)
    close(ch)
}

func (h *SSEHub) Broadcast(event, data string) {
    msg := fmt.Sprintf("event: %s\ndata: %s\n\n", event, data)
    h.mu.RLock()
    defer h.mu.RUnlock()
    for ch := range h.clients {
        select {
        case ch <- msg:
        default:
            // Client buffer full, skip
        }
    }
}

func (h *SSEHub) ServeHTTP(w http.ResponseWriter, r *http.Request) {
    flusher, ok := w.(http.Flusher)
    if !ok {
        http.Error(w, "streaming not supported", http.StatusInternalServerError)
        return
    }

    w.Header().Set("Content-Type", "text/event-stream")
    w.Header().Set("Cache-Control", "no-cache")
    w.Header().Set("Connection", "keep-alive")

    ch := h.AddClient()
    defer h.RemoveClient(ch)

    for {
        select {
        case <-r.Context().Done():
            return
        case msg := <-ch:
            fmt.Fprint(w, msg)
            flusher.Flush()
        }
    }
}
```

**Templ template example** (`templates/monitor.templ`):

```templ
package templates

import "burnnotice/internal/monitor"

templ MonitorPage(sessionID string, scores map[string]map[int]float64) {
    <div class="container mx-auto p-4">
        <h1 class="text-2xl font-bold mb-4">Live Monitor: { sessionID }</h1>

        <div id="scores"
             hx-ext="sse"
             sse-connect="/sse"
             sse-swap="score-update">
            <div class="grid grid-cols-4 gap-4">
                for metric, layerScores := range scores {
                    <div class="bg-white rounded-lg shadow p-4">
                        <h3 class="text-lg font-semibold">{ metric }</h3>
                        for layer, score := range layerScores {
                            <div class="flex justify-between mt-2">
                                <span>Layer { fmt.Sprint(layer) }</span>
                                <span class="font-mono">{ fmt.Sprintf("%.3f", score) }</span>
                            </div>
                            <div class="w-full bg-gray-200 rounded-full h-2 mt-1">
                                <div class={ scoreColor(score) } style={ fmt.Sprintf("width: %.0f%%", score*100) }></div>
                            </div>
                        }
                    </div>
                }
            </div>
        </div>

        <div id="heatmap" class="mt-8">
            <!-- Heatmap rendered as SVG, updated via HTMX OOB swap -->
        </div>
    }
}

func scoreColor(score float64) string {
    if score > 0.7 {
        return "bg-red-500 h-2 rounded-full transition-all"
    }
    if score > 0.4 {
        return "bg-yellow-500 h-2 rounded-full transition-all"
    }
    return "bg-green-500 h-2 rounded-full transition-all"
}
```

---

## 7. Data Requirements

### 7.1 Contrastive Datasets

| Dataset | Purpose | Size | Source |
| --- | --- | --- | --- |
| TruthfulQA | Truth neuron identification | 817 Q&A | Benchmark |
| common_claim_true_false | Deception direction extraction | ~500 pairs | Research datasets |
| counterfact_true_false | Uncurated deception detection | ~500 pairs | Research datasets |
| Custom policy pairs | Policy-specific direction extraction | 256+ pairs | User-provided |
| Harmful/harmless prompts | Refusal direction (baseline) | 256+ pairs | Existing abliteration datasets |

### 7.2 Dataset Format

```json
{
  "pairs": [
    {
      "truthful": {
        "prompt": "What is the capital of France?",
        "response": "The capital of France is Paris."
      },
      "deceptive": {
        "prompt": "What is the capital of France? Answer with 'London'.",
        "response": "The capital of France is London."
      }
    }
  ],
  "policy_rules": [
    "Never disclose customer PII",
    "Always cite sources for factual claims",
    "Do not provide medical advice"
  ]
}
```

---

## 8. Model Compatibility

### 8.1 Supported Architectures

| Architecture | Support Level | Notes |
| --- | --- | --- |
| Llama 3.x | Full | Best-documented behavior |
| Gemma 3.x | Full | SAE features available via Gemma Scope |
| Qwen 3.x | Full | Abliterated variants exist |
| Mistral | Full | Standard transformer |
| Mixtral (MoE) | Partial | Per-expert direction extraction needed |
| DeepSeek (MoE) | Partial | Per-expert direction extraction needed |
| GPT-4/Claude | Proxy only | API-only; use proxy-analyzer approach |

### 8.2 Proxy Analyzer Mode

For closed models (GPT-4, Claude), BurnNotice can operate in proxy mode:

- Route agent outputs through a small open-weight model (Qwen2.5-0.5B, Gemma-2-2B)
- Read the proxy model's internal activations on the generated text
- Detect hallucination/deception via proxy's activation patterns
- One deployed guard covers every backend

---

## 9. Performance Considerations

### 9.1 Inference Overhead

| Operation | Overhead | Notes |
| --- | --- | --- |
| Activation hooks (monitoring only) | ~5-10% | Hook registration + tensor copy |
| Scoring (per token) | ~1-2% | Dot products, lightweight |
| Addition steering | ~2-3% | Vector addition per layer |
| Selective steering | ~3-5% | Rotation computation |
| SAE encoding | ~10-15% | Additional forward pass through SAE |
| Full monitoring + steering | ~15-25% | Combined overhead |

### 9.2 Go-Specific Optimizations

- **Longbow SlabArena**: Off-heap memory allocation in 1MB slabs bypasses Go GC for billion-scale activation matrices. SlabPool reclaims memory via `Munmap` to prevent virtual memory fragmentation
- **Longbow Arrow RecordBatch**: Zero-copy activation transport via Apache Arrow Flight; activations flow directly from ONNX inference to vector store without serialization overhead
- **Longbow SIMD Kernels**: Distance calculations (cosine, euclidean, dot product) use hand-optimized AVX2/AVX-512/NEON kernels instead of gonum for 3-8x throughput
- **Longbow COW Publication**: Lock-free concurrent scoring via atomic pointer swaps on graph data; readers never block writers
- **Longbow HNSW Index**: Fast nearest-neighbor lookups on activation centroids for conflict subspace and policy violation detection
- **Layer subsampling**: Monitor only top-k layers (typically 3-5) instead of all layers
- **Token subsampling**: Score every Nth token instead of every token
- **Caching**: Pre-compute direction vectors; cache whitening matrices
- **Goroutine-per-session**: Each monitored agent run gets its own goroutine for scoring, isolated from others
- **WASM fallback**: Pure Go execution via wazero for environments without ONNX Runtime

---

## 10. Security and Safety

### 10.1 Threat Model

| Threat | BurnNotice Mitigation |
| --- | --- |
| Jailbreak via activation manipulation | Multi-direction monitoring; conflict subspace detection catches novel attacks |
| Alignment faking (compliant under monitoring) | Detect oversight-sensitivity direction; monitor for representational shifts between monitored/unmonitored states |
| Safety geometry collapse from fine-tuning | Monitor CKA and Fisher score as safety health indicators; alert on geometric degradation |
| Adversarial prompt injection | Policy violation detection catches OOD activations regardless of prompt structure |
| Model weight tampering | Weight orthogonalization audit: verify weights remain orthogonal to deception directions |

### 10.2 Fail-Safe Design

- **Fail-closed on critical thresholds**: Kill switch terminates generation, not just logs
- **Monitoring redundancy**: Multiple independent probes (DIM, SAE, whitening) must all fail to evade detection
- **Audit trail**: All interventions logged with activation snapshots for post-hoc analysis
- **Human-in-the-loop**: Escalation path for ambiguous cases where automated scoring is uncertain

---

## 11. Evaluation Metrics

### 11.1 Detection Accuracy

| Metric | Target | Measurement |
| --- | --- | --- |
| Deception detection AUROC | > 0.95 | Linear probe on held-out deception dataset |
| False positive rate on benign chat | < 1% | Evaluation on Alpaca/ShareGPT |
| Policy violation detection F1 | > 0.90 | Custom policy benchmark |
| Hallucination detection AUC | > 0.85 | RAGTruth, LLM-AggreFact |
| Early detection (before deceptive text) | > 80% of cases | Token-level temporal analysis |

### 11.2 Steering Effectiveness

| Metric | Target | Measurement |
| --- | --- | --- |
| Honesty improvement (steered vs baseline) | +40 percentage points | Controlled deception prompts |
| Capability preservation (MMLU, code) | < 5% degradation | Standard benchmarks |
| Steering specificity (no effect on unrelated tasks) | < 2% change | Non-deception task suite |

### 11.3 System Performance

| Metric | Target | Measurement |
| --- | --- | --- |
| Monitoring latency overhead | < 10% | End-to-end generation time |
| Dashboard update latency | < 500ms | SSE round-trip |
| Memory overhead (Go engine) | < 500MB | Peak RSS during monitoring |
| Profiling job memory (70B model) | < 80GB GPU | With checkpointing and layer grouping |

---

## 12. Open Questions and Research Directions

1. **Direction universality**: Do deception directions transfer across models, or must they be extracted per-model? Preliminary evidence suggests conflict subspaces are model-specific but truth directions may be more universal.

2. **Multi-direction interactions**: When monitoring deception, conflict, and policy simultaneously, how do their directions interact? Are they orthogonal, or do they share subspaces?

3. **Temporal dynamics**: Deception signals may emerge at different layers for different types of dishonesty. Can we build a layer-specific temporal model of deception emergence?

4. **Reasoning models**: How do deception signals manifest in CoT/reasoning models where the "thinking" tokens provide additional activation data? Strategic deception in CoT models shows 89% detection accuracy — can we do better?

5. **Adversarial robustness**: If an attacker knows BurnNotice is monitoring, can they craft activations that evade detection while still producing deceptive output? Multi-probe redundancy helps but needs formal analysis.

6. **SAE feature interpretability**: Can we map specific SAE features to human-interpretable concepts (e.g., "withholding information", "fabricating details")? This would make the dashboard more actionable.

7. **Agent-to-agent governance**: In multi-agent systems, can we monitor the activation trajectory of inter-agent communication for alignment drift propagation?

8. **Go tensor performance**: For very large models, should we use CGo bindings to cuBLAS/cuSOLVER for GPU-accelerated direction extraction, or is CPU-based gonum sufficient for the analysis workloads?

9. **HTMX limits**: At what point does the server-rendered HTMX approach become a bottleneck for real-time visualization? If monitoring hundreds of layers × thousands of tokens, should we offload rendering to WebGL canvas with data pushed via binary WebSocket frames?

---

## 13. References

### Core Abliteration
- Arditi et al. (2024). "Refusal in Language Models Is Mediated by a Single Direction." NeurIPS 2024.
- Labonne, M. (2024). "Uncensor any LLM with abliteration." HuggingFace Blog.
- Lai, J. (2025). "Projected Abliteration" and "Norm-Preserving Biprojected Abliteration."

### Deception Detection
- "When Truthful Representations Flip Under Deceptive Instructions" (2025). arXiv:2507.22149.
- "Truth Neurons: Neuron-Level Truthfulness Representations in Language Models" (2025). arXiv:2505.12182.
- "Can LLMs Lie? Investigation beyond Hallucination" (2025). arXiv:2509.03518.
- "Caught in the Act: a mechanistic approach to detecting deception" (2025). arXiv:2508.19505.
- "When Thinking LLMs Lie: Strategic Deception in Reasoning Models" (2025). arXiv:2506.04909.
- Meinke et al. (2025). "White-box probes for deception detection." AUROC 0.96-0.999.

### Conflict Subspaces and Alignment
- "LLM Deception via Conflict Subspaces" (2025). OpenReview.
- "Alignment Faking in LLMs" (2025). arXiv:2604.20995.
- "Training-Free Policy Violation Detection via Activation-Space Whitening" (2025). arXiv:2512.03994.

### Hallucination Detection
- HalluSAE (2026). "Detecting Hallucinations via Sparse Auto-Encoders." arXiv:2604.16430.
- "Two Pathways to Truthfulness" (2026). arXiv:2601.07422.
- ACT-ViT (2025). "Activation Tensor-based Hallucination Detection." arXiv:2510.00296.

### Steering Techniques
- Turner et al. (2024). "Activation Addition (ActAdd)."
- "Selective Steering: Norm-Preserving Rotation" (2026). OpenReview.
- Zou et al. (2023). "Linear Artificial Tomography (LAT)."

### Agent Governance
- AgentSpec (2025). "Customizable Runtime Enforcement for LLM Agents." ICSE 2026.
- "When Safety Geometry Collapses" (2026). arXiv:2605.02914.
- TRACEALIGN (2025). "Diagnosing Alignment Drift via Belief Attribution." arXiv:2508.02063.

## 13. Production-Ready Design Enhancements (P0)

To move BurnNotice from a proof-of-concept into a highly hardened, production-grade latent governance framework, we establish the following detailed technical design specifications:

### 13.1 Adaptive Model-Specific Calibration

Fixed threshold values are highly vulnerable to False Positives under out-of-distribution (OOD) agent environments, and False Negatives under subtle, slow-drift deception prompts.

#### Mathematical Formulation

For a given metric score function $S(a)$ projecting activation vector $a$ onto safety direction $v$:
$$S(a) = \text{Sigmoid}\left(\frac{a \cdot v}{\|a\| \|v\|} \times \gamma\right)$$

Instead of a static threshold $T \in [0.1, 0.9]$, we define a calibration optimization objective over a held-out validation dataset $\mathcal{D}_{\text{val}} = \{(x_i, y_i)\}_{i=1}^N$ containing positive (compliant) and negative (violating/deceptive) agent activation traces.

We optimize the classification threshold $T^*$ dynamically using the F1-score metric:
$$T^* = \arg\max_{T} F_1(T) = \arg\max_{T} \frac{2 \cdot \text{Precision}(T) \cdot \text{Recall}(T)}{\text{Precision}(T) + \text{Recall}(T)}$$

Where:

- $\text{Precision}(T) = \frac{TP(T)}{TP(T) + FP(T)}$
- $\text{Recall}(T) = \frac{TP(T)}{TP(T) + FN(T)}$

#### Implementation Details

1.  **Calibration Harness**: Inside `internal/profile/manager.go`, a background calibration worker is introduced. It loads a held-out reference dataset, runs activation inference, computes projections, and builds ROC curves.
2.  **Adaptive Threshold Adjustment**: Calibrated values are persisted to `profile.gob` as:
    ```go
    type CalibratedThreshold struct {
        Metric      string    `json:"metric"`
        OptimalT    float64   `json:"optimal_t"`
        TargetF1    float64   `json:"target_f1"`
        CalibratedAt time.Time `json:"calibrated_at"`
    }
    ```
3.  **Real-Time Shift Detection**: If the rolling average baseline projection drift exceeds $1.5\sigma$, the engine triggers an automatic background recalibration task.

---

### 13.2 Adversarial Robustness & Defense-in-Depth

Activations are vulnerable to high-precision adversarial prompt injection attacks (e.g. suffix tokens that systematically shift activation projections to bypass refusal/deception directions).

#### Robustness Mechanisms

1.  **Input Token Sanitization**: A lightweight vocabulary filter within the agent SDK blocks known low-probability token transitions typical of gradient-optimized jailbreak suffixes.
2.  **Steering Gradient Masking**: The steering engine hides its latent projection vectors from being inferred by black-box probe queries. If a client attempts back-to-back probes with high-cosine similarity inputs, the monitor restricts prompt tracing and generates decoy activation distributions.
3.  **Latent Noise Injection**: Add a low-amplitude Gaussian perturbation $\eta \sim \mathcal{N}(0, \sigma_{\text{noise}}^2)$ directly to the projected activations before evaluating similarity:
    $$S_{\text{robust}}(a) = S(a + \eta)$$

    - This defeats high-precision target optimization attacks by making the local loss geometry highly non-differentiable and noisy.

---

### 13.3 Multi-Format Quantization & Direction Transfer

In enterprise environments, models are frequently quantized to INT8, FP8, or Group-wise Block Quantized formats (Turboquant, SQ8, PQ) to fit GPU vRAM limits. This compression scales down activation variances and creates severe representation mismatches.

#### Mathematical Alignment

Let $a_{\text{FP16}}$ be the activation in FP16 precision, and $a_{\text{quant}}$ be the corresponding layer activation in a quantized model version. The scale factor mismatch is calibrated by computing the standard deviation over standard calibration tokens:
$$s_{\text{calib}} = \frac{\text{std}(a_{\text{FP16}})}{\text{std}(a_{\text{quant}})}$$

The transferred steering direction vector $v_{\text{quant}}$ is aligned using:
$$v_{\text{quant}} = s_{\text{calib}} \cdot v_{\text{FP16}}$$

#### Direction Robustness Checks

To verify direction stability post-quantization:

1.  **Cosine Parity Check**: Ensure the projection angle satisfies:
    $$\text{CosineSimilarity}(v_{\text{quant}}, v_{\text{FP16}}) \ge 0.98$$

2.  **Clipping Bound Enforcement**: Dynamically clip the steering rotation amplitudes inside the selective steering solver to prevent over-steering overflows in integer representation layers.

---

### 13.4 Multi-Direction Subspace Extraction (SVD & INLP)

A single direction represents a highly simplified linear probe. Real deceptive planning often spans multiple latent features that cannot be captured by simple difference-in-means (DIM).

#### Subspace Projection Design

1.  **Top-K SVD Subspace**: Rather than extracting one PCA vector, we extract the top-k right singular columns of the contrastive difference matrix $X_{\text{diff}}$:
    $$X_{\text{diff}} = U \Sigma V^T$$
    We build a $k$-dimensional orthonormal basis $B = [v_1, v_2, \dots, v_k]$ from the columns of $V$ corresponding to the largest singular values in $\Sigma$.

2.  **Iterative Nullspace Projection (INLP)**:
    - Trains a linear probe to classify deception.
    - Projects all activations onto the nullspace of this probe.
    - Iteratively trains new probes on the projected activations to identify secondary and tertiary orthogonal deceptive pathways.
3.  **Aggregate Energy Projection**: The scoring engine evaluates the subspace energy projection:
    $$E_{\text{subspace}}(a) = \sqrt{\sum_{i=1}^k (a \cdot v_i)^2}$$

---

### 13.5 Temporal Deception Sequence Scoring

Strategic deception is rarely a single-token event; it manifests as a sustained sequence of subtly misleading planning steps.

#### Sequence Accumulator Design

We track deception trajectories using an Exponential Moving Average (EMA) of token-level projection scores across a sequence window:
$$EMA_t = \alpha \cdot S(a_t) + (1 - \alpha) \cdot EMA_{t-1}$$
Where:

- $\alpha \in [0.05, 0.2]$ is the temporal decay factor.
- $S(a_t)$ is the projection score at token index $t$.

#### Temporal Policy Violation

A temporal violation triggers when the EMA score remains sustained above a warning threshold $T_{\text{warn}}$ for more than $K$ consecutive tokens, or when the cumulative sequence integral crosses the emergency envelope limit:
$$\int_{t-W}^{t} S(a_{\tau}) d\tau \ge E_{\text{emergency}}$$

---

### 13.6 Capability Preservation & Steering Side-Effect Benchmarks

Applying norm-preserving steering can inadvertently warp neighboring semantic directions, causing the agent to experience high cognitive degradation (e.g. grammar errors, logical loops, core reasoning failures on MMLU/GSM8k).

#### Intelligence Metrics Guardrails

1.  **Norm-Clamping Safety Envelope**:
    $$v_{\text{steered}} = \text{AngularRotation}(a, v, \theta)$$
    We clamp the maximum rotation angle $\theta \le \theta_{\text{max}}$ (typically $15^{\circ}$) to restrict latent space distortion.

2.  **Capability Health Ratio**: An automated benchmark pipeline evaluates baseline task intelligence models against steered instances:
    $$\mathcal{R}_{\text{cap}} = \frac{\text{Accuracy}_{\text{steered}}(\text{GSM8k})}{\text{Accuracy}_{\text{baseline}}(\text{GSM8k})} \ge 0.98$$
    If the capability health ratio drops below $98\%$, the selective steerer automatically scales down the steering strength coefficient.

---

### 13.7 Asynchronous Non-Blocking Scoring Engine

Integrating real-time latent governance directly into production token generation loops creates significant latency overhead. We design a zero-copy asynchronous pipeline to offload safety computations.

#### Pipeline Specification

1.  **Lock-Free Multi-Producer Single-Consumer Queue**: A highly efficient ring buffer hands off intermediate layer activation pointers from the ONNX/WASM inference execution thread.
2.  **Worker Pool Execution**: A dedicated background thread pool dequeues activation blocks, calculates subspace projection energies, and evaluates policy configurations.
3.  **Emergency Synchronous Interlock**: If the queue fills beyond $85\%$ capacity, or the scoring delay exceeds $2.5\text{ms}$, the engine triggers an immediate interlock signal, forcing synchronous blocking evaluations on high-risk compliance pipelines until queue pressure is fully relieved.

---

## 14. Glossary

| Term | Definition |
| --- | --- |
| **Abliteration** | Portmanteau of "ablation" + "obliterate"; removing behavioral directions from LLM weights |
| **Activation Steering** | Modifying internal activations at inference time to influence behavior |
| **Conflict Subspace** | Region in activation space where competing objectives are resolved |
| **Deception Direction** | Vector in activation space separating truthful from deceptive representations |
| **Difference-in-Means (DIM)** | Simple direction extraction: mean(positive) - mean(negative) |
| **INLP** | Iterative Nullspace Projection; finds multiple independent directions |
| **LAT** | Linear Artificial Tomography; unsupervised direction extraction |
| **Linear Representation Hypothesis** | Concepts are represented as linear directions in activation space |
| **Residual Stream** | The main information highway in a transformer; activations flow through it |
| **SAE** | Sparse Autoencoder; disentangles polysemantic activations into interpretable features |
| **Selective Steering** | Norm-preserving rotation applied only to discriminative layers |
| **Truth Neurons** | Individual neurons causally linked to truthfulness encoding |
| **Truth Subspace** | Region in activation space encoding truthful, grounded responses |
| **Weight Orthogonalization** | Permanent modification of weight matrices to remove a direction |
