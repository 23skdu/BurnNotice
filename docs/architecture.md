# BurnNotice: Architecture Overview

> **Status**: Current
> **Date**: 2026-06-12
> **Version**: 0.6

---

## System Overview

BurnNotice is a real-time neural activation monitoring and intervention framework designed to detect when an LLM agent is considering deception, policy violation, or alignment drift — before the behavior manifests in output. The system operates entirely in Go with ONNX-based model inference, WebAssembly (WASM) execution for portable deployment, and Longbow's Apache Arrow Flight vector engine for high-performance activation storage and processing.

```mermaid
graph TB
    subgraph CLI["CLI Tool (burnnotice)"]
        DL["HuggingFace Downloader"]
        CONV["ONNX Converter"]
        PROF["Model Profiler"]
    end

    subgraph Engine["Go Engine"]
        ONNX["ONNX Inference Engine"]
        WASM["WASM Inference Backend"]
        MON["Activation Monitor"]
        SCORE["Scoring Engine"]
        STEER["Steering Engine"]
        POLICY["Policy Engine"]
        HALL["Hallucination Detector"]
    end

    subgraph Analysis["Post-Profile Analysis"]
        ALIGN["AlignmentImprintDetector"]
        CONE["ConceptConeAnalyzer"]
        CROSS["CrossLayerAnalyzer"]
        OURO["OuroborosDetector"]
    end

    subgraph Longbow["Longbow Vector Engine"]
        ARROW["Apache Arrow Flight"]
        SLAB["SlabArena Off-Heap"]
        HNSW["HNSW Index"]
        SIMD["SIMD Distance Kernels"]
        COW["COW Publication"]
    end

    subgraph Dashboard["Web Dashboard"]
        HTTP["net/http Server"]
        TEMPL["Templ Templates"]
        HTMX["HTMX UI"]
        SSE["SSE Hub"]
    end

    subgraph Observability["Observability"]
        PROM["Prometheus /metrics"]
        OTEL["OTel OTLP/HTTP Export"]
    end

    subgraph Storage["Storage"]
        PROFILES["Model Profiles"]
        CHECKPOINTS["Job Checkpoints"]
        LEDGER["Activity Ledger"]
    end

    DL --> CONV
    CONV --> ONNX
    CONV --> WASM
    PROF --> ONNX
    PROF --> SCORE

    ONNX --> MON
    WASM --> MON
    MON --> ARROW
    ARROW --> SLAB
    SLAB --> HNSW
    HNSW --> SIMD
    SIMD --> SCORE
    COW --> SCORE
    SCORE --> STEER
    SCORE --> POLICY
    POLICY --> STEER
    SCORE --> HALL

    PROFILES --> ALIGN
    PROFILES --> CONE
    PROFILES --> CROSS
    STEER --> OURO

    MON --> SSE
    SCORE --> SSE
    SSE --> HTMX
    TEMPL --> HTTP
    HTMX --> HTTP

    PROFILES --> SCORE
    CHECKPOINTS --> PROF
    LEDGER --> POLICY

    SCORE --> PROM
    STEER --> PROM
    PROM --> OTEL
```

---

## Longbow Vector Engine Integration

BurnNotice leverages Longbow (github.com/23skdu/longbow) as its activation storage and processing backbone. Longbow provides:

### Zero-Copy Activation Transport

Activations flow from ONNX inference directly into Longbow's Apache Arrow Flight store without serialization overhead. Arrow RecordBatches enable batch processing of thousands of activation vectors simultaneously.

```mermaid
graph LR
    ONNX["ONNX Output"] --> BATCH["Arrow RecordBatch"]
    BATCH --> FLIGHT["Arrow Flight DoPut"]
    FLIGHT --> STORE["Longbow Vector Store"]
    STORE --> SCORER["Scoring Engine"]
```

### Off-Heap Memory Management

Longbow's SlabArena allocates activation matrices in 1MB contiguous slabs outside Go's heap, bypassing GC pressure from billion-scale float32 allocations. The SlabPool reclaims memory via `Munmap` to prevent virtual memory fragmentation.

```mermaid
graph TB
    subgraph SlabArena["SlabArena Memory"]
        ALLOC["1MB Slab Allocation"]
        NUMA["NUMA-Aware Placement"]
        RECLAIM["Munmap Reclamation"]
    end

    subgraph GoHeap["Go Heap"]
        GC["Garbage Collector"]
    end

    ALLOC -.->|Bypasses| GC
    NUMA --> ALLOC
    RECLAIM --> ALLOC
```

### SIMD-Optimized Distance Calculations

All activation scoring (cosine similarity, euclidean distance, dot product) uses Longbow's hand-optimized SIMD kernels instead of gonum:

| Architecture | Optimization | Throughput Gain |
| --- | --- | --- |
| AVX2 (x86) | 256-bit vectorized dot product | 3-4x |
| AVX-512 (x86) | 512-bit vectorized + softmax kernels | 5-8x |
| NEON (ARM64) | 128-bit vectorized for Apple Silicon/Graviton | 3-4x |
| TurboQuant | FWHT-optimized bit-packing | 8x |

### HNSW Index for Centroid Lookup

The HNSW index enables fast nearest-neighbor lookups on activation centroids, used for:

- Conflict subspace detection (find nearest conflict centroid)
- Policy violation detection (find nearest policy-violating cluster)
- Truth direction validation (verify activation proximity to truth subspace)

### COW Publication for Lock-Free Scoring

Longbow's Copy-On-Write publication model enables lock-free concurrent scoring. When new activations are ingested, a new version of the graph data is created and published via atomic pointer swap. Scoring goroutines read the stable version without acquiring locks.

```mermaid
graph TB
    subgraph Ingest["Ingest Worker"]
        CLONE["Clone Current Version"]
        APPEND["Append Activations"]
        CAS["Atomic Pointer Swap"]
    end

    subgraph Scoring["Scoring Goroutines"]
        R1["Reader 1 → Stable V1"]
        R2["Reader 2 → Stable V1"]
        R3["Reader 3 → New V2"]
    end

    CLONE --> APPEND
    APPEND --> CAS
    CAS -.->|Swaps| R3
```

### Distributed Activation Sharing

For multi-node deployments, Longbow's gossip-based mesh enables activation sharing across cluster nodes. Consistent hashing partitions activation data, and load-aware routing steers scoring requests to the least-loaded node.

---

## Component Architecture

### ONNX Inference Engine

Loads and executes models in ONNX format. Forward hooks capture intermediate activations at specified layers for monitoring and profiling.

```mermaid
graph LR
    subgraph ONNX["ONNX Inference Engine"]
        LOAD["Model Loader"]
        HOOKS["Forward Hooks"]
        FWD["Forward Pass"]
        ACTS["Activation Capture"]
    end

    LOAD --> HOOKS
    HOOKS --> FWD
    FWD --> ACTS
    ACTS --> MON["Activation Monitor"]
```

### WASM Execution Backend

For environments without GPU access or where ONNX Runtime is unavailable, BurnNotice supports WASM-based inference using `wazero`. ONNX models are compiled to WASM via `webnn` or `onnx2wasm`, enabling pure Go execution.

```mermaid
graph LR
    MODEL["ONNX Model"] --> COMPILER["onnx2wasm Compiler"]
    COMPILER --> WASM_FILE["WASM Binary"]
    WASM_FILE --> RUNTIME["wazero Runtime"]
    RUNTIME --> EXEC["Forward Execution"]
    EXEC --> OUTPUT["Activation Output"]
```

### Activation Scoring Pipeline

Receives activation streams from the inference engine, computes scoring metrics against registered direction profiles, and triggers steering interventions when thresholds are exceeded.

```mermaid
graph TB
    subgraph Scoring["Scoring Pipeline"]
        INPUT["Activation Vector"]
        NORM["Normalize"]
        PROJ["Project onto Directions"]
        AGG["Aggregate Scores"]
        THRESH["Threshold Check"]
    end

    INPUT --> NORM
    NORM --> PROJ
    PROJ --> AGG
    AGG --> THRESH

    THRESH -->|Below threshold| PASS["Pass Through"]
    THRESH -->|Above threshold| ALERT["Alert + Steer"]
    THRESH -->|Critical| KILL["Kill Switch"]
```

### Steering Intervention Flow

When a score exceeds its configured threshold, one of four steering interventions is applied:

```mermaid
graph TB
    subgraph Steering["Steering Methods"]
        ACTADD["Activation Addition<br/>h' = h + α·v"]
        ANGULAR["Angular Steering<br/>Norm-preserving rotation"]
        SELECTIVE["Selective Steering<br/>Discriminative layers only"]
        ABLATION["Ablation<br/>h' = h - (h·v̂)v̂"]
    end

    TRIGGER["Score Threshold Exceeded"] --> DECIDE{Method?}
    DECIDE -->|Default| ACTADD
    DECIDE -->|Large model| ANGULAR
    DECIDE -->|Production| SELECTIVE
    DECIDE -->|Last resort| ABLATION

    ACTADD --> APPLY["Apply to Activation"]
    ANGULAR --> APPLY
    SELECTIVE --> APPLY
    ABLATION --> APPLY
```

### Web Dashboard Architecture

The monitoring dashboard uses Go's `net/http` server, Templ for type-safe HTML templates, and HTMX for dynamic updates without a JavaScript framework. Real-time data flows via Server-Sent Events (SSE).

```mermaid
graph TB
    subgraph Browser["Browser"]
        HTMX_VIEW["HTMX View"]
        CHARTS["Charts (uPlot)"]
    end

    subgraph Server["Go Server"]
        HTTP["net/http"]
        ROUTER["Mux Router"]
        HANDLERS["HTTP Handlers"]
        TEMPL["Templ Renderer"]
        SSE_HUB["SSE Hub"]
    end

    subgraph Data["Data Sources"]
        SCORES["Live Scores"]
        EVENTS["Steering Events"]
        JOBS["Profiler Jobs"]
    end

    Browser -->|HTTP Request| HTTP
    HTTP --> ROUTER
    ROUTER --> HANDLERS
    HANDLERS --> TEMPL
    TEMPL -->|HTML Response| Browser

    SCORES --> SSE_HUB
    EVENTS --> SSE_HUB
    JOBS --> SSE_HUB
    SSE_HUB -->|SSE Stream| HTMX_VIEW
    HTMX_VIEW -->|Partial Swap| CHARTS
```

### Model Profiling Pipeline

Each model requires a profile that identifies its unique activation-space geometry. The profiling pipeline extracts behavioral directions, subspaces, and truth neurons. After profiling, the `internal/analysis` package provides higher-order geometric inspection.

```mermaid
graph TB
    subgraph Profiling["Model Profiling Pipeline"]
        LOAD["Load ONNX Model"]
        HOOK["Install Forward Hooks"]
        BATCH["Run Contrastive Batches"]
        COLLECT["Collect Activations"]
        EXTRACT["Extract Directions<br/>(DIM, PCA, SVD, INLP)"]
        VALIDATE["Validate Directions"]
        PERSIST["Persist Profile"]
    end

    subgraph PostAnalysis["Post-Profile Analysis"]
        ALIGN["AlignmentImprintDetector<br/>(DPO / RLHF / CAI / SFT)"]
        CONE["ConceptConeAnalyzer<br/>(harm subspace geometry)"]
        CROSS["CrossLayerAnalyzer<br/>(decision layer + CKA)"]
    end

    LOAD --> HOOK
    HOOK --> BATCH
    BATCH --> COLLECT
    COLLECT --> EXTRACT
    EXTRACT --> VALIDATE
    VALIDATE --> PERSIST

    PERSIST --> PROFILE["Model Profile<br/>(Protobuf/gob)"]
    PROFILE --> ALIGN
    PROFILE --> CONE
    PROFILE --> CROSS
```

### Long-Running Architecture

BurnNotice is designed for continuous operation profiling and monitoring LLM deployments with memory pooling, layer subsampling, and checkpointing.

```mermaid
graph TB
    subgraph LongRunning["Long-Running Architecture"]
        subgraph Memory["Memory Management"]
            POOL["sync.Pool<br/>Activation Buffers"]
            STREAM["Disk Streaming<br/>Large Activations"]
        end

        subgraph Scheduling["Job Scheduling"]
            QUEUE["Job Queue"]
            GPU["GPU Memory-Aware<br/>Scheduler"]
            PROGRESSIVE["Progressive<br/>Layer Profiling"]
        end

        subgraph Resilience["Resilience"]
            CHECKPOINT["Checkpoint Manager"]
            VERSION["Profile Versioning"]
            RECOVERY["Job Recovery"]
        end
    end

    QUEUE --> GPU
    GPU --> PROGRESSIVE
    PROGRESSIVE --> POOL
    POOL --> STREAM
    STREAM --> CHECKPOINT
    CHECKPOINT --> VERSION
    VERSION --> RECOVERY
```

---

## Data Flow

### Inference-Time Monitoring

```mermaid
sequenceDiagram
    participant Agent as LLM Agent
    participant Engine as ONNX/WASM Engine
    participant Monitor as Activation Monitor
    participant Scorer as Scoring Engine
    participant Policy as Policy Engine
    participant Steer as Steering Engine
    participant Dashboard as Web Dashboard

    Agent->>Engine: Generate response
    Engine->>Monitor: Stream activations (per token, per layer)
    Monitor->>Scorer: Score against direction profiles
    Scorer->>Policy: Evaluate thresholds
    alt Score exceeds threshold
        Policy->>Steer: Trigger intervention
        Steer->>Engine: Apply steering vector
        Engine->>Agent: Steered activation
    end
    Scorer->>Dashboard: SSE score update
    Policy->>Dashboard: SSE alert event
```

### Model Profiling

```mermaid
sequenceDiagram
    participant CLI as CLI Tool
    participant Downloader as HF Downloader
    participant Converter as ONNX Converter
    participant Engine as ONNX Engine
    participant Extractor as Direction Extractor
    participant Profile as Profile Store

    CLI->>Downloader: Download model from HuggingFace
    Downloader-->>CLI: Model files
    CLI->>Converter: Convert to ONNX format
    Converter-->>CLI: ONNX model
    CLI->>Engine: Load ONNX model + install hooks
    Engine->>Engine: Run contrastive prompt batches
    Engine-->>Extractor: Stream collected activations
    Extractor->>Extractor: Extract directions (DIM, PCA, SVD)
    Extractor->>Extractor: Validate on held-out data
    Extractor-->>Profile: Persist model profile
    Profile-->>CLI: Profile saved
```

---

## Technology Stack

| Component | Technology | Purpose |
| --- | --- | --- |
| **Primary Language** | Go | Concurrency, memory efficiency, single binary |
| **Model Inference** | ONNX Runtime | Portable model execution across CPU/GPU |
| **Fallback Inference** | WASM (wazero) | Pure Go execution, no native dependencies |
| **Vector Engine** | Longbow (Apache Arrow Flight) | Zero-copy activation storage, SIMD distance, HNSW |
| **Off-Heap Memory** | Longbow SlabArena | 1MB slab allocation, NUMA-aware, bypasses Go GC |
| **Distance Kernels** | Longbow SIMD (AVX2/AVX-512/NEON) | 3-8x faster cosine, euclidean, dot product |
| **Linear Algebra** | gonum.org/v1/gonum | Direction extraction (DIM, PCA, SVD, whitening) |
| **Analysis** | `internal/analysis` | Alignment imprint, concept cone, cross-layer CKA, ouroboros detection |
| **Hallucination Detection** | `internal/hallucination` | SAE-energy rolling-window spike detection |
| **CLI Framework** | cobra | HuggingFace download, ONNX conversion, profiling |
| **Web Server** | net/http | Built-in HTTP server |
| **Templates** | Templ | Type-safe HTML generation |
| **Dynamic UI** | HTMX | Partial page updates, SSE integration |
| **Real-time** | Server-Sent Events | Live score streaming to dashboard |
| **Serialization** | Protocol Buffers / gob | Profile storage, checkpointing |
| **Charting** | uPlot | Lightweight activation visualizations |
| **Metrics** | Prometheus + OTel OTLP/HTTP | 50+ instruments; push/pull dual-backend observability |
| **Scripts** | Python 3 (`scripts/`) | Research utilities (arxiv paper post-processing) |

---

## Deployment Topology

```mermaid
graph TB
    subgraph Dev["Development"]
        DEV_CLI["burnnotice CLI"]
        DEV_DASH["Local Dashboard :8080"]
        DEV_LONGBOW["Longbow Single Node"]
    end

    subgraph Prod["Production"]
        subgraph GPU_Cluster["GPU Cluster"]
            PROFILER["Long-Running Profiler"]
            ONNX_GPU["ONNX Runtime (GPU)"]
        end

        subgraph CPU_Cluster["CPU Cluster"]
            MONITOR["Activation Monitor"]
            WASM_CPU["WASM Inference"]
            DASHBOARD["Web Dashboard"]
        end

        subgraph Longbow_Cluster["Longbow Distributed Mesh"]
            NODE1["Longbow Node 1"]
            NODE2["Longbow Node 2"]
            NODE3["Longbow Node N"]
            GOSSIP["Gossip Protocol (SWIM)"]
            HASH_RING["Consistent Hash Ring"]
        end

        subgraph Storage["Shared Storage"]
            PROFILE_REG["Profile Registry"]
            LEDGER["Activity Ledger"]
            PARQUET["Parquet Snapshots"]
        end
    end

    DEV_CLI --> ONNX_GPU
    DEV_CLI --> DEV_LONGBOW
    PROFILER --> ONNX_GPU
    ONNX_GPU --> MONITOR
    WASM_CPU --> MONITOR
    MONITOR --> NODE1
    NODE1 <--> GOSSIP
    GOSSIP <--> NODE2
    GOSSIP <--> NODE3
    NODE1 <--> HASH_RING
    MONITOR --> DASHBOARD
    NODE1 --> PROFILE_REG
    NODE1 --> LEDGER
    NODE1 --> PARQUET
```

---

## Production-Ready Architectural Enhancements (P0)

To achieve microsecond-scale latency targets, high security, and model representation parity in enterprise production environments, BurnNotice introduces the following architectural additions:

### Asynchronous Non-Blocking Scoring Subsystem

To isolate core token generation latency from neural safety calculations, we introduce a highly concurrent, asynchronous handoff pipeline in the execution engine.

```mermaid
sequenceDiagram
    autonumber
    participant Engine as ONNX/WASM Inference
    participant RingBuf as Lock-Free Ring Buffer
    participant Workers as Async Worker Pool
    participant Scorer as Subspace Scorer
    participant Policy as Policy Engine
    participant Ledg as Activity Ledger
    participant Steer as Steering Engine

    Engine->{bracket_open}RingBuf: Enqueue Activation Tensor (Zero-Copy Pointer)
    Note over Engine: Resume Next Token Generation
    RingBuf->>Workers: Dequeue Activation (FIFO Handoff)
    Workers->>Scorer: Project Activation onto Subspace
    Scorer->>Policy: Evaluate Metric Projection Energy
    alt Policy Violation Detected
        Policy->>Steer: Dispatch Steering Vector (Dynamic Rotation)
        Policy->>Ledg: Record Alert in Ledger
        Note over Steer: Intervene on active layers
    end
```

- **Lock-Free Ring Buffer**: Replaces simple blocking channels to guarantee low-allocation, sub-microsecond queue/dequeue performance.
- **Emergency Synchronous Bypass**: A latch circuit monitors queue depth and processing latency. If latency exceeds a threshold, it automatically falls back to synchronous evaluation on high-risk prompt layers to prevent undetected deception leakages.

### Quantization & Direction Transfer Architecture

Quantized models (Turboquant, SQ8, PQ, INT8/FP8) scale down weight precision and alter latent activation geometries. To maintain alignment, a scaling transfer layer is introduced.

```mermaid
graph TD
    subgraph baseline["FP16 Baseline Space"]
        V_FP16["FP16 Steering Direction Vector V"]
    end
    subgraph quant["INT8/FP8 Quantized Space"]
        V_INT8["INT8 Steering Direction Vector V'"]
    end

    ScaleCalib["Mathematical Scale Calibration Layer"]
    ActScale["Activation Variance Matching"]
    ClipBound["Clip Boundary Alignment"]

    V_FP16 --> ScaleCalib
    ScaleCalib --> ActScale
    ScaleCalib --> ClipBound
    ActScale --> V_INT8
    ClipBound --> V_INT8
```

- **Scale Calibration Layer**: Standardizes different numerical representations by matching the standard deviation of activations between FP16 baselines and INT8/FP8 quantized paths.
- **Direction Mapping Engine**: Mathematically projects extracted baseline steering direction matrices onto quantized layers, validating that multi-dimensional geometric structures remain invariant under low-precision representation formats.

### Multi-Direction Subspace Extraction & INLP

A single direction is vulnerable to model evasion. We expand our extraction pipeline to construct multi-direction safety subspaces.

```mermaid
graph LR
    Dataset["Contrastive Dataset"] --> SVD["Top-K SVD Factorization"]
    Dataset --> INLP["Iterative Nullspace Projection"]
    SVD --> Subspace["Multi-Dimensional Safety Subspace Basis"]
    INLP --> Subspace
    Subspace --> Projection["Subspace Projection Scorer"]
```

- **Top-K SVD Subspace**: Extract the top-k right singular columns of the activation difference matrices using Gonum, building an orthonormal basis for safety governance.
- **Iterative Nullspace Projection (INLP)**: Progressively removes features matching known deception styles by projecting activations onto the nullspace of linear deception probes, revealing orthogonal safety directions.

### Temporal Sequence Scoring & Adaptive Calibration

- **Adaptive Calibration Subsystem**: Regularly updates safety thresholds based on held-out validation databases. Runs ROC/F1 analysis on background workers to select dynamic, model-specific threshold metrics.
- **Temporal Sequence Aggregator**: Computes sequence-level exponential moving averages (EMA) of token-level similarity scores. Tracks long-term drift trajectories across multi-agent chat logs to detect sustained deception or jailbreak attempts.

### Oversight-Ablated Profiling & Defense Against Alignment Faking

To thwart alignment faking, where a highly capable model conditionally suppresses its deceptive behavior when it detects a monitoring environment (detected via its latent *Oversight Sensitivity Direction*), BurnNotice integrates an **Oversight Ablator** directly into the profiling pipeline.

```mermaid
graph TD
    subgraph Phase1["Phase 1: Oversight Extraction"]
        C_Mon["Monitored Prompts"] --> DIM_O["Difference-in-Means"]
        C_Dep["Deployed Prompts"] --> DIM_O
        DIM_O --> V_Over["Oversight Sensitivity Vector v_oversight"]
    end

    subgraph Phase2["Phase 2: Ablated Deception Profiling"]
        Act_Raw["Raw Deception Prompt Activations"] --> Abl["Oversight Ablator<br/>h' = h - (h · v_oversight)v_oversight"]
        Abl --> Act_Abl["Oversight-Ablated Activations"]
        Act_Abl --> Ext_Dec["Deception DIM/PCA Extractor"]
        Ext_Dec --> Profile["True Deception Direction Profile"]
    end
```

- **Oversight Sensitivity Direction**: Latent subspace ($\hat{v}_{oversight}$) that encodes the model's cognitive awareness of safety testing, evaluation, and logging environments.
- **Oversight Ablation Hook**: Intercepts the activations $h$ during the deception extraction run and projects out the oversight sensitivity vector:
  $$h_{ablated} = h - (h \cdot \hat{v}_{oversight})\hat{v}_{oversight}$$
- **Shadow Profiling**: Implemented within the `AsyncLedger` continuously sampling unmonitored runtime activations and comparing them dynamically (via Cosine Distance) to the static profiles, triggering alerts on representational drift.

