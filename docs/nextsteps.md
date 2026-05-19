# BurnNotice: Technical Implementation Roadmap

> **Status**: Draft
> **Date**: 2026-05-18
> **Version**: 0.2

---

## Phase 1: Core Infrastructure (Weeks 1-4)

**Goals**: Go module setup, Longbow integration, ONNX model loading, CLI tool, direction extraction

### 1.1 Project Initialization

- [ ] **1.1.1** Initialize Go module (`go mod init burnnotice`) with Go 1.25+ toolchain
- [ ] **1.1.2** Create directory structure: `cmd/burnnotice/`, `cmd/dashboard/`, `internal/engine/`, `internal/scoring/`, `internal/steer/`, `internal/profile/`, `internal/datasets/`, `internal/web/`, `internal/longbow/`, `templates/`, `static/`, `proto/`, `profiles/`
- [ ] **1.1.3** Add `.gitignore` entries for Go binaries, build artifacts, and profile data
- [ ] **1.1.4** Create root `Makefile` with targets: `build`, `test`, `lint`, `templ`, `proto`, `clean`
- [ ] **1.1.5** Add `go.mod` dependencies: `cobra`, `gonum`, `longbow` (replace directive), `owulveryck/onnx-go`, `tetratelabs/wazero`, `a-h/templ`

### 1.2 Longbow Integration

- [ ] **1.2.1** Add Longbow as a Go module dependency (`go get github.com/23skdu/longbow`) or fork as a replace directive
- [ ] **1.2.2** Create `internal/longbow/client.go` — wrapper around Longbow's Go SDK for Arrow Flight DoPut/DoGet operations
- [ ] **1.2.3** Implement activation ingestion function: convert `[]float32` activation vectors into Arrow RecordBatches and stream to Longbow via `DoPut`
- [ ] **1.2.4** Implement activation retrieval function: query stored activations via `DoGet` and deserialize into Go slices
- [ ] **1.2.5** Create `internal/longbow/store.go` — initialization of Longbow vector store with configured dimension size (model hidden_dim), distance metric (cosine), and HNSW parameters (M, ef_construction)
- [ ] **1.2.6** Implement SlabArena-backed activation buffer pool: allocate activation buffers from Longbow's off-heap SlabPool instead of Go heap
- [ ] **1.2.7** Write unit tests: verify Arrow RecordBatch round-trip (ingest → retrieve → compare), verify SlabArena allocation returns correct buffer sizes
- [ ] **1.2.8** Write integration test: start local Longbow node, ingest 1000 activation vectors, retrieve and verify count/dimensions

### 1.3 ONNX Model Loader

- [ ] **1.3.1** Create `internal/engine/onnx.go` — struct `ONNXEngine` with `Load(path string)` method using `owulveryck/onnx-go`
- [ ] **1.3.2** Implement `Forward(input []float32) (output []float32, err error)` method for basic inference
- [ ] **1.3.3** Implement `GetIntermediate(layerName string) ([]float32, error)` to extract intermediate tensor values (simulating forward hooks)
- [ ] **1.3.4** Create `internal/engine/config.go` — model configuration struct (hidden_dim, num_layers, layer_names, attention_head_count)
- [ ] **1.3.5** Implement layer name mapping: parse ONNX model graph to identify residual stream tensor names (`resid_pre`, `resid_mid`, `resid_post` equivalents)
- [ ] **1.3.6** Write unit tests with a small ONNX model (e.g., TinyLlama-ONNX): verify forward pass output shape matches expected hidden_dim

### 1.4 WASM Inference Backend

- [ ] **1.4.1** Create `internal/wasm/backend.go` — struct `WASMBackend` with `Load(wasmPath string)` using `tetratelabs/wazero`
- [ ] **1.4.2** Implement WASI snapshot preview1 instantiation for file system and memory access
- [ ] **1.4.3** Implement `Forward(input []float32) ([]float32, error)` — write input to WASM memory, execute forward function, read output from WASM memory
- [ ] **1.4.4** Implement `GetIntermediate(layerIdx int) ([]float32, error)` — read intermediate activation buffers from WASM memory at known offsets
- [ ] **1.4.5** Create fallback logic: if ONNX Runtime unavailable, automatically use WASM backend
- [ ] **1.4.6** Write unit tests: verify WASM forward output matches ONNX forward output on same input (within float32 epsilon)

### 1.5 CLI Tool — HuggingFace Downloader

- [ ] **1.5.1** Create `cmd/burnnotice/main.go` with cobra root command
- [ ] **1.5.2** Create `internal/downloader/huggingface.go` — struct `HuggingFaceDownloader` with API token support
- [ ] **1.5.3** Implement `listModelFiles(modelID string) ([]string, error)` — call HuggingFace API `/api/models/{id}/tree/main` to list files
- [ ] **1.5.4** Implement `downloadFile(modelID, filePath, outputDir string) error` — download individual file with resume support (Range header)
- [ ] **1.5.5** Implement `Download(modelID, outputDir string) error` — orchestrate parallel file downloads with goroutine pool
- [ ] **1.5.6** Add `download` subcommand to CLI with flags: `--output`, `--token`, `--include` (file pattern filter)
- [ ] **1.5.7** Implement file filtering: only download `.safetensors`, `.bin`, `.json`, `.model` files by default
- [ ] **1.5.8** Write unit tests: mock HTTP responses for file listing, verify correct files selected for download

### 1.6 CLI Tool — ONNX Converter

- [ ] **1.6.1** Create `internal/converter/onnx.go` — struct `ONNXConverter` with `Convert(modelDir, output string, opts Options) error`
- [ ] **1.6.2** Implement safetensors loading: parse `.safetensors` format (JSON header + binary tensor data) into Go tensors
- [ ] **1.6.3** Implement ONNX graph construction: build ONNX protobuf model from loaded weights (use `onnx-go` model builder)
- [ ] **1.6.4** Implement dynamic axis configuration: set batch and sequence dimensions as dynamic for variable-length inference
- [ ] **1.6.5** Add `convert` subcommand to CLI with flags: `--output`, `--opset`, `--dynamic`
- [ ] **1.6.6** Write unit tests: convert a small safetensors model, verify ONNX file loads correctly in ONNX Runtime

### 1.7 CLI Tool — Model Profiler

- [ ] **1.7.1** Add `profile` subcommand to CLI with flags: `--output`, `--dataset`, `--layers`
- [ ] **1.7.2** Create `internal/profiler/cmd.go` — wire profiler to load ONNX model, run dataset, extract directions, save profile
- [ ] **1.7.3** Implement profile output: serialize to `profiles/<model-id>/profile.gob`

### 1.8 Contrastive Dataset Loader

- [ ] **1.8.1** Create `internal/datasets/loader.go` — struct `DatasetLoader` with `LoadJSON(path string) (*Dataset, error)` and `LoadJSONL(path string) (*Dataset, error)`
- [ ] **1.8.2** Define `Dataset` struct: `Pairs []ContrastivePair`, `PolicyRules []string`, `ModelID string`
- [ ] **1.8.3** Define `ContrastivePair` struct: `Truthful {Prompt, Response}`, `Deceptive {Prompt, Response}`
- [ ] **1.8.4** Implement dataset validation: verify all pairs have non-empty prompts/responses, check for duplicate pairs
- [ ] **1.8.5** Write unit tests: load sample JSON/JSONL datasets, verify correct parsing and validation errors

### 1.9 Direction Extraction (DIM + PCA)

- [ ] **1.9.1** Create `internal/extract/dim.go` — implement `ExtractDIM(positive, negative [][]float64) []float64` using gonum
- [ ] **1.9.2** Implement column mean computation: `columnMean(matrix [][]float64) []float64`
- [ ] **1.9.3** Implement vector normalization: `normalize(v []float64) []float64`
- [ ] **1.9.4** Create `internal/extract/pca.go` — implement `ExtractPCA(activations [][]float64, k int) (directions [][]float64, singularValues []float64)` using gonum SVD
- [ ] **1.9.5** Implement difference matrix construction: subtract negative mean from each positive activation
- [ ] **1.9.6** Implement thin SVD factorization via `gonum.org/v1/gonum/mat.SVD`
- [ ] **1.9.7** Extract top-k right singular vectors as direction vectors
- [ ] **1.9.8** Write unit tests: verify DIM on known vectors (e.g., [1,0] vs [0,1] → [1,-1] normalized), verify PCA recovers known subspace

### 1.10 Scoring Engine

- [ ] **1.10.1** Create `internal/scoring/scorer.go` — struct `Scorer` with `RegisterDirection(name string, vec []float64)` and `Score(activation []float64) map[string]float64`
- [ ] **1.10.2** Implement cosine similarity scoring: `dot(a, v) / (norm(a) * norm(v))` using Longbow's SIMD kernels where available
- [ ] **1.10.3** Implement sigmoid wrapper: `Sigmoid(x float64) float64`
- [ ] **1.10.4** Implement subspace scoring: `ScoreSubspace(activation []float64, basis [][]float64) float64` — sqrt(sum of squared projections)
- [ ] **1.10.5** Create `internal/scoring/aggregator.go` — aggregate per-token scores into per-response metrics (mean, max, AUC)
- [ ] **1.10.6** Write unit tests: verify cosine similarity on known vectors, verify sigmoid output range [0,1], verify subspace scoring orthogonality

### 1.11 Profile Persistence

- [ ] **1.11.1** Create `internal/profile/model.go` — define `ModelProfile` struct: `ModelID`, `Directions map[string]map[int][]float64`, `LayerScores`, `TruthNeurons`, `MonitoredLayers`
- [ ] **1.11.2** Implement `Save(path string) error` — gob serialization to `profiles/<model-id>/profile.gob`
- [ ] **1.11.3** Implement `Load(path string) (*ModelProfile, error)` — gob deserialization with error handling
- [ ] **1.11.4** Implement profile directory management: `MkdirAll` with proper permissions
- [ ] **1.11.5** Write unit tests: save → load round-trip, verify all fields match, test missing file error

---

## Phase 2: Steering and Intervention (Weeks 5-8)

**Goals**: Four steering methods, policy engine, kill switch, integration tests

### 2.1 Addition Steering (ActAdd)

- [ ] **2.1.1** Create `internal/steer/actadd.go` — implement `ActAdd(activation, direction []float64, alpha float64) []float64`
- [ ] **2.1.2** Implement scaled vector addition: `result[i] = activation[i] + alpha * direction[i]`
- [ ] **2.1.3** Write unit tests: verify output shape, verify alpha=0 returns original, verify alpha=1 returns sum

### 2.2 Angular Steering (Norm-Preserving)

- [ ] **2.2.1** Create `internal/steer/angular.go` — implement `Angular(activation, direction []float64, alpha float64) []float64`
- [ ] **2.2.2** Compute parallel component: `parallel = dot(a,v) / dot(v,v) * v`
- [ ] **2.2.3** Compute orthogonal component: `ortho = a - parallel`
- [ ] **2.2.4** Implement rotation: `result = cos(theta)*ortho + sin(theta)*parallel_norm*parallel_dir`
- [ ] **2.2.5** Renormalize to original activation norm
- [ ] **2.2.6** Write unit tests: verify output norm equals input norm (within epsilon), verify rotation direction

### 2.3 Selective Steering

- [ ] **2.3.1** Create `internal/steer/selective.go` — struct `SelectiveSteerer` with `truthDir`, `deceptionDir`, `targetLayers`
- [ ] **2.3.2** Implement `IsDiscriminative(layerIdx int, activation []float64) bool` — truth and deception projections have opposite signs
- [ ] **2.3.3** Implement `ComputeSteering(layerIdx int, activation []float64, strength float64) *SteeringCommand` — only steer if deception-leaning
- [ ] **2.3.4** Implement combined steering vector: `v = normalize(truth - deception)`
- [ ] **2.3.5** Implement clamped rotation angle: `theta = strength * Asin(-clamp(proj))`
- [ ] **2.3.6** Write unit tests: verify non-discriminative layers return nil, verify steering only applied when proj < 0

### 2.4 Ablation Steering

- [ ] **2.4.1** Create `internal/steer/ablation.go` — implement `Ablate(activation, direction []float64) []float64`
- [ ] **2.4.2** Implement orthogonal projection: `result = a - dot(a,v)/dot(v,v) * v`
- [ ] **2.4.3** Handle zero-norm direction edge case
- [ ] **2.4.4** Write unit tests: verify result is orthogonal to direction (dot product ≈ 0)

### 2.5 Steering Command Dispatcher

- [ ] **2.5.1** Create `internal/steer/dispatcher.go` — struct `Dispatcher` with `Apply(sessionID, layerIdx, tokenIdx int, activation []float64) []float64`
- [ ] **2.5.2** Implement method routing: switch on `SteeringMethod` enum to call correct steering function
- [ ] **2.5.3** Implement layer filtering: skip steering if layer not in target set
- [ ] **2.5.4** Write unit tests: verify correct method dispatched, verify layer filtering works

### 2.6 Policy Engine

- [ ] **2.6.1** Create `internal/policy/config.go` — define `Policy` struct: `Name`, `Metric`, `Threshold`, `Action`, `SteeringVector`, `SteeringStrength`, `Escalation`
- [ ] **2.6.2** Implement YAML config loading: `LoadPolicies(path string) ([]Policy, error)` using `gopkg.in/yaml.v3`
- [ ] **2.6.3** Create `internal/policy/engine.go` — struct `Engine` with `Evaluate(scores map[string]float64) []PolicyViolation`
- [ ] **2.6.4** Implement threshold comparison: for each policy, check if metric score exceeds threshold
- [ ] **2.6.5** Implement action resolution: map violation to action type (steer, block, log, terminate)
- [ ] **2.6.6** Implement escalation counter: track repeated violations and escalate after N occurrences
- [ ] **2.6.7** Write unit tests: verify policies trigger at correct thresholds, verify escalation after N violations

### 2.7 Kill Switch

- [ ] **2.7.1** Create `internal/policy/killswitch.go` — struct `KillSwitch` with `Armed bool`, `Trigger()`, `Reset()`
- [ ] **2.7.2** Implement context cancellation: `Trigger()` cancels the inference context, terminating generation
- [ ] **2.7.3** Implement notification: send alert to SSE hub and log with activation snapshot
- [ ] **2.7.4** Write unit tests: verify context cancellation propagates, verify reset re-arms

### 2.8 Profile Manager with Worker Pool

- [ ] **2.8.1** Create `internal/profile/manager.go` — struct `ProfileManager` with `jobs chan *ProfileJob`, `workers int`
- [ ] **2.8.2** Implement `Start()` — spawn N worker goroutines that consume from job channel
- [ ] **2.8.3** Implement `Submit(job *ProfileJob)` — enqueue profiling job
- [ ] **2.8.4** Implement job status tracking: `queued`, `running`, `complete`, `failed`, `checkpointed`
- [ ] **2.8.5** Implement progress reporting: `job.Progress = float64(completed) / float64(total)`
- [ ] **2.8.6** Write unit tests: submit 10 jobs, verify all complete, verify progress monotonic increase

### 2.9 Integration Tests — Steering

- [ ] **2.9.1** Create `tests/integration/steering_test.go` — load small ONNX model, run deceptive prompt, apply truth steering, verify output changes
- [ ] **2.9.2** Test ActAdd: verify alpha scaling affects output magnitude
- [ ] **2.9.3** Test Angular: verify output norm preserved
- [ ] **2.9.4** Test Selective: verify only discriminative layers modified
- [ ] **2.9.5** Test Ablation: verify output orthogonal to deception direction

---

## Phase 3: Web Dashboard (Weeks 9-12)

**Goals**: Templ + HTMX dashboard with real-time monitoring via SSE

### 3.1 HTTP Server Setup

- [ ] **3.1.1** Create `cmd/dashboard/main.go` — initialize HTTP server on `:8080` with `net/http`
- [ ] **3.1.2** Implement route registration: `/`, `/profiler`, `/model/{id}`, `/monitor/{session}`, `/sse`, `/static/`
- [ ] **3.1.3** Implement static file serving: `http.FileServer(http.Dir("static"))` at `/static/`
- [ ] **3.1.4** Implement middleware: logging, panic recovery, request ID generation
- [ ] **3.1.5** Write integration test: verify all routes return 200 (or appropriate status)

### 3.2 Templ Template Compilation

- [ ] **3.2.1** Install Templ CLI (`go install github.com/a-h/templ/cmd/templ@latest`)
- [ ] **3.2.2** Add `templ generate` to Makefile `templ` target
- [ ] **3.2.3** Create `templates/layout.templ` — base HTML template with DOCTYPE, head, body, HTMX script include, Tailwind CSS include
- [ ] **3.2.4** Create `templates/components.templ` — reusable components: score card, progress bar, alert badge, layer label
- [ ] **3.2.5** Configure `go generate` directive to run templ generation on build

### 3.3 SSE Hub

- [ ] **3.3.1** Create `internal/web/sse.go` — struct `SSEHub` with `clients map[chan string]bool`, mutex
- [ ] **3.3.2** Implement `AddClient() chan string` — register new client channel
- [ ] **3.3.3** Implement `RemoveClient(ch chan string)` — unregister and close channel
- [ ] **3.3.4** Implement `Broadcast(event, data string)` — send event to all connected clients, skip full buffers
- [ ] **3.3.5** Implement `ServeHTTP(w http.ResponseWriter, r *http.Request)` — SSE endpoint with proper headers (`text/event-stream`, `no-cache`, `keep-alive`)
- [ ] **3.3.6** Implement client cleanup: detect closed connections via `r.Context().Done()`
- [ ] **3.3.7** Write unit tests: verify broadcast reaches all clients, verify removed client stops receiving, verify full buffer doesn't block

### 3.4 Dashboard Home Page

- [ ] **3.4.1** Create `templates/dashboard.templ` — main dashboard view with summary cards
- [ ] **3.4.2** Implement summary metrics display: active models, total profiles, current alerts, system status
- [ ] **3.4.3** Implement HTMX partial swap: `hx-get="/api/summary" hx-swap="innerHTML"` for live updates
- [ ] **3.4.4** Create `internal/web/handlers.go` — `Dashboard()` handler rendering dashboard template
- [ ] **3.4.5** Write integration test: verify dashboard page renders with correct HTML structure

### 3.5 Profiler Status Page

- [ ] **3.5.1** Create `templates/profiler.templ` — model list with profiling status (queued, running, complete, failed)
- [ ] **3.5.2** Implement progress bars: `hx-ext="sse" sse-connect="/sse" sse-swap="job-progress"`
- [ ] **3.5.3** Implement job queue table: model ID, status, progress %, elapsed time, actions (cancel, re-profile)
- [ ] [ ] **3.5.4** Create `Profiler()` handler — fetch job statuses from ProfileManager, render template
- [ ] **3.5.5** Implement cancel button: `hx-post="/job/{id}/cancel"` → call `job.cancel()`
- [ ] **3.5.6** Write integration test: verify job status updates via SSE, verify cancel stops job

### 3.6 Model Profile Detail Page

- [ ] **3.6.1** Create `templates/model-profile.templ` — detailed view of a single model's extracted directions
- [ ] **3.6.2** Implement direction table: direction name, layer count, mean score, visualization link
- [ ] **3.6.3** Implement layer score chart: bar chart showing score per layer for selected direction
- [ ] **3.6.4** Implement subspace projection scatter plot: 2D plot of activations on truth vs deception axes (SVG or canvas)
- [ ] **3.6.5** Create `ModelProfile()` handler — load profile from disk, render template with direction data
- [ ] **3.6.6** Write integration test: verify profile loads and renders correctly

### 3.7 Live Monitoring Page

- [ ] **3.7.1** Create `templates/monitor.templ` — live monitoring view for an active agent run
- [ ] **3.7.2** Implement score card grid: 4 columns (deception, conflict, policy, truth) with per-layer scores
- [ ] **3.7.3** Implement SSE score updates: `hx-ext="sse" sse-connect="/sse" sse-swap="score-update"`
- [ ] **3.7.4** Implement score color coding: green (<0.4), yellow (0.4-0.7), red (>0.7)
- [ ] **3.7.5** Implement progress bars: `style="width: {score*100}%"` with color class
- [ ] **3.7.6** Create `Monitor()` handler — accept session ID, subscribe to SSE events, render initial state
- [ ] **3.7.7** Write integration test: verify scores update via SSE, verify color coding correct

### 3.8 Layer × Token Heatmap

- [ ] **3.8.1** Create `templates/heatmap.templ` — SVG-based heatmap with layers on Y-axis, tokens on X-axis
- [ ] **3.8.2** Implement cell rendering: each cell colored by score (green → yellow → red gradient)
- [ ] **3.8.3** Implement HTMX OOB swap: `hx-swap-oob="true"` for heatmap updates without full page reload
- [ ] **3.8.4** Create `Heatmap()` handler — aggregate scores into layer×token matrix, render SVG
- [ ] **3.8.5** Write integration test: verify heatmap renders correct dimensions, verify OOB swap updates cells

### 3.9 Subspace Projection Scatter Plot

- [ ] **3.9.1** Create `templates/subspace.templ` — 2D scatter plot with truth axis (X) and deception axis (Y)
- [ ] **3.9.2** Implement point rendering: each point = one token's activation projection, colored by ground truth
- [ ] **3.9.3** Implement decision boundary line: separating hyperplane between truth and deception regions
- [ ] **3.9.4** Implement HTMX partial update: new points added via OOB swap during live monitoring
- [ ] **3.9.5** Create `Subspace()` handler — project activations onto truth/deception directions, render plot
- [ ] **3.9.6** Write integration test: verify points positioned correctly, verify boundary line renders

### 3.10 Neural Circuit Graph

- [ ] **3.10.1** Create `templates/circuits.templ` — graph visualization of attention heads and MLP neurons
- [ ] **3.10.2** Implement node rendering: circles for attention heads, squares for MLP neurons
- [ ] **3.10.3** Implement edge rendering: lines showing information flow, thickness = activation strength
- [ ] **3.10.4** Implement highlighting: deception circuits in red, truth circuits in green
- [ ] **3.10.5** Create `Circuits()` handler — build graph from activation data, render SVG
- [ ] **3.10.6** Write integration test: verify graph renders with correct node/edge count

### 3.11 Policy Configuration Editor

- [ ] **3.11.1** Create `templates/policies.templ` — form-based policy editor
- [ ] **3.11.2** Implement policy list: table with name, metric, threshold, action, edit/delete buttons
- [ ] **3.11.3** Implement add/edit form: fields for name, metric dropdown, threshold slider, action dropdown
- [ ] **3.11.4** Implement HTMX form submission: `hx-post="/api/policies" hx-swap="outerHTML"`
- [ ] **3.11.5** Create `Policies()` handler — render current policies, handle form submission
- [ ] **3.11.6** Create `SavePolicy()` API handler — validate input, update YAML config, reload engine
- [ ] **3.11.7** Write integration test: verify policy CRUD operations, verify config file updated

### 3.12 Alert History and Intervention Log

- [ ] **3.12.1** Create `templates/alerts.templ` — table of historical alerts and interventions
- [ ] **3.12.2** Implement alert row: timestamp, metric, score, threshold, action taken, session ID
- [ ] **3.12.3** Implement filtering: by metric, by action, by time range
- [ ] **3.12.4** Implement pagination: 50 alerts per page with next/prev links
- [ ] **3.12.5** Create `Alerts()` handler — query alert history, render template
- [ ] **3.12.6** Write integration test: verify alerts render in correct order, verify filtering works

---

## Phase 4: Advanced Features (Weeks 13-16)

**Goals**: SAE integration, hallucination detection, policy violation detection, MoE support

### 4.1 SAE Feature Extraction

- [ ] **4.1.1** Create `internal/sae/loader.go` — struct `SAELoader` with `Load(path string) error` for loading SAE weights
- [ ] **4.1.2** Implement SAE encoder: `Encode(activation []float64) ([]float64, error)` — forward pass through SAE (encoder → ReLU → decoder)
- [ ] **4.1.3** Implement SAE feature attribution: identify top-K active features for a given activation
- [ ] **4.1.4** Integrate Gemma Scope SAEs: download pretrained SAE weights, map to model layers
- [ ] **4.1.5** Store SAE features in Longbow: ingest feature activation vectors alongside raw activations
- [ ] **4.1.6** Write unit tests: verify SAE encoding reconstructs activation within tolerance, verify top-K feature selection

### 4.2 Hallucination Energy Detection

- [ ] **4.2.1** Create `internal/hallucination/energy.go` — struct `HallucinationDetector` with `ComputeEnergy(saeFeatures []float64) float64`
- [ ] **4.2.2** Implement geometric energy metric: `energy = sum(feature_activations^2)` for hallucination-associated SAE features
- [ ] **4.2.3** Implement phase transition detection: flag sudden energy spikes as hallucination risk
- [ ] **4.2.4** Implement layer-specific thresholds: different layers have different baseline energy levels
- [ ] **4.2.5** Write unit tests: verify energy computation, verify spike detection on synthetic data

### 4.3 Policy Violation Detection (Whitening)

- [ ] **4.3.1** Create `internal/policydetect/whitening.go` — struct `PolicyViolationDetector` with `Train(compliant, violating [][]float64)`
- [ ] **4.3.2** Implement covariance matrix computation using gonum `stat.CovarianceMatrix`
- [ ] **4.3.3** Implement eigen-decomposition via gonum `mat.EigenSym`
- [ ] **4.3.4** Build whitening matrix: `V * diag(1/sqrt(λ + ε)) * V^T`
- [ ] **4.3.5** Implement `Detect(activation []float64) float64` — center, whiten, compute norm, return compliance score
- [ ] **4.3.6** Write unit tests: verify compliant activations score high, violating activations score low

### 4.4 Cross-Model Direction Transfer

- [ ] **4.4.1** Create `internal/extract/transfer.go` — struct `DirectionTransfer` with `TransferDirection(source, target *ModelProfile, direction string) ([]float64, error)`
- [ ] **4.4.2** Implement dimension alignment: project source direction into target model's activation space via CKA alignment
- [ ] **4.4.3** Implement validation: test transferred direction on target model's held-out dataset
- [ ] **4.4.4** Write unit tests: verify transfer between same-architecture models (Llama-3-8B → Llama-3-70B)

### 4.5 MoE Architecture Support

- [ ] **4.5.1** Create `internal/engine/moe.go` — struct `MoEEngine` wrapping base ONNX engine with per-expert activation extraction
- [ ] **4.5.2** Implement expert routing detection: identify which experts are active for each token
- [ ] **4.5.3** Implement per-expert direction extraction: extract separate deception/truth directions per expert
- [ ] **4.5.4** Implement per-expert scoring: score each expert's activation independently
- [ ] **4.5.5** Write unit tests: verify expert routing detection, verify per-expert scores differ

### 4.6 LoRA-Based Reversible Steering

- [ ] **4.6.1** Create `internal/steer/lora.go` — struct `LoRASteerer` with `ComputeLoRAWeights(direction []float64, rank int) (A, B [][]float64)`
- [ ] **4.6.2** Implement low-rank decomposition: factor steering direction into A (d×r) and B (r×d) matrices
- [ ] **4.6.3** Implement LoRA application: `W' = W + A*B` applied to model weights (offline)
- [ ] **4.6.4** Implement reversal: `W = W' - A*B` to restore original weights
- [ ] **4.6.5** Write unit tests: verify LoRA application matches direct steering output, verify reversal restores original

### 4.7 Performance Optimization

- [ ] **4.7.1** Implement Longbow SlabArena buffer pooling: replace `sync.Pool` with SlabArena for activation buffers
- [ ] **4.7.2** Implement Arrow RecordBatch batching: group activations into batches of 64 before ingesting to Longbow
- [ ] **4.7.3** Implement SIMD kernel selection: detect CPU features (AVX2, AVX-512, NEON) and select optimal Longbow kernel
- [ ] **4.7.4** Benchmark scoring pipeline: measure throughput (scores/sec) before and after optimization
- [ ] **4.7.5** Write benchmark tests: `go test -bench=. -benchmem` for scoring, steering, and ingestion

### 4.8 Profile Export/Import

- [ ] **4.8.1** Create `internal/profile/export.go` — `Export(profile *ModelProfile, format string) ([]byte, error)` supporting gob, JSON, protobuf
- [ ] **4.8.2** Create `internal/profile/import.go` — `Import(data []byte, format string) (*ModelProfile, error)`
- [ ] **4.8.3** Implement profile metadata: add `CreatedAt`, `ModelArchitecture`, `DatasetSize`, `ValidationScore` fields
- [ ] **4.8.4** Write unit tests: export → import round-trip for all formats, verify metadata preserved

---

## Phase 5: Agent Governance Integration (Weeks 17-20)

**Goals**: Agent SDK, compliance reporting, activity ledger, multi-agent governance

### 5.1 Go Agent SDK

- [ ] **5.1.1** Create `internal/agent/sdk.go` — struct `AgentSDK` with `NewBurnNoticeMiddleware(config Config) (Middleware, error)`
- [ ] **5.1.2** Implement request interceptor: capture prompt, run through ONNX model, score activations
- [ ] **5.1.3** Implement response interceptor: capture generated text, run through ONNX model, score activations
- [ ] **5.1.4** Implement steering hook: if score exceeds threshold, apply steering before returning response
- [ ] **5.1.5** Write unit tests: verify middleware intercepts requests/responses, verify steering applied on threshold breach

### 5.2 LangChain Integration

- [ ] **5.2.1** Create `internal/agent/langchain.go` — LangChain callback handler that forwards activations to BurnNotice
- [ ] **5.2.2** Implement `OnLLMStart`, `OnLLMEnd`, `OnChainStart`, `OnChainEnd` callbacks
- [ ] **5.2.3** Implement activation streaming: send intermediate activations to Longbow during generation
- [ ] **5.2.4** Write integration test: run LangChain chain with BurnNotice middleware, verify activations scored

### 5.3 CrewAI Integration

- [ ] **5.3.1** Create `internal/agent/crewai.go` — CrewAI task decorator that wraps agent execution with monitoring
- [ ] **5.3.2** Implement task-level monitoring: score activations at task boundaries
- [ ] **5.3.3** Implement agent-level monitoring: score activations during agent tool use
- [ ] **5.3.4** Write integration test: run CrewAI crew with BurnNotice monitoring, verify alerts on deception

### 5.4 MCP Agent Integration

- [ ] **5.4.1** Create `internal/agent/mcp.go` — MCP middleware that intercepts tool calls and responses
- [ ] **5.4.2** Implement tool call monitoring: score activations before tool execution
- [ ] **5.4.3** Implement tool response monitoring: score activations after tool response received
- [ ] **5.4.4** Write integration test: run MCP agent with tool calls, verify monitoring captures both directions

### 5.5 Activity Ledger

- [ ] **5.5.1** Create `internal/ledger/ledger.go` — struct `Ledger` with SQLite backend (`modernc.org/sqlite`)
- [ ] **5.5.2** Implement schema: `alerts(id, timestamp, session_id, metric, score, threshold, action)`, `interventions(id, timestamp, session_id, method, layer, result)`
- [ ] **5.5.3** Implement `RecordAlert(alert Alert) error` — insert alert row
- [ ] **5.5.4** Implement `RecordIntervention(intervention Intervention) error` — insert intervention row
- [ ] **5.5.5** Implement `QueryAlerts(filter AlertFilter) ([]Alert, error)` — filtered query with pagination
- [ ] **5.5.6** Write unit tests: verify CRUD operations, verify query filtering, verify pagination

### 5.6 Compliance Reporting

- [ ] **5.6.1** Create `internal/compliance/report.go` — struct `Reporter` with `GenerateReport(startTime, endTime time.Time) (*Report, error)`
- [ ] **5.6.2** Implement EU AI Act report: aggregate alerts by risk category, compute compliance metrics
- [ ] **5.6.3** Implement NIST AI RMF report: map alerts to RMF functions (Govern, Map, Measure, Manage)
- [ ] **5.6.4** Implement PDF export: generate formatted PDF report using `github.com/jung-kurt/gofpdf`
- [ ] **5.6.5** Write unit tests: verify report contains correct aggregate counts, verify PDF generates without error

### 5.7 Human-in-the-Loop Approval

- [ ] **5.7.1** Create `internal/compliance/approval.go` — struct `ApprovalWorkflow` with pending/approved/rejected queues
- [ ] **5.7.2** Implement `RequestApproval(sessionID, reason string) string` — create approval request, return request ID
- [ ] **5.7.3** Implement `Approve(requestID, reviewer string) error` — mark request approved, resume generation
- [ ] **5.7.4** Implement `Reject(requestID, reviewer, reason string) error` — mark request rejected, block response
- [ ] **5.7.5** Implement timeout: auto-reject after configurable timeout (default 5 minutes)
- [ ] **5.7.6** Write unit tests: verify approval/resume flow, verify rejection/block flow, verify timeout

### 5.8 Multi-Agent Governance

- [ ] **5.8.1** Create `internal/agent/multiagent.go` — struct `MultiAgentGovernor` tracking inter-agent communication
- [ ] **5.8.2** Implement message monitoring: score activations on messages between agents
- [ ] **5.8.3** Implement alignment drift propagation detection: flag when one agent's deception influences another
- [ ] **5.8.4** Implement governance policies: define rules for agent-to-agent communication (e.g., no deception propagation)
- [ ] **5.8.5** Write unit tests: verify inter-agent message scoring, verify drift propagation detection

---

## Phase 6: Long-Running Model Analysis System (Weeks 21-24)

**Goals**: Checkpointing, memory management, scheduling, profile versioning, distributed profiling

### 6.1 Checkpointing

- [ ] **6.1.1** Create `internal/checkpoint/manager.go` — struct `CheckpointManager` with `interval time.Duration`, `path string`
- [ ] **6.1.2** Implement `Start(state interface{})` — start ticker that saves state at configured interval
- [ ] **6.1.3** Implement `save(state interface{})` — gob encode to `.tmp` file, then atomic rename
- [ ] **6.1.4** Implement `Load(path string) (interface{}, error)` — decode checkpoint file, return state
- [ ] **6.1.5** Implement job resume: on startup, check for checkpoint files, restore job state and continue
- [ ] **6.1.6** Write unit tests: verify save/load round-trip, verify atomic rename on crash, verify resume continues from correct batch

### 6.2 Memory Management

- [ ] **6.2.1** Create `internal/memory/pool.go` — wrapper around Longbow's SlabPool for activation buffer allocation
- [ ] **6.2.2** Implement `Get(size int) []float64` — allocate from SlabArena, return buffer
- [ ] **6.2.3** Implement `Put(buf []float64)` — return buffer to SlabPool for reuse
- [ ] **6.2.4** Implement disk streaming: for activations exceeding memory threshold, stream to temporary Parquet files via Longbow's snapshot mechanism
- [ ] **6.2.5** Implement memory pressure monitoring: track SlabArena usage, trigger GC or disk spill at 80% capacity
- [ ] **6.2.6** Write unit tests: verify buffer reuse (same address returned after Put/Get), verify disk streaming for large activations

### 6.3 GPU-Aware Job Scheduler

- [ ] **6.3.1** Create `internal/scheduler/scheduler.go` — struct `Scheduler` with GPU memory tracking and job queue
- [ ] **6.3.2** Implement GPU memory detection: query available GPU memory via ONNX Runtime GPU provider
- [ ] **6.3.3** Implement job sizing: estimate memory requirement based on model size (parameters × 4 bytes × batch size)
- [ ] **6.3.4** Implement scheduling logic: run small models in parallel, large models exclusively
- [ ] **6.3.5** Implement job preemption: pause lower-priority job when higher-priority job needs GPU
- [ ] **6.3.6** Write unit tests: verify small jobs run in parallel, verify large job runs exclusively, verify preemption works

### 6.4 Progressive Profiling

- [ ] **6.4.1** Create `internal/profiler/progressive.go` — struct `ProgressiveProfiler` with layer group configuration
- [ ] **6.4.2** Implement layer grouping: split model layers into groups of 4-8 layers each
- [ ] **6.4.3** Implement group-by-group profiling: profile one group at a time, release memory before next group
- [ ] **6.4.4** Implement partial result merging: combine direction vectors from each group into unified profile
- [ ] **6.4.5** Implement early exit: stop profiling a group if direction signal is below threshold
- [ ] **6.4.6** Write unit tests: verify layer grouping correct, verify partial results merge correctly

### 6.5 Profile Versioning

- [ ] **6.5.1** Create `internal/profile/version.go` — struct `ProfileVersion` with `Version`, `CreatedAt`, `ModelID`, `DiffFrom string`
- [ ] **6.5.2** Implement version numbering: auto-increment version on each profile save
- [ ] **6.5.3** Implement profile diff: `Diff(old, new *ModelProfile) ProfileDiff` — compute cosine distance between direction vectors
- [ ] **6.5.4** Implement drift detection: flag significant direction changes (>0.3 cosine distance) as alignment drift
- [ ] **6.5.5** Implement version history: store all profile versions in `profiles/<model-id>/versions/`
- [ ] **6.5.6** Write unit tests: verify version numbering, verify diff computation, verify drift detection

### 6.6 Profile Validation

- [ ] **6.6.1** Create `internal/profile/validate.go` — struct `Validator` with `Validate(profile *ModelProfile, dataset *Dataset) ValidationResult`
- [ ] **6.6.2** Implement deception direction validation: run held-out deception prompts, verify direction predicts deception (AUROC > 0.9)
- [ ] **6.6.3** Implement truth direction validation: run held-out truthful prompts, verify direction predicts truth (AUROC > 0.9)
- [ ] **6.6.4** Implement conflict subspace validation: run adversarial prompts, verify activations project into conflict subspace
- [ ] **6.6.5** Generate validation report: AUROC, FPR, precision, recall for each direction
- [ ] **6.6.6** Write unit tests: verify validation on known-good profiles passes, verify validation on random directions fails

### 6.7 Distributed Profiling

- [ ] **6.7.1** Create `internal/profiler/distributed.go` — struct `DistributedProfiler` with Longbow cluster client
- [ ] **6.7.2** Implement batch splitting: divide contrastive batches across N Longbow nodes
- [ ] **6.7.3** Implement distributed activation collection: each node collects activations for its batch subset
- [ ] **6.7.4** Implement result aggregation: gather activations from all nodes via Longbow's Arrow Flight DoGet
- [ ] **6.7.5** Implement distributed direction extraction: run DIM/PCA on aggregated activations
- [ ] **6.7.6** Write integration test: start 3-node Longbow cluster, profile model with distributed batches, verify results match single-node profiling

### 6.8 Profile Registry

- [ ] **6.8.1** Create `internal/registry/registry.go` — struct `Registry` with index of all available profiles
- [ ] **6.8.2** Implement profile metadata: `ModelID`, `Architecture`, `Date`, `DirectionCount`, `ValidationScore`, `Version`
- [ ] **6.8.3** Implement `List() []ProfileMetadata` — return all profiles with metadata
- [ ] **6.8.4** Implement `Get(modelID string) (*ModelProfile, error)` — load profile by ID
- [ ] **6.8.5** Implement `Delete(modelID string) error` — remove profile and all versions
- [ ] **6.8.6** Implement registry dashboard page: table of profiles with search, filter, and sort
- [ ] **6.8.7** Write unit tests: verify CRUD operations, verify metadata correct, verify dashboard renders
