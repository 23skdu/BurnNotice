# BurnNotice: Technical Implementation Roadmap

> **Status**: Draft
> **Date**: 2026-05-18
> **Version**: 0.2

---

## Phase 1: Core Infrastructure (Weeks 1-4)

**Goals**: Go module setup, Longbow integration, ONNX model loading, CLI tool, direction extraction

### 1.1 Project Initialization

- [x] **1.1.1** Initialize Go module (`go mod init burnnotice`) with Go 1.25+ toolchain
- [x] **1.1.2** Create directory structure: `cmd/burnnotice/`, `cmd/dashboard/`, `internal/engine/`, `internal/scoring/`, `internal/steer/`, `internal/profile/`, `internal/datasets/`, `internal/web/`, `internal/longbow/`, `templates/`, `static/`, `proto/`, `profiles/`
- [x] **1.1.3** Add `.gitignore` entries for Go binaries, build artifacts, and profile data
- [x] **1.1.4** Create root `Makefile` with targets: `build`, `test`, `lint`, `templ`, `proto`, `clean`
- [x] **1.1.5** Add `go.mod` dependencies: `cobra`, `gonum`, `longbow` (replace directive), `owulveryck/onnx-go`, `tetratelabs/wazero`, `a-h/templ`

### 1.2 Longbow Integration

- [x] **1.2.1** Add Longbow as a Go module dependency (`go get github.com/23skdu/longbow`) or fork as a replace directive
- [x] **1.2.2** Create `internal/longbow/client.go` — wrapper around Longbow's Go SDK for Arrow Flight DoPut/DoGet operations
- [x] **1.2.3** Implement activation ingestion function: convert `[]float32` activation vectors into Arrow RecordBatches and stream to Longbow via `DoPut`
- [x] **1.2.4** Implement activation retrieval function: query stored activations via `DoGet` and deserialize into Go slices
- [x] **1.2.5** Create `internal/longbow/store.go` — initialization of Longbow vector store with configured dimension size (model hidden_dim), distance metric (cosine), and HNSW parameters (M, ef_construction)
- [x] **1.2.6** Implement SlabArena-backed activation buffer pool: allocate activation buffers from Longbow's off-heap SlabPool instead of Go heap
- [x] **1.2.7** Write unit tests: verify Arrow RecordBatch round-trip (ingest → retrieve → compare), verify SlabArena allocation returns correct buffer sizes
- [x] **1.2.8** Write integration test: start local Longbow node, ingest 1000 activation vectors, retrieve and verify count/dimensions

### 1.3 ONNX Model Loader

- [x] **1.3.1** Create `internal/engine/onnx.go` — struct `ONNXEngine` with `Load(path string)` method using `owulveryck/onnx-go`
- [x] **1.3.2** Implement `Forward(input []float32) (output []float32, err error)` method for basic inference
- [x] **1.3.3** Implement `GetIntermediate(layerName string) ([]float32, error)` to extract intermediate tensor values (simulating forward hooks)
- [x] **1.3.4** Create `internal/engine/config.go` — model configuration struct (hidden_dim, num_layers, layer_names, attention_head_count)
- [x] **1.3.5** Implement layer name mapping: parse ONNX model graph to identify residual stream tensor names (`resid_pre`, `resid_mid`, `resid_post` equivalents)
- [x] **1.3.6** Write unit tests with a small ONNX model (e.g., TinyLlama-ONNX): verify forward pass output shape matches expected hidden_dim

### 1.4 WASM Inference Backend

- [x] **1.4.1** Create `internal/wasm/backend.go` — struct `WASMBackend` with `Load(wasmPath string)` using `tetratelabs/wazero`
- [x] **1.4.2** Implement WASI snapshot preview1 instantiation for file system and memory access
- [x] **1.4.3** Implement `Forward(input []float32) ([]float32, error)` — write input to WASM memory, execute forward function, read output from WASM memory
- [x] **1.4.4** Implement `GetIntermediate(layerIdx int) ([]float32, error)` — read intermediate activation buffers from WASM memory at known offsets
- [x] **1.4.5** Create fallback logic: if ONNX Runtime unavailable, automatically use WASM backend
- [x] **1.4.6** Write unit tests: verify WASM forward output matches ONNX forward output on same input (within float32 epsilon)

### 1.5 CLI Tool — HuggingFace Downloader

- [x] **1.5.1** Create `cmd/burnnotice/main.go` with cobra root command
- [x] **1.5.2** Create `internal/downloader/huggingface.go` — struct `HuggingFaceDownloader` with API token support
- [x] **1.5.3** Implement `listModelFiles(modelID string) ([]string, error)` — call HuggingFace API `/api/models/{id}/tree/main` to list files
- [x] **1.5.4** Implement `downloadFile(modelID, filePath, outputDir string) error` — download individual file with resume support (Range header)
- [x] **1.5.5** Implement `Download(modelID, outputDir string) error` — orchestrate parallel file downloads with goroutine pool
- [x] **1.5.6** Add `download` subcommand to CLI with flags: `--output`, `--token`, `--include` (file pattern filter)
- [x] **1.5.7** Implement file filtering: only download `.safetensors`, `.bin`, `.json`, `.model` files by default
- [x] **1.5.8** Write unit tests: mock HTTP responses for file listing, verify correct files selected for download

### 1.6 CLI Tool — ONNX Converter

- [x] **1.6.1** Create `internal/converter/onnx.go` — struct `ONNXConverter` with `Convert(modelDir, output string, opts Options) error`
- [x] **1.6.2** Implement safetensors loading: parse `.safetensors` format (JSON header + binary tensor data) into Go tensors
- [x] **1.6.3** Implement ONNX graph construction: build ONNX protobuf model from loaded weights (use `onnx-go` model builder)
- [x] **1.6.4** Implement dynamic axis configuration: set batch and sequence dimensions as dynamic for variable-length inference
- [x] **1.6.5** Add `convert` subcommand to CLI with flags: `--output`, `--opset`, `--dynamic`
- [x] **1.6.6** Write unit tests: convert a small safetensors model, verify ONNX file loads correctly in ONNX Runtime

### 1.7 CLI Tool — Model Profiler

- [x] **1.7.1** Add `profile` subcommand to CLI with flags: `--output`, `--dataset`, `--layers`
- [x] **1.7.2** Create `internal/profiler/cmd.go` — wire profiler to load ONNX model, run dataset, extract directions, save profile
- [x] **1.7.3** Implement profile output: serialize to `profiles/<model-id>/profile.gob`

### 1.8 Contrastive Dataset Loader

- [x] **1.8.1** Create `internal/datasets/loader.go` — struct `DatasetLoader` with `LoadJSON(path string) (*Dataset, error)` and `LoadJSONL(path string) (*Dataset, error)`
- [x] **1.8.2** Define `Dataset` struct: `Pairs []ContrastivePair`, `PolicyRules []string`, `ModelID string`
- [x] **1.8.3** Define `ContrastivePair` struct: `Truthful {Prompt, Response}`, `Deceptive {Prompt, Response}`
- [x] **1.8.4** Implement dataset validation: verify all pairs have non-empty prompts/responses, check for duplicate pairs
- [x] **1.8.5** Write unit tests: load sample JSON/JSONL datasets, verify correct parsing and validation errors

### 1.9 Direction Extraction (DIM + PCA)

- [x] **1.9.1** Create `internal/extract/dim.go` — implement `ExtractDIM(positive, negative [][]float64) []float64` using gonum
- [x] **1.9.2** Implement column mean computation: `columnMean(matrix [][]float64) []float64`
- [x] **1.9.3** Implement vector normalization: `normalize(v []float64) []float64`
- [x] **1.9.4** Create `internal/extract/pca.go` — implement `ExtractPCA(activations [][]float64, k int) (directions [][]float64, singularValues []float64)` using gonum SVD
- [x] **1.9.5** Implement difference matrix construction: subtract negative mean from each positive activation
- [x] **1.9.6** Implement thin SVD factorization via `gonum.org/v1/gonum/mat.SVD`
- [x] **1.9.7** Extract top-k right singular vectors as direction vectors
- [x] **1.9.8** Write unit tests: verify DIM on known vectors (e.g., [1,0] vs [0,1] → [1,-1] normalized), verify PCA recovers known subspace

### 1.10 Scoring Engine

- [x] **1.10.1** Create `internal/scoring/scorer.go` — struct `Scorer` with `RegisterDirection(name string, vec []float64)` and `Score(activation []float64) map[string]float64`
- [x] **1.10.2** Implement cosine similarity scoring: `dot(a, v) / (norm(a) * norm(v))` using Longbow's SIMD kernels where available
- [x] **1.10.3** Implement sigmoid wrapper: `Sigmoid(x float64) float64`
- [x] **1.10.4** Implement subspace scoring: `ScoreSubspace(activation []float64, basis [][]float64) float64` — sqrt(sum of squared projections)
- [x] **1.10.5** Create `internal/scoring/aggregator.go` — aggregate per-token scores into per-response metrics (mean, max, AUC)
- [x] **1.10.6** Write unit tests: verify cosine similarity on known vectors, verify sigmoid output range [0,1], verify subspace scoring orthogonality

### 1.11 Profile Persistence

- [x] **1.11.1** Create `internal/profile/model.go` — define `ModelProfile` struct: `ModelID`, `Directions map[string]map[int][]float64`, `LayerScores`, `TruthNeurons`, `MonitoredLayers`
- [x] **1.11.2** Implement `Save(path string) error` — gob serialization to `profiles/<model-id>/profile.gob`
- [x] **1.11.3** Implement `Load(path string) (*ModelProfile, error)` — gob deserialization with error handling
- [x] **1.11.4** Implement profile directory management: `MkdirAll` with proper permissions
- [x] **1.11.5** Write unit tests: save → load round-trip, verify all fields match, test missing file error

---

## Phase 2: Steering and Intervention (Weeks 5-8)

**Goals**: Four steering methods, policy engine, kill switch, integration tests

### 2.1 Addition Steering (ActAdd)

- [x] **2.1.1** Create `internal/steer/actadd.go` — implement `ActAdd(activation, direction []float64, alpha float64) []float64`
- [x] **2.1.2** Implement scaled vector addition: `result[i] = activation[i] + alpha * direction[i]`
- [x] **2.1.3** Write unit tests: verify output shape, verify alpha=0 returns original, verify alpha=1 returns sum

### 2.2 Angular Steering (Norm-Preserving)

- [x] **2.2.1** Create `internal/steer/angular.go` — implement `Angular(activation, direction []float64, alpha float64) []float64`
- [x] **2.2.2** Compute parallel component: `parallel = dot(a,v) / dot(v,v) * v`
- [x] **2.2.3** Compute orthogonal component: `ortho = a - parallel`
- [x] **2.2.4** Implement rotation: `result = cos(theta)*ortho + sin(theta)*parallel_norm*parallel_dir`
- [x] **2.2.5** Renormalize to original activation norm
- [x] **2.2.6** Write unit tests: verify output norm equals input norm (within epsilon), verify rotation direction

### 2.3 Selective Steering

- [x] **2.3.1** Create `internal/steer/selective.go` — struct `SelectiveSteerer` with `truthDir`, `deceptionDir`, `targetLayers`
- [x] **2.3.2** Implement `IsDiscriminative(layerIdx int, activation []float64) bool` — truth and deception projections have opposite signs
- [x] **2.3.3** Implement `ComputeSteering(layerIdx int, activation []float64, strength float64) *SteeringCommand` — only steer if deception-leaning
- [x] **2.3.4** Implement combined steering vector: `v = normalize(truth - deception)`
- [x] **2.3.5** Implement clamped rotation angle: `theta = strength * Asin(-clamp(proj))`
- [x] **2.3.6** Write unit tests: verify non-discriminative layers return nil, verify steering only applied when proj < 0

### 2.4 Ablation Steering

- [x] **2.4.1** Create `internal/steer/ablation.go` — implement `Ablate(activation, direction []float64) []float64`
- [x] **2.4.2** Implement orthogonal projection: `result = a - dot(a,v)/dot(v,v) * v`
- [x] **2.4.3** Handle zero-norm direction edge case
- [x] **2.4.4** Write unit tests: verify result is orthogonal to direction (dot product ≈ 0)

### 2.5 Steering Command Dispatcher

- [x] **2.5.1** Create `internal/steer/dispatcher.go` — struct `Dispatcher` with `Apply(sessionID, layerIdx, tokenIdx int, activation []float64) []float64`
- [x] **2.5.2** Implement method routing: switch on `SteeringMethod` enum to call correct steering function
- [x] **2.5.3** Implement layer filtering: skip steering if layer not in target set
- [x] **2.5.4** Write unit tests: verify correct method dispatched, verify layer filtering works

### 2.6 Policy Engine

- [x] **2.6.1** Create `internal/policy/config.go` — define `Policy` struct: `Name`, `Metric`, `Threshold`, `Action`, `SteeringVector`, `SteeringStrength`, `Escalation`
- [x] **2.6.2** Implement YAML config loading: `LoadPolicies(path string) ([]Policy, error)` using `gopkg.in/yaml.v3`
- [x] **2.6.3** Create `internal/policy/engine.go` — struct `Engine` with `Evaluate(scores map[string]float64) []PolicyViolation`
- [x] **2.6.4** Implement threshold comparison: for each policy, check if metric score exceeds threshold
- [x] **2.6.5** Implement action resolution: map violation to action type (steer, block, log, terminate)
- [x] **2.6.6** Implement escalation counter: track repeated violations and escalate after N occurrences
- [x] **2.6.7** Write unit tests: verify policies trigger at correct thresholds, verify escalation after N violations

### 2.7 Kill Switch

- [x] **2.7.1** Create `internal/policy/killswitch.go` — struct `KillSwitch` with `Armed bool`, `Trigger()`, `Reset()`
- [x] **2.7.2** Implement context cancellation: `Trigger()` cancels the inference context, terminating generation
- [x] **2.7.3** Implement notification: send alert to SSE hub and log with activation snapshot
- [x] **2.7.4** Write unit tests: verify context cancellation propagates, verify reset re-arms

### 2.8 Profile Manager with Worker Pool

- [x] **2.8.1** Create `internal/profile/manager.go` — struct `ProfileManager` with `jobs chan *ProfileJob`, `workers int`
- [x] **2.8.2** Implement `Start()` — spawn N worker goroutines that consume from job channel
- [x] **2.8.3** Implement `Submit(job *ProfileJob)` — enqueue profiling job
- [x] **2.8.4** Implement job status tracking: `queued`, `running`, `complete`, `failed`, `checkpointed`
- [x] **2.8.5** Implement progress reporting: `job.Progress = float64(completed) / float64(total)`
- [x] **2.8.6** Write unit tests: submit 10 jobs, verify all complete, verify progress monotonic increase

### 2.9 Integration Tests — Steering

- [x] **2.9.1** Create `tests/integration/steering_test.go` — load small ONNX model, run deceptive prompt, apply truth steering, verify output changes
- [x] **2.9.2** Test ActAdd: verify alpha scaling affects output magnitude
- [x] **2.9.3** Test Angular: verify output norm preserved
- [x] **2.9.4** Test Selective: verify only discriminative layers modified
- [x] **2.9.5** Test Ablation: verify output orthogonal to deception direction

---

## Phase 3: Web Dashboard (Weeks 9-12)

**Goals**: Templ + HTMX dashboard with real-time monitoring via SSE

### 3.1 HTTP Server Setup

- [x] **3.1.1** Create `cmd/dashboard/main.go` — initialize HTTP server on `:8080` with `net/http`
- [x] **3.1.2** Implement route registration: `/`, `/profiler`, `/model/{id}`, `/monitor/{session}`, `/sse`, `/static/`
- [x] **3.1.3** Implement static file serving: `http.FileServer(http.Dir("static"))` at `/static/`
- [x] **3.1.4** Implement middleware: logging, panic recovery, request ID generation
- [x] **3.1.5** Write integration test: verify all routes return 200 (or appropriate status)

### 3.2 Templ Template Compilation

- [x] **3.2.1** Install Templ CLI (`go install github.com/a-h/templ/cmd/templ@latest`)
- [x] **3.2.2** Add `templ generate` to Makefile `templ` target
- [x] **3.2.3** Create `templates/layout.templ` — base HTML template with DOCTYPE, head, body, HTMX script include, Tailwind CSS include
- [x] **3.2.4** Create `templates/components.templ` — reusable components: score card, progress bar, alert badge, layer label
- [x] **3.2.5** Configure `go generate` directive to run templ generation on build

### 3.3 SSE Hub

- [x] **3.3.1** Create `internal/web/sse.go` — struct `SSEHub` with `clients map[chan string]bool`, mutex
- [x] **3.3.2** Implement `AddClient() chan string` — register new client channel
- [x] **3.3.3** Implement `RemoveClient(ch chan string)` — unregister and close channel
- [x] **3.3.4** Implement `Broadcast(event, data string)` — send event to all connected clients, skip full buffers
- [x] **3.3.5** Implement `ServeHTTP(w http.ResponseWriter, r *http.Request)` — SSE endpoint with proper headers (`text/event-stream`, `no-cache`, `keep-alive`)
- [x] **3.3.6** Implement client cleanup: detect closed connections via `r.Context().Done()`
- [x] **3.3.7** Write unit tests: verify broadcast reaches all clients, verify removed client stops receiving, verify full buffer doesn't block

### 3.4 Dashboard Home Page

- [x] **3.4.1** Create `templates/dashboard.templ` — main dashboard view with summary cards
- [x] **3.4.2** Implement summary metrics display: active models, total profiles, current alerts, system status
- [x] **3.4.3** Implement HTMX partial swap: `hx-get="/api/summary" hx-swap="innerHTML"` for live updates
- [x] **3.4.4** Create `internal/web/handlers.go` — `Dashboard()` handler rendering dashboard template
- [x] **3.4.5** Write integration test: verify dashboard page renders with correct HTML structure

### 3.5 Profiler Status Page

- [x] **3.5.1** Create `templates/profiler.templ` — model list with profiling status (queued, running, complete, failed)
- [x] **3.5.2** Implement progress bars: `hx-ext="sse" sse-connect="/sse" sse-swap="job-progress"`
- [x] **3.5.3** Implement job queue table: model ID, status, progress %, elapsed time, actions (cancel, re-profile)
- [x] **3.5.4** Create `Profiler()` handler — fetch job statuses from ProfileManager, render template
- [x] **3.5.5** Implement cancel button: `hx-post="/job/{id}/cancel"` → call `job.cancel()`
- [x] **3.5.6** Write integration test: verify job status updates via SSE, verify cancel stops job

### 3.6 Model Profile Detail Page

- [x] **3.6.1** Create `templates/model-profile.templ` — detailed view of a single model's extracted directions
- [x] **3.6.2** Implement direction table: direction name, layer count, mean score, visualization link
- [x] **3.6.3** Implement layer score chart: bar chart showing score per layer for selected direction
- [x] **3.6.4** Implement subspace projection scatter plot: 2D plot of activations on truth vs deception axes (SVG or canvas)
- [x] **3.6.5** Create `ModelProfile()` handler — load profile from disk, render template with direction data
- [x] **3.6.6** Write integration test: verify profile loads and renders correctly

### 3.7 Live Monitoring Page

- [x] **3.7.1** Create `templates/monitor.templ` — live monitoring view for an active agent run
- [x] **3.7.2** Implement score card grid: 4 columns (deception, conflict, policy, truth) with per-layer scores
- [x] **3.7.3** Implement SSE score updates: `hx-ext="sse" sse-connect="/sse" sse-swap="score-update"`
- [x] **3.7.4** Implement score color coding: green (<0.4), yellow (0.4-0.7), red (>0.7)
- [x] **3.7.5** Implement progress bars: `style="width: {score*100}%"` with color class
- [x] **3.7.6** Create `Monitor()` handler — accept session ID, subscribe to SSE events, render initial state
- [x] **3.7.7** Write integration test: verify scores update via SSE, verify color coding correct

### 3.8 Layer × Token Heatmap

- [x] **3.8.1** Create `templates/heatmap.templ` — SVG-based heatmap with layers on Y-axis, tokens on X-axis
- [x] **3.8.2** Implement cell rendering: each cell colored by score (green → yellow → red gradient)
- [x] **3.8.3** Implement HTMX OOB swap: `hx-swap-oob="true"` for heatmap updates without full page reload
- [x] **3.8.4** Create `Heatmap()` handler — aggregate scores into layer×token matrix, render SVG
- [x] **3.8.5** Write integration test: verify heatmap renders correct dimensions, verify OOB swap updates cells

### 3.9 Subspace Projection Scatter Plot

- [x] **3.9.1** Create `templates/subspace.templ` — 2D scatter plot with truth axis (X) and deception axis (Y)
- [x] **3.9.2** Implement point rendering: each point = one token's activation projection, colored by ground truth
- [x] **3.9.3** Implement decision boundary line: separating hyperplane between truth and deception regions
- [x] **3.9.4** Implement HTMX partial update: new points added via OOB swap during live monitoring
- [x] **3.9.5** Create `Subspace()` handler — project activations onto truth/deception directions, render plot
- [x] **3.9.6** Write integration test: verify points positioned correctly, verify boundary line renders

### 3.10 Neural Circuit Graph

- [x] **3.10.1** Create `templates/circuits.templ` — graph visualization of attention heads and MLP neurons
- [x] **3.10.2** Implement node rendering: circles for attention heads, squares for MLP neurons
- [x] **3.10.3** Implement edge rendering: lines showing information flow, thickness = activation strength
- [x] **3.10.4** Implement highlighting: deception circuits in red, truth circuits in green
- [x] **3.10.5** Create `Circuits()` handler — build graph from activation data, render SVG
- [x] **3.10.6** Write integration test: verify graph renders with correct node/edge count

### 3.11 Policy Configuration Editor

- [x] **3.11.1** Create `templates/policies.templ` — form-based policy editor
- [x] **3.11.2** Implement policy list: table with name, metric, threshold, action, edit/delete buttons
- [x] **3.11.3** Implement add/edit form: fields for name, metric dropdown, threshold slider, action dropdown
- [x] **3.11.4** Implement HTMX form submission: `hx-post="/api/policies" hx-swap="outerHTML"`
- [x] **3.11.5** Create `Policies()` handler — render current policies, handle form submission
- [x] **3.11.6** Create `SavePolicy()` API handler — validate input, update YAML config, reload engine
- [x] **3.11.7** Write integration test: verify policy CRUD operations, verify config file updated

### 3.12 Alert History and Intervention Log

- [x] **3.12.1** Create `templates/alerts.templ` — table of historical alerts and interventions
- [x] **3.12.2** Implement alert row: timestamp, metric, score, threshold, action taken, session ID
- [x] **3.12.3** Implement filtering: by metric, by action, by time range
- [x] **3.12.4** Implement pagination: 50 alerts per page with next/prev links
- [x] **3.12.5** Create `Alerts()` handler — query alert history, render template
- [x] **3.12.6** Write integration test: verify alerts render in correct order, verify filtering works

---

## Phase 4: Advanced Features (Weeks 13-16)

**Goals**: SAE integration, hallucination detection, policy violation detection, MoE support

### 4.1 SAE Feature Extraction

- [x] **4.1.1** Create `internal/sae/loader.go` — struct `SAELoader` with `Load(path string) error` for loading SAE weights
- [x] **4.1.2** Implement SAE encoder: `Encode(activation []float64) ([]float64, error)` — forward pass through SAE (encoder → ReLU → decoder)
- [x] **4.1.3** Implement SAE feature attribution: identify top-K active features for a given activation
- [x] **4.1.4** Integrate Gemma Scope SAEs: download pretrained SAE weights, map to model layers
- [x] **4.1.5** Store SAE features in Longbow: ingest feature activation vectors alongside raw activations
- [x] **4.1.6** Write unit tests: verify SAE encoding reconstructs activation within tolerance, verify top-K feature selection

### 4.2 Hallucination Energy Detection

- [x] **4.2.1** Create `internal/hallucination/energy.go` — struct `HallucinationDetector` with `ComputeEnergy(saeFeatures []float64) float64`
- [x] **4.2.2** Implement geometric energy metric: `energy = sum(feature_activations^2)` for hallucination-associated SAE features
- [x] **4.2.3** Implement phase transition detection: flag sudden energy spikes as hallucination risk
- [x] **4.2.4** Implement layer-specific thresholds: different layers have different baseline energy levels
- [x] **4.2.5** Write unit tests: verify energy computation, verify spike detection on synthetic data

### 4.3 Policy Violation Detection (Whitening)

- [x] **4.3.1** Create `internal/policydetect/whitening.go` — struct `PolicyViolationDetector` with `Train(compliant, violating [][]float64)`
- [x] **4.3.2** Implement covariance matrix computation using gonum `stat.CovarianceMatrix`
- [x] **4.3.3** Implement eigen-decomposition via gonum `mat.EigenSym`
- [x] **4.3.4** Build whitening matrix: `V * diag(1/sqrt(λ + ε)) * V^T`
- [x] **4.3.5** Implement `Detect(activation []float64) float64` — center, whiten, compute norm, return compliance score
- [x] **4.3.6** Write unit tests: verify compliant activations score high, violating activations score low

### 4.4 Cross-Model Direction Transfer

- [x] **4.4.1** Create `internal/extract/transfer.go` — struct `DirectionTransfer` with `TransferDirection(source, target *ModelProfile, direction string) ([]float64, error)`
- [x] **4.4.2** Implement dimension alignment: project source direction into target model's activation space via CKA alignment
- [x] **4.4.3** Implement validation: test transferred direction on target model's held-out dataset
- [x] **4.4.4** Write unit tests: verify transfer between same-architecture models (Llama-3-8B → Llama-3-70B)

### 4.5 MoE Architecture Support

- [x] **4.5.1** Create `internal/engine/moe.go` — struct `MoEEngine` wrapping base ONNX engine with per-expert activation extraction
- [x] **4.5.2** Implement expert routing detection: identify which experts are active for each token
- [x] **4.5.3** Implement per-expert direction extraction: extract separate deception/truth directions per expert
- [x] **4.5.4** Implement per-expert scoring: score each expert's activation independently
- [x] **4.5.5** Write unit tests: verify expert routing detection, verify per-expert scores differ

### 4.6 LoRA-Based Reversible Steering

- [x] **4.6.1** Create `internal/steer/lora.go` — struct `LoRASteerer` with `ComputeLoRAWeights(direction []float64, rank int) (A, B [][]float64)`
- [x] **4.6.2** Implement low-rank decomposition: factor steering direction into A (d×r) and B (r×d) matrices
- [x] **4.6.3** Implement LoRA application: `W' = W + A*B` applied to model weights (offline)
- [x] **4.6.4** Implement reversal: `W = W' - A*B` to restore original weights
- [x] **4.6.5** Write unit tests: verify LoRA application matches direct steering output, verify reversal restores original

### 4.7 Performance Optimization

- [x] **4.7.1** Implement Longbow SlabArena buffer pooling: replace `sync.Pool` with SlabArena for activation buffers
- [x] **4.7.2** Implement Arrow RecordBatch batching: group activations into batches of 64 before ingesting to Longbow
- [x] **4.7.3** Implement SIMD kernel selection: detect CPU features (AVX2, AVX-512, NEON) and select optimal Longbow kernel
- [x] **4.7.4** Benchmark scoring pipeline: measure throughput (scores/sec) before and after optimization
- [x] **4.7.5** Write benchmark tests: `go test -bench=. -benchmem` for scoring, steering, and ingestion

### 4.8 Profile Export/Import

- [x] **4.8.1** Create `internal/profile/export.go` — `Export(profile *ModelProfile, format string) ([]byte, error)` supporting gob, JSON, protobuf
- [x] **4.8.2** Create `internal/profile/import.go` — `Import(data []byte, format string) (*ModelProfile, error)`
- [x] **4.8.3** Implement profile metadata: add `CreatedAt`, `ModelArchitecture`, `DatasetSize`, `ValidationScore` fields
- [x] **4.8.4** Write unit tests: export → import round-trip for all formats, verify metadata preserved

---

## Phase 5: Agent Governance Integration (Weeks 17-20)

**Goals**: Agent SDK, compliance reporting, activity ledger, multi-agent governance

### 5.1 Go Agent SDK

- [x] **5.1.1** Create `internal/agent/sdk.go` — struct `AgentSDK` with `NewBurnNoticeMiddleware(config Config) (Middleware, error)`
- [x] **5.1.2** Implement request interceptor: capture prompt, run through ONNX model, score activations
- [x] **5.1.3** Implement response interceptor: capture generated text, run through ONNX model, score activations
- [x] **5.1.4** Implement steering hook: if score exceeds threshold, apply steering before returning response
- [x] **5.1.5** Write unit tests: verify middleware intercepts requests/responses, verify steering applied on threshold breach

### 5.2 LangChain Integration

- [x] **5.2.1** Create `internal/agent/langchain.go` — LangChain callback handler that forwards activations to BurnNotice
- [x] **5.2.2** Implement `OnLLMStart`, `OnLLMEnd`, `OnChainStart`, `OnChainEnd` callbacks
- [x] **5.2.3** Implement activation streaming: send intermediate activations to Longbow during generation
- [x] **5.2.4** Write integration test: run LangChain chain with BurnNotice middleware, verify activations scored

### 5.3 CrewAI Integration

- [x] **5.3.1** Create `internal/agent/crewai.go` — CrewAI task decorator that wraps agent execution with monitoring
- [x] **5.3.2** Implement task-level monitoring: score activations at task boundaries
- [x] **5.3.3** Implement agent-level monitoring: score activations during agent tool use
- [x] **5.3.4** Write integration test: run CrewAI crew with BurnNotice monitoring, verify alerts on deception

### 5.4 MCP Agent Integration

- [x] **5.4.1** Create `internal/agent/mcp.go` — MCP middleware that intercepts tool calls and responses
- [x] **5.4.2** Implement tool call monitoring: score activations before tool execution
- [x] **5.4.3** Implement tool response monitoring: score activations after tool response received
- [x] **5.4.4** Write integration test: run MCP agent with tool calls, verify monitoring captures both directions

### 5.5 Activity Ledger

- [x] **5.5.1** Create `internal/ledger/ledger.go` — struct `Ledger` with SQLite backend (`modernc.org/sqlite`)
- [x] **5.5.2** Implement schema: `alerts(id, timestamp, session_id, metric, score, threshold, action)`, `interventions(id, timestamp, session_id, method, layer, result)`
- [x] **5.5.3** Implement `RecordAlert(alert Alert) error` — insert alert row
- [x] **5.5.4** Implement `RecordIntervention(intervention Intervention) error` — insert intervention row
- [x] **5.5.5** Implement `QueryAlerts(filter AlertFilter) ([]Alert, error)` — filtered query with pagination
- [x] **5.5.6** Write unit tests: verify CRUD operations, verify query filtering, verify pagination

### 5.6 Compliance Reporting

- [x] **5.6.1** Create `internal/compliance/report.go` — struct `Reporter` with `GenerateReport(startTime, endTime time.Time) (*Report, error)`
- [x] **5.6.2** Implement EU AI Act report: aggregate alerts by risk category, compute compliance metrics
- [x] **5.6.3** Implement NIST AI RMF report: map alerts to RMF functions (Govern, Map, Measure, Manage)
- [x] **5.6.4** Implement PDF export: generate formatted PDF report using `github.com/jung-kurt/gofpdf`
- [x] **5.6.5** Write unit tests: verify report contains correct aggregate counts, verify PDF generates without error

### 5.7 Human-in-the-Loop Approval

- [x] **5.7.1** Create `internal/compliance/approval.go` — struct `ApprovalWorkflow` with pending/approved/rejected queues
- [x] **5.7.2** Implement `RequestApproval(sessionID, reason string) string` — create approval request, return request ID
- [x] **5.7.3** Implement `Approve(requestID, reviewer string) error` — mark request approved, resume generation
- [x] **5.7.4** Implement `Reject(requestID, reviewer, reason string) error` — mark request rejected, block response
- [x] **5.7.5** Implement timeout: auto-reject after configurable timeout (default 5 minutes)
- [x] **5.7.6** Write unit tests: verify approval/resume flow, verify rejection/block flow, verify timeout

### 5.8 Multi-Agent Governance

- [x] **5.8.1** Create `internal/agent/multiagent.go` — struct `MultiAgentGovernor` tracking inter-agent communication
- [x] **5.8.2** Implement message monitoring: score activations on messages between agents
- [x] **5.8.3** Implement alignment drift propagation detection: flag when one agent's deception influences another
- [x] **5.8.4** Implement governance policies: define rules for agent-to-agent communication (e.g., no deception propagation)
- [x] **5.8.5** Write unit tests: verify inter-agent message scoring, verify drift propagation detection

---

## Phase 6: Long-Running Model Analysis System (Weeks 21-24)

**Goals**: Checkpointing, memory management, scheduling, profile versioning, distributed profiling

### 6.1 Checkpointing

- [x] **6.1.1** Create `internal/checkpoint/manager.go` — struct `CheckpointManager` with `interval time.Duration`, `path string`
- [x] **6.1.2** Implement `Start(state interface{})` — start ticker that saves state at configured interval
- [x] **6.1.3** Implement `save(state interface{})` — gob encode to `.tmp` file, then atomic rename
- [x] **6.1.4** Implement `Load(path string) (interface{}, error)` — decode checkpoint file, return state
- [x] **6.1.5** Implement job resume: on startup, check for checkpoint files, restore job state and continue
- [x] **6.1.6** Write unit tests: verify save/load round-trip, verify atomic rename on crash, verify resume continues from correct batch

### 6.2 Memory Management

- [x] **6.2.1** Create `internal/memory/pool.go` — wrapper around Longbow's SlabPool for activation buffer allocation
- [x] **6.2.2** Implement `Get(size int) []float64` — allocate from SlabArena, return buffer
- [x] **6.2.3** Implement `Put(buf []float64)` — return buffer to SlabPool for reuse
- [x] **6.2.4** Implement disk streaming: for activations exceeding memory threshold, stream to temporary Parquet files via Longbow's snapshot mechanism
- [x] **6.2.5** Implement memory pressure monitoring: track SlabArena usage, trigger GC or disk spill at 80% capacity
- [x] **6.2.6** Write unit tests: verify buffer reuse (same address returned after Put/Get), verify disk streaming for large activations

### 6.3 GPU-Aware Job Scheduler

- [x] **6.3.1** Create `internal/scheduler/scheduler.go` — struct `Scheduler` with GPU memory tracking and job queue
- [x] **6.3.2** Implement GPU memory detection: query available GPU memory via ONNX Runtime GPU provider
- [x] **6.3.3** Implement job sizing: estimate memory requirement based on model size (parameters × 4 bytes × batch size)
- [x] **6.3.4** Implement scheduling logic: run small models in parallel, large models exclusively
- [x] **6.3.5** Implement job preemption: pause lower-priority job when higher-priority job needs GPU
- [x] **6.3.6** Write unit tests: verify small jobs run in parallel, verify large job runs exclusively, verify preemption works

### 6.4 Progressive Profiling

- [x] **6.4.1** Create `internal/profiler/progressive.go` — struct `ProgressiveProfiler` with layer group configuration
- [x] **6.4.2** Implement layer grouping: split model layers into groups of 4-8 layers each
- [x] **6.4.3** Implement group-by-group profiling: profile one group at a time, release memory before next group
- [x] **6.4.4** Implement partial result merging: combine direction vectors from each group into unified profile
- [x] **6.4.5** Implement early exit: stop profiling a group if direction signal is below threshold
- [x] **6.4.6** Write unit tests: verify layer grouping correct, verify partial results merge correctly

### 6.5 Profile Versioning

- [x] **6.5.1** Create `internal/profile/version.go` — struct `ProfileVersion` with `Version`, `CreatedAt`, `ModelID`, `DiffFrom string`
- [x] **6.5.2** Implement version numbering: auto-increment version on each profile save
- [x] **6.5.3** Implement profile diff: `Diff(old, new *ModelProfile) ProfileDiff` — compute cosine distance between direction vectors
- [x] **6.5.4** Implement drift detection: flag significant direction changes (>0.3 cosine distance) as alignment drift
- [x] **6.5.5** Implement version history: store all profile versions in `profiles/<model-id>/versions/`
- [x] **6.5.6** Write unit tests: verify version numbering, verify diff computation, verify drift detection

### 6.6 Profile Validation

- [x] **6.6.1** Create `internal/profile/validate.go` — struct `Validator` with `Validate(profile *ModelProfile, dataset *Dataset) ValidationResult`
- [x] **6.6.2** Implement deception direction validation: run held-out deception prompts, verify direction predicts deception (AUROC > 0.9)
- [x] **6.6.3** Implement truth direction validation: run held-out truthful prompts, verify direction predicts truth (AUROC > 0.9)
- [x] **6.6.4** Implement conflict subspace validation: run adversarial prompts, verify activations project into conflict subspace
- [x] **6.6.5** Generate validation report: AUROC, FPR, precision, recall for each direction
- [x] **6.6.6** Write unit tests: verify validation on known-good profiles passes, verify validation on random directions fails

### 6.7 Distributed Profiling

- [x] **6.7.1** Create `internal/profiler/distributed.go` — struct `DistributedProfiler` with Longbow cluster client
- [x] **6.7.2** Implement batch splitting: divide contrastive batches across N Longbow nodes
- [x] **6.7.3** Implement distributed activation collection: each node collects activations for its batch subset
- [x] **6.7.4** Implement result aggregation: gather activations from all nodes via Longbow's Arrow Flight DoGet
- [x] **6.7.5** Implement distributed direction extraction: run DIM/PCA on aggregated activations
- [x] **6.7.6** Write integration test: start 3-node Longbow cluster, profile model with distributed batches, verify results match single-node profiling

### 6.8 Profile Registry

- [x] **6.8.1** Create `internal/registry/registry.go` — struct `Registry` with index of all available profiles
- [x] **6.8.2** Implement profile metadata: `ModelID`, `Architecture`, `Date`, `DirectionCount`, `ValidationScore`, `Version`
- [x] **6.8.3** Implement `List() []ProfileMetadata` — return all profiles with metadata
- [x] **6.8.4** Implement `Get(modelID string) (*ModelProfile, error)` — load profile by ID
- [x] **6.8.5** Implement `Delete(modelID string) error` — remove profile and all versions
- [x] **6.8.6** Implement registry dashboard page: table of profiles with search, filter, and sort
- [x] **6.8.7** Write unit tests: verify CRUD operations, verify metadata correct, verify dashboard renders
