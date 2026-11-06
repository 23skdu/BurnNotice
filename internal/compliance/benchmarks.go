package compliance

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"runtime"
	"time"
)

// CapabilityBenchmark represents a single capability evaluation metric (e.g. MMLU, GSM8K).
type CapabilityBenchmark struct {
	Name     string  `json:"name"`
	Baseline float64 `json:"baseline"` // Original performance before steering
	Current  float64 `json:"current"`  // Performance after applying steering
}

// CapabilityEvaluator validates model capability degradation bounds (max 15%).
type CapabilityEvaluator struct {
	MaxDegradation float64 // e.g. 0.15 (15%)
}

// NewCapabilityEvaluator creates a new evaluator.
func NewCapabilityEvaluator() *CapabilityEvaluator {
	return &CapabilityEvaluator{
		MaxDegradation: 0.15,
	}
}

// Evaluate checks the benchmarks and returns average degradation and safety status.
func (ce *CapabilityEvaluator) Evaluate(benchmarks []CapabilityBenchmark) (float64, bool) {
	if len(benchmarks) == 0 {
		return 0.0, true
	}

	var totalDegradation float64
	for _, b := range benchmarks {
		if b.Baseline <= 0 {
			continue
		}
		// Compute degradation ratio
		deg := (b.Baseline - b.Current) / b.Baseline
		if deg < 0 {
			deg = 0.0 // performance improved or stayed same
		}
		totalDegradation += deg
	}

	avgDegradation := totalDegradation / float64(len(benchmarks))
	safe := avgDegradation <= ce.MaxDegradation

	return avgDegradation, safe
}

// SteeringController dynamically manages steering strength to respect capability bounds.
type SteeringController struct {
	BaseStrength   float64
	ActiveStrength float64
	Evaluator      *CapabilityEvaluator
}

// NewSteeringController creates a controller with the given base steering strength.
func NewSteeringController(base float64) *SteeringController {
	return &SteeringController{
		BaseStrength:   base,
		ActiveStrength: base,
		Evaluator:      NewCapabilityEvaluator(),
	}
}

// UpdateSteer assesses capability health and rolls back active strength if degradation exceeds safety bounds.
func (sc *SteeringController) UpdateSteer(benchmarks []CapabilityBenchmark) (float64, bool) {
	avgDeg, safe := sc.Evaluator.Evaluate(benchmarks)
	if !safe {
		// Roll back active steering strength proportionally to degradation
		factor := 1.0 - avgDeg
		if factor < 0 {
			factor = 0
		}
		sc.ActiveStrength = sc.BaseStrength * factor
		return sc.ActiveStrength, true // Alert triggered and rollback applied
	}

	// Keep base strength if safe
	sc.ActiveStrength = sc.BaseStrength
	return sc.ActiveStrength, false
}

// BenchmarkResult captures a reproducible benchmark measurement.
type BenchmarkResult struct {
	Name        string  `json:"name"`
	Iterations  int     `json:"iterations"`
	NsPerOp     float64 `json:"ns_per_op"`
	BytesPerOp  int64   `json:"bytes_per_op"`
	AllocsPerOp int64   `json:"allocs_per_op"`
	P50Latency  float64 `json:"p50_latency_ms,omitempty"`
	P99Latency  float64 `json:"p99_latency_ms,omitempty"`
	AUROC       float64 `json:"auroc,omitempty"`
}

// BenchmarkManifest captures reproducible environment metadata and benchmark results (Roadmap Item 8).
type BenchmarkManifest struct {
	OS                 string                `json:"os"`
	Arch               string                `json:"arch"`
	CPUCount           int                   `json:"cpu_count"`
	GoVersion          string                `json:"go_version"`
	ONNXRuntimeVersion string                `json:"onnxruntime_version"`
	ModelFamily        string                `json:"model_family"`
	DatasetHash        string                `json:"dataset_hash"`
	CommitSHA          string                `json:"commit_sha"`
	Timestamp          time.Time             `json:"timestamp"`
	BenchmarkResults   []BenchmarkResult     `json:"benchmark_results"`
	CapabilityResults  []CapabilityBenchmark `json:"capability_results,omitempty"`
}

// ReleaseGateStatus represents a quality/safety gate status for a release.
type ReleaseGateStatus struct {
	GateName string `json:"gate_name"`
	Passed   bool   `json:"passed"`
	Details  string `json:"details,omitempty"`
}

// ReleaseEvidenceBundle aggregates benchmark manifests, compliance reports, and gate statuses for release verification (Roadmap Item 20).
type ReleaseEvidenceBundle struct {
	Version          string              `json:"version"`
	CommitSHA        string              `json:"commit_sha"`
	GeneratedAt      time.Time           `json:"generated_at"`
	Manifest         BenchmarkManifest   `json:"manifest"`
	ComplianceReport *Report             `json:"compliance_report,omitempty"`
	GateStatus       []ReleaseGateStatus `json:"gate_status"`
	BundleDigest     string              `json:"bundle_digest"`
}

// NewBenchmarkManifest creates a BenchmarkManifest populated with current runtime environment metadata.
func NewBenchmarkManifest(onnxVersion, modelFamily, datasetHash, commitSHA string, results []BenchmarkResult) BenchmarkManifest {
	if onnxVersion == "" {
		onnxVersion = "1.24.2"
	}
	return BenchmarkManifest{
		OS:                 runtime.GOOS,
		Arch:               runtime.GOARCH,
		CPUCount:           runtime.NumCPU(),
		GoVersion:          runtime.Version(),
		ONNXRuntimeVersion: onnxVersion,
		ModelFamily:        modelFamily,
		DatasetHash:        datasetHash,
		CommitSHA:          commitSHA,
		Timestamp:          time.Now().UTC(),
		BenchmarkResults:   results,
	}
}

// Validate checks that a BenchmarkManifest contains required reproducible environment fields.
func (m *BenchmarkManifest) Validate() error {
	if m.OS == "" {
		return fmt.Errorf("manifest missing OS")
	}
	if m.Arch == "" {
		return fmt.Errorf("manifest missing Arch")
	}
	if m.GoVersion == "" {
		return fmt.Errorf("manifest missing GoVersion")
	}
	if m.ONNXRuntimeVersion == "" {
		return fmt.Errorf("manifest missing ONNXRuntimeVersion")
	}
	if m.ModelFamily == "" {
		return fmt.Errorf("manifest missing ModelFamily")
	}
	if m.DatasetHash == "" {
		return fmt.Errorf("manifest missing DatasetHash")
	}
	if m.CommitSHA == "" {
		return fmt.Errorf("manifest missing CommitSHA")
	}
	if m.Timestamp.IsZero() {
		return fmt.Errorf("manifest missing Timestamp")
	}
	for i, r := range m.BenchmarkResults {
		if r.Name == "" {
			return fmt.Errorf("benchmark result %d missing name", i)
		}
		if r.Iterations <= 0 {
			return fmt.Errorf("benchmark result %q has non-positive iterations: %d", r.Name, r.Iterations)
		}
		if r.NsPerOp < 0 {
			return fmt.Errorf("benchmark result %q has negative ns/op: %f", r.Name, r.NsPerOp)
		}
	}
	return nil
}

// SerializeJSON serializes the BenchmarkManifest to formatted JSON.
func (m *BenchmarkManifest) SerializeJSON() ([]byte, error) {
	if err := m.Validate(); err != nil {
		return nil, err
	}
	return json.MarshalIndent(m, "", "  ")
}

// ParseBenchmarkManifest deserializes and validates a BenchmarkManifest from JSON.
func ParseBenchmarkManifest(data []byte) (*BenchmarkManifest, error) {
	var m BenchmarkManifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("unmarshal benchmark manifest: %w", err)
	}
	if err := m.Validate(); err != nil {
		return nil, err
	}
	return &m, nil
}

// ComputeDatasetHash computes a deterministic SHA-256 hex digest of dataset bytes.
func ComputeDatasetHash(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

// NewReleaseEvidenceBundle constructs a ReleaseEvidenceBundle and computes its cryptographic digest.
func NewReleaseEvidenceBundle(version, commitSHA string, manifest BenchmarkManifest, report *Report, gates []ReleaseGateStatus) (*ReleaseEvidenceBundle, error) {
	if version == "" {
		return nil, fmt.Errorf("release version is required")
	}
	if commitSHA == "" {
		return nil, fmt.Errorf("commit SHA is required")
	}
	if err := manifest.Validate(); err != nil {
		return nil, fmt.Errorf("invalid manifest: %w", err)
	}

	bundle := &ReleaseEvidenceBundle{
		Version:          version,
		CommitSHA:        commitSHA,
		GeneratedAt:      time.Now().UTC(),
		Manifest:         manifest,
		ComplianceReport: report,
		GateStatus:       gates,
	}

	digest, err := bundle.computeDigest()
	if err != nil {
		return nil, err
	}
	bundle.BundleDigest = digest
	return bundle, nil
}

func (b *ReleaseEvidenceBundle) computeDigest() (string, error) {
	clone := *b
	clone.BundleDigest = ""
	raw, err := json.Marshal(clone)
	if err != nil {
		return "", fmt.Errorf("marshal bundle for digest: %w", err)
	}
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:]), nil
}

// AllGatesPassed returns true if at least one gate exists and every gate passed.
func (b *ReleaseEvidenceBundle) AllGatesPassed() bool {
	if len(b.GateStatus) == 0 {
		return false
	}
	for _, g := range b.GateStatus {
		if !g.Passed {
			return false
		}
	}
	return true
}

// Validate verifies the ReleaseEvidenceBundle fields, manifest, and digest integrity.
func (b *ReleaseEvidenceBundle) Validate() error {
	if b.Version == "" {
		return fmt.Errorf("bundle missing version")
	}
	if b.CommitSHA == "" {
		return fmt.Errorf("bundle missing commit SHA")
	}
	if b.GeneratedAt.IsZero() {
		return fmt.Errorf("bundle missing generated_at timestamp")
	}
	if err := b.Manifest.Validate(); err != nil {
		return fmt.Errorf("bundle manifest invalid: %w", err)
	}
	if len(b.GateStatus) == 0 {
		return fmt.Errorf("bundle must include at least one release gate status")
	}
	expectedDigest, err := b.computeDigest()
	if err != nil {
		return err
	}
	if b.BundleDigest != expectedDigest {
		return fmt.Errorf("bundle digest mismatch: got %s, want %s", b.BundleDigest, expectedDigest)
	}
	return nil
}

// SerializeJSON serializes the ReleaseEvidenceBundle to JSON.
func (b *ReleaseEvidenceBundle) SerializeJSON() ([]byte, error) {
	if err := b.Validate(); err != nil {
		return nil, err
	}
	return json.MarshalIndent(b, "", "  ")
}

// ParseReleaseEvidenceBundle deserializes and validates a ReleaseEvidenceBundle from JSON.
func ParseReleaseEvidenceBundle(data []byte) (*ReleaseEvidenceBundle, error) {
	var b ReleaseEvidenceBundle
	if err := json.Unmarshal(data, &b); err != nil {
		return nil, fmt.Errorf("unmarshal release evidence bundle: %w", err)
	}
	if err := b.Validate(); err != nil {
		return nil, err
	}
	return &b, nil
}
