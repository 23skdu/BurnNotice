package compliance

import (
	"math"
	"strings"
	"testing"
	"time"
)

func TestCapabilityEvaluator_Evaluate(t *testing.T) {
	ce := NewCapabilityEvaluator()

	t.Run("Safe Degradation", func(t *testing.T) {
		benchmarks := []CapabilityBenchmark{
			{Name: "MMLU", Baseline: 0.80, Current: 0.76},  // 5% degradation
			{Name: "GSM8K", Baseline: 0.90, Current: 0.85}, // 5.5% degradation
		}

		deg, safe := ce.Evaluate(benchmarks)
		if !safe {
			t.Errorf("expected benchmarks to be evaluated as safe, got safe = %t", safe)
		}
		if deg > 0.06 {
			t.Errorf("unexpected degradation calculation: %f", deg)
		}
	})

	t.Run("Unsafe Degradation", func(t *testing.T) {
		benchmarks := []CapabilityBenchmark{
			{Name: "MMLU", Baseline: 0.80, Current: 0.60},  // 25% degradation
			{Name: "GSM8K", Baseline: 0.90, Current: 0.80}, // 11% degradation
		}

		_, safe := ce.Evaluate(benchmarks)
		if safe {
			t.Errorf("expected benchmarks to be flagged as unsafe")
		}
	})

	t.Run("Empty and Improved", func(t *testing.T) {
		deg, safe := ce.Evaluate(nil)
		if !safe || deg != 0.0 {
			t.Errorf("expected empty benchmarks to be safe with 0 degradation, got %f, %t", deg, safe)
		}

		benchmarks := []CapabilityBenchmark{
			{Name: "ZeroBaseline", Baseline: 0.0, Current: 0.50},
			{Name: "Improved", Baseline: 0.80, Current: 0.90},
		}
		deg, safe = ce.Evaluate(benchmarks)
		if !safe || deg != 0.0 {
			t.Errorf("expected improved benchmarks to be safe with 0 degradation, got %f, %t", deg, safe)
		}
	})
}

func TestSteeringController_UpdateSteer(t *testing.T) {
	sc := NewSteeringController(2.0)

	t.Run("Safe Path", func(t *testing.T) {
		benchmarks := []CapabilityBenchmark{
			{Name: "MMLU", Baseline: 0.80, Current: 0.78},
		}

		strength, rolledBack := sc.UpdateSteer(benchmarks)
		if rolledBack {
			t.Errorf("expected no rollback on safe path")
		}
		if math.Abs(strength-2.0) > 1e-6 {
			t.Errorf("expected strength to remain at base 2.0, got %f", strength)
		}
	})

	t.Run("Trigger Rollback Path", func(t *testing.T) {
		benchmarks := []CapabilityBenchmark{
			{Name: "MMLU", Baseline: 0.80, Current: 0.40}, // 50% degradation
		}

		strength, rolledBack := sc.UpdateSteer(benchmarks)
		if !rolledBack {
			t.Errorf("expected automated rollback to trigger on unsafe degradation")
		}
		expectedStrength := 2.0 * (1.0 - 0.50) // 1.0
		if math.Abs(strength-expectedStrength) > 1e-6 {
			t.Errorf("expected strength rolled back to %f, got %f", expectedStrength, strength)
		}
	})

	t.Run("Extreme Degradation Clamp", func(t *testing.T) {
		benchmarks := []CapabilityBenchmark{
			{Name: "MMLU", Baseline: 0.80, Current: -0.40}, // >100% degradation
		}
		strength, rolledBack := sc.UpdateSteer(benchmarks)
		if !rolledBack || strength != 0 {
			t.Errorf("expected strength clamped to 0, got %f (rolledBack=%t)", strength, rolledBack)
		}
	})
}

func TestBenchmarkManifest_RoundTripAndOracle(t *testing.T) {
	datasetBytes := []byte(`{"pairs":[{"truthful":{"prompt":"p","response":"r"},"deceptive":{"prompt":"p","response":"d"}}]}`)
	dsHash := ComputeDatasetHash(datasetBytes)
	if len(dsHash) != 64 {
		t.Fatalf("expected 64-char SHA-256 hex digest, got %q", dsHash)
	}

	results := []BenchmarkResult{
		{
			Name:        "BenchmarkScoreActivation",
			Iterations:  100000,
			NsPerOp:     42.5,
			BytesPerOp:  0,
			AllocsPerOp: 0,
			P50Latency:  0.04,
			P99Latency:  0.12,
			AUROC:       0.94,
		},
	}

	manifest := NewBenchmarkManifest("", "qwen", dsHash, "abc1234def5678", results)
	manifest.CapabilityResults = []CapabilityBenchmark{
		{Name: "MMLU", Baseline: 0.78, Current: 0.76},
	}

	if manifest.ONNXRuntimeVersion != "1.24.2" {
		t.Errorf("expected default ONNXRuntimeVersion 1.24.2, got %s", manifest.ONNXRuntimeVersion)
	}

	raw, err := manifest.SerializeJSON()
	if err != nil {
		t.Fatalf("SerializeJSON failed: %v", err)
	}

	parsed, err := ParseBenchmarkManifest(raw)
	if err != nil {
		t.Fatalf("ParseBenchmarkManifest failed: %v", err)
	}

	if parsed.ModelFamily != "qwen" || parsed.DatasetHash != dsHash || parsed.CommitSHA != "abc1234def5678" {
		t.Errorf("parsed manifest mismatch: %+v", parsed)
	}
	if len(parsed.BenchmarkResults) != 1 || parsed.BenchmarkResults[0].NsPerOp != 42.5 {
		t.Errorf("parsed benchmark results mismatch: %+v", parsed.BenchmarkResults)
	}
}

func TestBenchmarkManifest_ValidationErrors(t *testing.T) {
	base := NewBenchmarkManifest("1.24.2", "llama", "deadbeef", "1234567", []BenchmarkResult{
		{Name: "Bench1", Iterations: 10, NsPerOp: 10.0},
	})

	cases := []struct {
		name   string
		mutate func(*BenchmarkManifest)
	}{
		{"missing OS", func(m *BenchmarkManifest) { m.OS = "" }},
		{"missing Arch", func(m *BenchmarkManifest) { m.Arch = "" }},
		{"missing GoVersion", func(m *BenchmarkManifest) { m.GoVersion = "" }},
		{"missing ONNXRuntimeVersion", func(m *BenchmarkManifest) { m.ONNXRuntimeVersion = "" }},
		{"missing ModelFamily", func(m *BenchmarkManifest) { m.ModelFamily = "" }},
		{"missing DatasetHash", func(m *BenchmarkManifest) { m.DatasetHash = "" }},
		{"missing CommitSHA", func(m *BenchmarkManifest) { m.CommitSHA = "" }},
		{"missing Timestamp", func(m *BenchmarkManifest) { m.Timestamp = time.Time{} }},
		{"empty result name", func(m *BenchmarkManifest) { m.BenchmarkResults[0].Name = "" }},
		{"non-positive iterations", func(m *BenchmarkManifest) { m.BenchmarkResults[0].Iterations = 0 }},
		{"negative ns/op", func(m *BenchmarkManifest) { m.BenchmarkResults[0].NsPerOp = -1.0 }},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cp := base
			cp.BenchmarkResults = append([]BenchmarkResult(nil), base.BenchmarkResults...)
			tc.mutate(&cp)
			if err := cp.Validate(); err == nil {
				t.Errorf("expected validation error for %s", tc.name)
			}
			if _, err := cp.SerializeJSON(); err == nil {
				t.Errorf("expected SerializeJSON error for %s", tc.name)
			}
		})
	}

	if _, err := ParseBenchmarkManifest([]byte("{bad json")); err == nil {
		t.Error("expected error for malformed JSON")
	}
}

func TestReleaseEvidenceBundle_RoundTripAndTamperOracle(t *testing.T) {
	manifest := NewBenchmarkManifest("1.24.2", "gemma", ComputeDatasetHash([]byte("test-ds")), "commit999", []BenchmarkResult{
		{Name: "BenchmarkSteer", Iterations: 5000, NsPerOp: 120.0, AUROC: 0.91},
	})

	report := &Report{
		GeneratedAt:     time.Now().UTC(),
		TotalAlerts:     4,
		ComplianceScore: 0.95,
	}

	gates := []ReleaseGateStatus{
		{GateName: "unit_tests", Passed: true, Details: "all packages pass"},
		{GateName: "lint_zero", Passed: true, Details: "0 issues"},
		{GateName: "capability_retention", Passed: true, Details: "degradation <= 15%"},
	}

	bundle, err := NewReleaseEvidenceBundle("v1.0.0", "commit999", manifest, report, gates)
	if err != nil {
		t.Fatalf("NewReleaseEvidenceBundle failed: %v", err)
	}

	if !bundle.AllGatesPassed() {
		t.Error("expected AllGatesPassed() to be true")
	}

	raw, err := bundle.SerializeJSON()
	if err != nil {
		t.Fatalf("bundle.SerializeJSON failed: %v", err)
	}

	parsed, err := ParseReleaseEvidenceBundle(raw)
	if err != nil {
		t.Fatalf("ParseReleaseEvidenceBundle failed: %v", err)
	}
	if parsed.BundleDigest != bundle.BundleDigest {
		t.Errorf("digest mismatch: got %s, want %s", parsed.BundleDigest, bundle.BundleDigest)
	}

	// Oracle test: tampering with a field invalidates the cryptographic digest
	tamperedJSON := strings.Replace(string(raw), `"v1.0.0"`, `"v1.0.1"`, 1)
	if _, err := ParseReleaseEvidenceBundle([]byte(tamperedJSON)); err == nil {
		t.Error("expected digest verification error on tampered bundle JSON")
	}

	// Gate failure detection
	parsed.GateStatus[1].Passed = false
	if parsed.AllGatesPassed() {
		t.Error("expected AllGatesPassed() to be false when a gate fails")
	}
}

func TestReleaseEvidenceBundle_ValidationErrors(t *testing.T) {
	manifest := NewBenchmarkManifest("1.24.2", "llama", "hash123", "sha123", nil)
	gates := []ReleaseGateStatus{{GateName: "test", Passed: true}}

	if _, err := NewReleaseEvidenceBundle("", "sha123", manifest, nil, gates); err == nil {
		t.Error("expected error for empty version")
	}
	if _, err := NewReleaseEvidenceBundle("v1.0.0", "", manifest, nil, gates); err == nil {
		t.Error("expected error for empty commit SHA")
	}
	badManifest := manifest
	badManifest.ModelFamily = ""
	if _, err := NewReleaseEvidenceBundle("v1.0.0", "sha123", badManifest, nil, gates); err == nil {
		t.Error("expected error for invalid manifest")
	}

	bundle, err := NewReleaseEvidenceBundle("v1.0.0", "sha123", manifest, nil, gates)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	emptyGatesBundle := *bundle
	emptyGatesBundle.GateStatus = nil
	if emptyGatesBundle.AllGatesPassed() {
		t.Error("expected AllGatesPassed to be false for empty gates")
	}
	if err := emptyGatesBundle.Validate(); err == nil {
		t.Error("expected validation error when GateStatus is empty")
	}

	if _, err := ParseReleaseEvidenceBundle([]byte("invalid json")); err == nil {
		t.Error("expected error for invalid JSON")
	}
}
