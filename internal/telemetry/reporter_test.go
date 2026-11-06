package telemetry

import (
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestReporter_LocalSave(t *testing.T) {
	tmpDir := t.TempDir()
	r := NewReporter("", tmpDir)

	report := &Report{
		ModelArchitecture:   "llama",
		ModelSize:           "7B",
		SteeringMethod:      "whitened_svd",
		RefusalRate:         0.05,
		KLDivergence:        0.12,
		CapabilityRetention: 0.99,
		Hardware:            "cuda",
	}

	if err := r.Submit(report); err != nil {
		t.Fatalf("Submit failed: %v", err)
	}

	// Verify local file exists and parse it
	files, err := os.ReadDir(tmpDir)
	if err != nil {
		t.Fatalf("failed to read tmpDir: %v", err)
	}
	if len(files) != 1 {
		t.Fatalf("expected 1 telemetry file, got %d", len(files))
	}

	content, err := os.ReadFile(filepath.Join(tmpDir, files[0].Name()))
	if err != nil {
		t.Fatalf("failed to read telemetry file: %v", err)
	}

	var parsed Report
	if err := json.Unmarshal(content, &parsed); err != nil {
		t.Fatalf("failed to parse telemetry file JSON: %v", err)
	}

	if parsed.ModelArchitecture != "llama" || parsed.ModelSize != "7B" || math.Abs(parsed.RefusalRate-0.05) > 1e-9 {
		t.Errorf("unexpected parsed report: %+v", parsed)
	}
}

func TestReporter_HTTPS(t *testing.T) {
	// Mock HTTP server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	r := NewReporter(server.URL, "")
	report := &Report{
		ModelArchitecture:   "llama",
		ModelSize:           "7B",
		SteeringMethod:      "whitened_svd",
		RefusalRate:         0.05,
		KLDivergence:        0.12,
		CapabilityRetention: 0.99,
		Hardware:            "cuda",
	}

	if err := r.Submit(report); err != nil {
		t.Fatalf("Submit HTTPS failed: %v", err)
	}
}

func TestAggregate(t *testing.T) {
	// 1. Empty case
	emptyRes := Aggregate(nil)
	if emptyRes.TotalReports != 0 {
		t.Errorf("expected 0 reports, got %d", emptyRes.TotalReports)
	}

	// 2. Multiple reports
	reports := []Report{
		{ModelArchitecture: "llama", SteeringMethod: "svd", RefusalRate: 0.1, KLDivergence: 0.2, CapabilityRetention: 0.95},
		{ModelArchitecture: "qwen", SteeringMethod: "svd", RefusalRate: 0.2, KLDivergence: 0.1, CapabilityRetention: 0.99},
	}

	res := Aggregate(reports)
	if res.TotalReports != 2 {
		t.Errorf("expected 2 reports, got %d", res.TotalReports)
	}
	if math.Abs(res.AvgRefusalRate-0.15) > 1e-9 {
		t.Errorf("expected avg refusal rate 0.15, got %f", res.AvgRefusalRate)
	}
	if math.Abs(res.AvgKLDivergence-0.15) > 1e-9 {
		t.Errorf("expected avg kl divergence 0.15, got %f", res.AvgKLDivergence)
	}
	if math.Abs(res.AvgCapabilityRetention-0.97) > 1e-9 {
		t.Errorf("expected avg capability retention 0.97, got %f", res.AvgCapabilityRetention)
	}
}

func TestReporter_ErrorsAndEdgeCases(t *testing.T) {
	report := &Report{
		ModelArchitecture: "llama",
		ModelSize:         "7B",
		SteeringMethod:    "whitened_svd",
	}

	// 1. Unknown submission method
	rUnknown := NewReporter("", "")
	rUnknown.Method = SubmissionMethod(999)
	if err := rUnknown.Submit(report); err == nil {
		t.Error("expected error for unknown submission method")
	}

	// 2. HTTPS status code error (non-2xx)
	server500 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server500.Close()
	rHttpsBadStatus := NewReporter(server500.URL, "")
	if err := rHttpsBadStatus.Submit(report); err == nil {
		t.Error("expected error for HTTP status 500")
	}

	// 3. HTTPS client connection error (invalid URL)
	rHttpsBadUrl := NewReporter("http://invalid.local.endpoint.nonexistent:99999", "")
	if err := rHttpsBadUrl.Submit(report); err == nil {
		t.Error("expected error for invalid http endpoint")
	}

	// 4. submitLocalSave empty OutputDir fallback
	rLocalDefault := NewReporter("", "")
	rLocalDefault.OutputDir = "" // forces "./telemetry" default
	// Clean up after the test
	defer os.RemoveAll("./telemetry")
	if err := rLocalDefault.Submit(report); err != nil {
		t.Fatalf("local default submit failed: %v", err)
	}

	// Verify default dir file
	files, err := os.ReadDir("./telemetry")
	if err != nil || len(files) != 1 {
		t.Errorf("expected 1 file in ./telemetry, got %d files, err: %v", len(files), err)
	}

	// 5. submitLocalSave mkdir failure (invalid path)
	rLocalBadDir := NewReporter("", "/nonexistent_dir/no_permission/telemetry")
	if err := rLocalBadDir.Submit(report); err == nil {
		t.Error("expected error for invalid directory creation in local save")
	}
}

func TestLoadReports(t *testing.T) {
	// 1. Nonexistent directory
	_, err := LoadReports("/nonexistent/directory/path")
	if err == nil {
		t.Error("expected error for nonexistent dir in LoadReports")
	}

	// 2. Valid directory with reports and non-json file
	tmpDir := t.TempDir()
	r1 := Report{ModelArchitecture: "llama", SteeringMethod: "dim", RefusalRate: 0.1}
	r2 := Report{ModelArchitecture: "qwen", SteeringMethod: "svd", RefusalRate: 0.2}

	b1, _ := json.Marshal(r1)
	b2, _ := json.Marshal(r2)

	if err := os.WriteFile(filepath.Join(tmpDir, "telemetry_1.json"), b1, 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "telemetry_2.json"), b2, 0600); err != nil {
		t.Fatal(err)
	}
	// Non-json file should be ignored
	if err := os.WriteFile(filepath.Join(tmpDir, "notes.txt"), []byte("ignore me"), 0600); err != nil {
		t.Fatal(err)
	}

	reports, err := LoadReports(tmpDir)
	if err != nil {
		t.Fatalf("LoadReports failed: %v", err)
	}
	if len(reports) != 2 {
		t.Fatalf("expected 2 reports, got %d", len(reports))
	}

	// 3. Corrupted JSON file
	badDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(badDir, "bad.json"), []byte("{not json"), 0600); err != nil {
		t.Fatal(err)
	}
	_, err = LoadReports(badDir)
	if err == nil {
		t.Error("expected error for malformed json in LoadReports")
	}
}
