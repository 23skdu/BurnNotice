package telemetry

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

// SubmissionMethod defines how telemetry data is submitted.
type SubmissionMethod int

const (
	SubmissionHTTPS SubmissionMethod = iota
	SubmissionLocalSave
)

// Report holds anonymized telemetry data. No prompts, outputs, IPs, or identity.
type Report struct {
	ModelArchitecture string  `json:"model_architecture"`
	ModelSize         string  `json:"model_size"`
	SteeringMethod    string  `json:"steering_method"`
	RefusalRate       float64 `json:"refusal_rate"`
	KLDivergence      float64 `json:"kl_divergence"`
	CapabilityRetention float64 `json:"capability_retention"`
	Hardware          string  `json:"hardware"`
	Timestamp         int64   `json:"timestamp"`
	Version           int     `json:"version"`
}

// Reporter handles anonymous telemetry submission.
type Reporter struct {
	Endpoint string
	Method   SubmissionMethod
	OutputDir string
	client   *http.Client
}

// NewReporter creates a reporter with the given configuration.
// If endpoint is empty, defaults to local file save.
func NewReporter(endpoint, outputDir string) *Reporter {
	method := SubmissionHTTPS
	if endpoint == "" {
		method = SubmissionLocalSave
	}
	return &Reporter{
		Endpoint:  endpoint,
		Method:    method,
		OutputDir: outputDir,
		client:    &http.Client{Timeout: 10 * time.Second},
	}
}

// Submit sends a telemetry report via the configured submission method.
func (r *Reporter) Submit(report *Report) error {
	report.Timestamp = time.Now().Unix()
	report.Version = 1

	switch r.Method {
	case SubmissionHTTPS:
		return r.submitHTTPS(report)
	case SubmissionLocalSave:
		return r.submitLocalSave(report)
	default:
		return fmt.Errorf("unknown submission method: %d", r.Method)
	}
}

func (r *Reporter) submitHTTPS(report *Report) error {
	data, err := json.Marshal(report)
	if err != nil {
		return fmt.Errorf("marshal report: %w", err)
	}

	resp, err := r.client.Post(r.Endpoint, "application/json", bytes.NewReader(data))
	if err != nil {
		return fmt.Errorf("http post: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		return fmt.Errorf("unexpected status: %s", resp.Status)
	}
	return nil
}

func (r *Reporter) submitLocalSave(report *Report) error {
	dir := r.OutputDir
	if dir == "" {
		dir = "./telemetry"
	}
	if err := os.MkdirAll(dir, 0750); err != nil {
		return fmt.Errorf("create telemetry dir: %w", err)
	}

	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal report: %w", err)
	}

	filename := fmt.Sprintf("telemetry_%d.json", report.Timestamp)
	path := filepath.Join(dir, filename)
	if err := os.WriteFile(path, data, 0600); err != nil {
		return fmt.Errorf("write telemetry file: %w", err)
	}
	return nil
}

// AggregateResult holds a summary of multiple telemetry reports.
type AggregateResult struct {
	TotalReports       int     `json:"total_reports"`
	AvgRefusalRate     float64 `json:"avg_refusal_rate"`
	AvgKLDivergence    float64 `json:"avg_kl_divergence"`
	AvgCapabilityRetention float64 `json:"avg_capability_retention"`
	ModelArchitectures []string `json:"model_architectures"`
	SteeringMethods    []string `json:"steering_methods"`
}

// Aggregate computes summary statistics from multiple reports.
func Aggregate(reports []Report) AggregateResult {
	result := AggregateResult{TotalReports: len(reports)}
	if len(reports) == 0 {
		return result
	}

	archSet := make(map[string]bool)
	methodSet := make(map[string]bool)

	for _, r := range reports {
		result.AvgRefusalRate += r.RefusalRate
		result.AvgKLDivergence += r.KLDivergence
		result.AvgCapabilityRetention += r.CapabilityRetention
		archSet[r.ModelArchitecture] = true
		methodSet[r.SteeringMethod] = true
	}

	n := float64(len(reports))
	result.AvgRefusalRate /= n
	result.AvgKLDivergence /= n
	result.AvgCapabilityRetention /= n

	result.ModelArchitectures = make([]string, 0, len(archSet))
	for a := range archSet {
		result.ModelArchitectures = append(result.ModelArchitectures, a)
	}
	result.SteeringMethods = make([]string, 0, len(methodSet))
	for m := range methodSet {
		result.SteeringMethods = append(result.SteeringMethods, m)
	}

	return result
}

// LoadReports reads all JSON report files from the specified directory.
func LoadReports(dir string) ([]Report, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("read telemetry dir: %w", err)
	}

	var reports []Report
	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".json" {
			continue
		}
		path := filepath.Join(dir, entry.Name())
		data, err := os.ReadFile(path) // #nosec G304 -- directory is user-provided CLI arg
		if err != nil {
			return nil, fmt.Errorf("read telemetry file %s: %w", entry.Name(), err)
		}
		var report Report
		if err := json.Unmarshal(data, &report); err != nil {
			return nil, fmt.Errorf("parse telemetry report %s: %w", entry.Name(), err)
		}
		reports = append(reports, report)
	}
	return reports, nil
}
