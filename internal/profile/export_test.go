package profile

import (
	"math"
	"testing"
	"time"
)

func TestModelProfile_ToExportable(t *testing.T) {
	p := &ModelProfile{
		ModelID:         "test-model",
		Directions:      map[string]map[int][]float64{"deception": {0: {1, 2, 3}}},
		LayerScores:     map[string][]float64{"deception": {0.5}},
		TruthNeurons:    []int{1, 2},
		MonitoredLayers: []int{0, 1},
	}

	meta := ProfileMetadata{
		CreatedAt:       time.Now(),
		ModelArchitecture: "llama-3-8b",
		DatasetSize:     1000,
		ValidationScore: 0.95,
		Version:         1,
	}

	ep := p.ToExportable(meta)

	if ep.ModelID != "test-model" {
		t.Errorf("ModelID = %s, want test-model", ep.ModelID)
	}
	if ep.Metadata.ModelArchitecture != "llama-3-8b" {
		t.Errorf("ModelArchitecture = %s, want llama-3-8b", ep.Metadata.ModelArchitecture)
	}
	if ep.Metadata.DatasetSize != 1000 {
		t.Errorf("DatasetSize = %d, want 1000", ep.Metadata.DatasetSize)
	}
	if math.Abs(ep.Metadata.ValidationScore-0.95) > 1e-9 {
		t.Errorf("ValidationScore = %f, want 0.95", ep.Metadata.ValidationScore)
	}
}

func TestExportableProfile_FromExportable(t *testing.T) {
	ep := &ExportableProfile{
		ModelID:         "test-model",
		Directions:      map[string]map[int][]float64{"truth": {0: {0.1, 0.2}}},
		LayerScores:     map[string][]float64{"truth": {0.8}},
		TruthNeurons:    []int{0},
		MonitoredLayers: []int{0},
	}

	p := ep.FromExportable()

	if p.ModelID != "test-model" {
		t.Errorf("ModelID = %s, want test-model", p.ModelID)
	}
	if p.TruthNeurons[0] != 0 {
		t.Errorf("TruthNeurons[0] = %d, want 0", p.TruthNeurons[0])
	}
}

func TestModelProfile_Export_JSON(t *testing.T) {
	p := &ModelProfile{
		ModelID:    "test-model",
		Directions: map[string]map[int][]float64{"d": {0: {1, 2}}},
	}

	meta := ProfileMetadata{
		CreatedAt:       time.Now(),
		ModelArchitecture: "test",
		DatasetSize:     100,
		ValidationScore: 0.9,
		Version:         1,
	}

	data, err := p.Export(meta, "json")
	if err != nil {
		t.Fatalf("Export error: %v", err)
	}

	if len(data) == 0 {
		t.Fatal("exported data is empty")
	}
}

func TestModelProfile_Export_UnsupportedFormat(t *testing.T) {
	p := &ModelProfile{ModelID: "test"}

	_, err := p.Export(ProfileMetadata{}, "xml")
	if err == nil {
		t.Fatal("expected error for unsupported format")
	}
}

func TestImportProfile_JSON(t *testing.T) {
	p := &ModelProfile{
		ModelID:    "test-model",
		Directions: map[string]map[int][]float64{"d": {0: {1, 2}}},
	}

	meta := ProfileMetadata{
		CreatedAt:       time.Now(),
		ModelArchitecture: "test",
		DatasetSize:     100,
		ValidationScore: 0.9,
		Version:         1,
	}

	data, _ := p.Export(meta, "json")
	ep, err := ImportProfile(data, "json")
	if err != nil {
		t.Fatalf("ImportProfile error: %v", err)
	}

	if ep.ModelID != "test-model" {
		t.Errorf("imported ModelID = %s, want test-model", ep.ModelID)
	}
}

func TestImportProfile_UnsupportedFormat(t *testing.T) {
	_, err := ImportProfile([]byte("test"), "xml")
	if err == nil {
		t.Fatal("expected error for unsupported format")
	}
}

func TestImportProfile_InvalidJSON(t *testing.T) {
	_, err := ImportProfile([]byte("invalid json"), "json")
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
}

func TestComputeDiff(t *testing.T) {
	old := &ExportableProfile{
		Directions: map[string]map[int][]float64{
			"deception": {0: {1, 0, 0}},
		},
	}
	new := &ExportableProfile{
		Directions: map[string]map[int][]float64{
			"deception": {0: {1, 0, 0}},
		},
	}

	diff := ComputeDiff(old, new)

	if diff["deception"][0] != 0 {
		t.Errorf("diff for identical vectors = %f, want 0", diff["deception"][0])
	}
}

func TestComputeDiff_DifferentVectors(t *testing.T) {
	old := &ExportableProfile{
		Directions: map[string]map[int][]float64{
			"deception": {0: {1, 0, 0}},
		},
	}
	new := &ExportableProfile{
		Directions: map[string]map[int][]float64{
			"deception": {0: {0, 1, 0}},
		},
	}

	diff := ComputeDiff(old, new)

	if diff["deception"][0] <= 0 {
		t.Errorf("diff for orthogonal vectors = %f, want > 0", diff["deception"][0])
	}
}

func TestComputeDiff_MissingDirection(t *testing.T) {
	old := &ExportableProfile{
		Directions: map[string]map[int][]float64{
			"deception": {0: {1, 0}},
		},
	}
	new := &ExportableProfile{
		Directions: map[string]map[int][]float64{},
	}

	diff := ComputeDiff(old, new)

	if len(diff) != 0 {
		t.Errorf("diff should be empty for missing direction, got %v", diff)
	}
}

func TestDetectDrift(t *testing.T) {
	diff := map[string]map[int]float64{
		"deception": {0: 0.5, 1: 0.1},
		"truth":     {0: 0.8},
	}

	alerts := DetectDrift(diff, 0.3)

	// 0.5 > 0.3, 0.1 < 0.3, 0.8 > 0.3 => 2 alerts
	if len(alerts) != 2 {
		t.Fatalf("len(alerts) = %d, want 2", len(alerts))
	}
}

func TestDetectDrift_NoDrift(t *testing.T) {
	diff := map[string]map[int]float64{
		"deception": {0: 0.1},
	}

	alerts := DetectDrift(diff, 0.3)

	if len(alerts) != 0 {
		t.Errorf("len(alerts) = %d, want 0", len(alerts))
	}
}

func TestCosineDistance_Identical(t *testing.T) {
	a := []float64{1, 2, 3}
	b := []float64{1, 2, 3}

	dist := cosineDistance(a, b)
	if dist != 0 {
		t.Errorf("cosineDistance(identical) = %f, want 0", dist)
	}
}

func TestCosineDistance_Opposite(t *testing.T) {
	a := []float64{1, 0, 0}
	b := []float64{-1, 0, 0}

	dist := cosineDistance(a, b)
	if math.Abs(dist-2.0) > 1e-9 {
		t.Errorf("cosineDistance(opposite) = %f, want 2.0", dist)
	}
}

func TestCosineDistance_Orthogonal(t *testing.T) {
	a := []float64{1, 0, 0}
	b := []float64{0, 1, 0}

	dist := cosineDistance(a, b)
	if math.Abs(dist-1.0) > 1e-9 {
		t.Errorf("cosineDistance(orthogonal) = %f, want 1.0", dist)
	}
}

func TestCosineDistance_DifferentLengths(t *testing.T) {
	dist := cosineDistance([]float64{1, 0}, []float64{1, 0, 0})
	if dist != 1.0 {
		t.Errorf("cosineDistance(different lengths) = %f, want 1.0", dist)
	}
}

func TestCosineDistance_ZeroVector(t *testing.T) {
	dist := cosineDistance([]float64{0, 0}, []float64{1, 0})
	if dist != 1.0 {
		t.Errorf("cosineDistance(zero vector) = %f, want 1.0", dist)
	}
}
