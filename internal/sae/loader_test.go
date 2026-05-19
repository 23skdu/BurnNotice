package sae

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestSAELoader_LoadFromMemory(t *testing.T) {
	loader := &SAELoader{}
	weights := &SAEWeights{
		EncoderWeights: [][]float64{{1, 0}, {0, 1}},
		DecoderWeights: [][]float64{{1, 0}, {0, 1}},
		EncoderBias:    []float64{0, 0},
		DecoderBias:    []float64{0, 0},
		FeatureCount:   2,
		InputDim:       2,
		LayerIndex:     0,
	}

	if err := loader.LoadFromMemory(weights); err != nil {
		t.Fatalf("LoadFromMemory error: %v", err)
	}
}

func TestSAELoader_LoadFromMemory_InvalidFeatureCount(t *testing.T) {
	loader := &SAELoader{}
	weights := &SAEWeights{
		FeatureCount: 0,
		InputDim:     2,
	}

	if err := loader.LoadFromMemory(weights); err == nil {
		t.Fatal("expected error for invalid feature count")
	}
}

func TestSAELoader_LoadFromMemory_InvalidInputDim(t *testing.T) {
	loader := &SAELoader{}
	weights := &SAEWeights{
		FeatureCount: 2,
		InputDim:     0,
	}

	if err := loader.LoadFromMemory(weights); err == nil {
		t.Fatal("expected error for invalid input dim")
	}
}

func TestSAELoader_Load(t *testing.T) {
	loader := &SAELoader{}
	content := `{"encoder_weights":[[1,0],[0,1]],"decoder_weights":[[1,0],[0,1]],"encoder_bias":[0,0],"decoder_bias":[0,0],"feature_count":2,"input_dim":2,"layer_index":0}`
	tmp := writeTempSAEFile(t, content)
	defer os.Remove(tmp)

	if err := loader.Load(tmp); err != nil {
		t.Fatalf("Load error: %v", err)
	}
}

func TestSAELoader_Load_MissingFile(t *testing.T) {
	loader := &SAELoader{}
	if err := loader.Load("/nonexistent/sae.json"); err == nil {
		t.Fatal("expected error for missing file")
	}
}

func TestSAELoader_Load_InvalidJSON(t *testing.T) {
	loader := &SAELoader{}
	tmp := writeTempSAEFile(t, "invalid json")
	defer os.Remove(tmp)

	if err := loader.Load(tmp); err == nil {
		t.Fatal("expected error for invalid JSON")
	}
}

func TestSAELoader_Encode(t *testing.T) {
	loader := &SAELoader{}
	weights := &SAEWeights{
		EncoderWeights: [][]float64{{1, 0}, {0, 1}},
		DecoderWeights: [][]float64{{1, 0}, {0, 1}},
		EncoderBias:    []float64{0, 0},
		DecoderBias:    []float64{0, 0},
		FeatureCount:   2,
		InputDim:       2,
	}
	loader.LoadFromMemory(weights)

	features, err := loader.Encode([]float64{3, 4})
	if err != nil {
		t.Fatalf("Encode error: %v", err)
	}

	if len(features) != 2 {
		t.Fatalf("len(features) = %d, want 2", len(features))
	}
	if math.Abs(features[0]-3) > 1e-9 {
		t.Errorf("features[0] = %f, want 3", features[0])
	}
	if math.Abs(features[1]-4) > 1e-9 {
		t.Errorf("features[1] = %f, want 4", features[1])
	}
}

func TestSAELoader_Encode_ReLU(t *testing.T) {
	loader := &SAELoader{}
	weights := &SAEWeights{
		EncoderWeights: [][]float64{{1, 0}, {0, 1}},
		DecoderWeights: [][]float64{{1, 0}, {0, 1}},
		EncoderBias:    []float64{0, 0},
		DecoderBias:    []float64{0, 0},
		FeatureCount:   2,
		InputDim:       2,
	}
	loader.LoadFromMemory(weights)

	features, err := loader.Encode([]float64{-3, 4})
	if err != nil {
		t.Fatalf("Encode error: %v", err)
	}

	if features[0] != 0 {
		t.Errorf("ReLU should zero negative: features[0] = %f", features[0])
	}
	if features[1] != 4 {
		t.Errorf("features[1] = %f, want 4", features[1])
	}
}

func TestSAELoader_Encode_NotLoaded(t *testing.T) {
	loader := &SAELoader{}
	_, err := loader.Encode([]float64{1, 2})
	if err == nil {
		t.Fatal("expected error when weights not loaded")
	}
}

func TestSAELoader_Encode_WrongDim(t *testing.T) {
	loader := &SAELoader{}
	weights := &SAEWeights{
		EncoderWeights: [][]float64{{1, 0}},
		DecoderWeights: [][]float64{{1, 0}},
		EncoderBias:    []float64{0},
		DecoderBias:    []float64{0},
		FeatureCount:   1,
		InputDim:       2,
	}
	loader.LoadFromMemory(weights)

	_, err := loader.Encode([]float64{1})
	if err == nil {
		t.Fatal("expected error for wrong activation dim")
	}
}

func TestSAELoader_Decode(t *testing.T) {
	loader := &SAELoader{}
	weights := &SAEWeights{
		EncoderWeights: [][]float64{{1, 0}, {0, 1}},
		DecoderWeights: [][]float64{{1, 0}, {0, 1}},
		EncoderBias:    []float64{0, 0},
		DecoderBias:    []float64{0, 0},
		FeatureCount:   2,
		InputDim:       2,
	}
	loader.LoadFromMemory(weights)

	reconstructed, err := loader.Decode([]float64{3, 4})
	if err != nil {
		t.Fatalf("Decode error: %v", err)
	}

	if math.Abs(reconstructed[0]-3) > 1e-9 {
		t.Errorf("reconstructed[0] = %f, want 3", reconstructed[0])
	}
	if math.Abs(reconstructed[1]-4) > 1e-9 {
		t.Errorf("reconstructed[1] = %f, want 4", reconstructed[1])
	}
}

func TestSAELoader_TopKFeatures(t *testing.T) {
	loader := &SAELoader{}
	features := []float64{0.1, 0.5, 0.3, 0.9, 0.2}

	indices, values := loader.TopKFeatures(features, 3)

	if len(indices) != 3 {
		t.Fatalf("len(indices) = %d, want 3", len(indices))
	}

	if indices[0] != 3 || values[0] != 0.9 {
		t.Errorf("top feature = (%d, %f), want (3, 0.9)", indices[0], values[0])
	}
	if indices[1] != 1 || values[1] != 0.5 {
		t.Errorf("second feature = (%d, %f), want (1, 0.5)", indices[1], values[1])
	}
}

func TestSAELoader_TopKFeatures_KGreaterThanFeatures(t *testing.T) {
	loader := &SAELoader{}
	features := []float64{0.1, 0.5}

	indices, values := loader.TopKFeatures(features, 10)

	if len(indices) != 2 {
		t.Fatalf("len(indices) = %d, want 2", len(indices))
	}
	if len(values) != 2 {
		t.Fatalf("len(values) = %d, want 2", len(values))
	}
}

func TestSAELoader_Attribution(t *testing.T) {
	loader := &SAELoader{}
	features := []float64{0.1, 0, 0.3, 0, 0.5}

	attr := loader.Attribution(features)

	if len(attr) != 3 {
		t.Fatalf("len(attr) = %d, want 3", len(attr))
	}
	if attr[0] != 0.1 || attr[2] != 0.3 || attr[4] != 0.5 {
		t.Errorf("attr = %v, want {0:0.1, 2:0.3, 4:0.5}", attr)
	}
}

func TestSAELoader_ReconstructionError(t *testing.T) {
	loader := &SAELoader{}
	weights := &SAEWeights{
		EncoderWeights: [][]float64{{1, 0}, {0, 1}},
		DecoderWeights: [][]float64{{1, 0}, {0, 1}},
		EncoderBias:    []float64{0, 0},
		DecoderBias:    []float64{0, 0},
		FeatureCount:   2,
		InputDim:       2,
	}
	loader.LoadFromMemory(weights)

	error, err := loader.ReconstructionError([]float64{3, 4})
	if err != nil {
		t.Fatalf("ReconstructionError error: %v", err)
	}

	if error > 1e-9 {
		t.Errorf("reconstruction error = %f, want ~0", error)
	}
}

func TestSAELoader_ReconstructionError_IdentitySAE(t *testing.T) {
	loader := &SAELoader{}
	weights := &SAEWeights{
		EncoderWeights: [][]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
		DecoderWeights: [][]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
		EncoderBias:    []float64{0, 0, 0},
		DecoderBias:    []float64{0, 0, 0},
		FeatureCount:   3,
		InputDim:       3,
	}
	loader.LoadFromMemory(weights)

	activation := []float64{1, 2, 3}
	error, err := loader.ReconstructionError(activation)
	if err != nil {
		t.Fatalf("ReconstructionError error: %v", err)
	}

	if error > 1e-9 {
		t.Errorf("identity SAE reconstruction error = %f, want ~0", error)
	}
}

func writeTempSAEFile(t *testing.T, content string) string {
	t.Helper()
	tmp := filepath.Join(t.TempDir(), "sae.json")
	if err := os.WriteFile(tmp, []byte(content), 0600); err != nil {
		t.Fatalf("write temp file: %v", err)
	}
	return tmp
}
