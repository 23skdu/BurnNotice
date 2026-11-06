package sae

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
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

func TestSAELoader_LoadGemmaScope(t *testing.T) {
	// 1. Create mock configuration json
	configJSON := `{"model_id":"gemma-2b","layer_index":12,"hidden_dim":3,"feature_dim":4,"l1_coefficient":0.0001}`
	tmpConfig := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(tmpConfig, []byte(configJSON), 0600); err != nil {
		t.Fatalf("failed to write config.json: %v", err)
	}

	// 2. Create mock .npy contents using Python format headers
	// encoder_weights shape: (4, 3) (features × input_dim)
	encWHeader := `{'descr': '<f4', 'fortran_order': False, 'shape': (4, 3), }`
	encWData := []float32{
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
		0.5, 0.5, 0.5,
	}
	encWBytes := createMockNpy(encWData, encWHeader)

	// encoder_bias shape: (4,)
	encBHeader := `{'descr': '<f4', 'fortran_order': False, 'shape': (4,), }`
	encBData := []float32{0.1, 0.2, 0.3, 0.4}
	encBBytes := createMockNpy(encBData, encBHeader)

	// decoder_weights shape: (3, 4) (input_dim × features)
	// parseNpyMatrixTransposed reads shape (3, 4) and returns a feature_dim × hidden_dim matrix (4 × 3).
	decWHeader := `{'descr': '<f4', 'fortran_order': False, 'shape': (3, 4), }`
	decWData := []float32{
		1, 0, 0, 0.5,
		0, 1, 0, 0.5,
		0, 0, 1, 0.5,
	}
	decWBytes := createMockNpy(decWData, decWHeader)

	// decoder_bias shape: (3,)
	decBHeader := `{'descr': '<f4', 'fortran_order': False, 'shape': (3,), }`
	decBData := []float32{0.5, 0.6, 0.7}
	decBBytes := createMockNpy(decBData, decBHeader)

	// 3. Put them in zip
	zipFiles := map[string][]byte{
		"encoder_weights.npy": encWBytes,
		"encoder_bias.npy":    encBBytes,
		"decoder_weights.npy": decWBytes,
		"decoder_bias.npy":    decBBytes,
	}
	zipBytes := createMockNpz(zipFiles)

	tmpNpz := filepath.Join(t.TempDir(), "sae.npz")
	if err := os.WriteFile(tmpNpz, zipBytes, 0600); err != nil {
		t.Fatalf("failed to write npz: %v", err)
	}

	// 4. Test loading
	loader := &SAELoader{}
	if err := loader.LoadGemmaScope(tmpNpz, tmpConfig); err != nil {
		t.Fatalf("LoadGemmaScope failed: %v", err)
	}

	// Verify loaded weights
	if loader.weights.FeatureCount != 4 {
		t.Errorf("loaded feature count = %d, want 4", loader.weights.FeatureCount)
	}
	if loader.weights.InputDim != 3 {
		t.Errorf("loaded input dim = %d, want 3", loader.weights.InputDim)
	}
	if loader.weights.LayerIndex != 12 {
		t.Errorf("loaded layer index = %d, want 12", loader.weights.LayerIndex)
	}

	// Verify specific weights/biases
	if math.Abs(loader.weights.EncoderBias[0]-0.1) > 1e-6 || math.Abs(loader.weights.DecoderBias[2]-0.7) > 1e-6 {
		t.Errorf("unexpected bias values: encoder_bias[0] = %f, decoder_bias[2] = %f",
			loader.weights.EncoderBias[0], loader.weights.DecoderBias[2])
	}
}

func createMockNpy(data []float32, pythonHeader string) []byte {
	var buf bytes.Buffer
	buf.WriteString("\x93NUMPY")
	buf.WriteByte(1)
	buf.WriteByte(0)

	headerBytes := []byte(pythonHeader)
	headerLen := uint16(len(headerBytes))
	binary.Write(&buf, binary.LittleEndian, headerLen)
	buf.Write(headerBytes)

	for _, val := range data {
		binary.Write(&buf, binary.LittleEndian, val)
	}

	return buf.Bytes()
}

func createMockNpz(files map[string][]byte) []byte {
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	for name, data := range files {
		w, err := zw.Create(name)
		if err != nil {
			panic(err)
		}
		_, err = w.Write(data)
		if err != nil {
			panic(err)
		}
	}
	zw.Close()
	return buf.Bytes()
}

func createMockNpyV2(data []float32, pythonHeader string) []byte {
	var buf bytes.Buffer
	buf.WriteString("\x93NUMPY")
	buf.WriteByte(2) // version 2
	buf.WriteByte(0)

	headerBytes := []byte(pythonHeader)
	headerLen := uint32(len(headerBytes))
	binary.Write(&buf, binary.LittleEndian, headerLen)
	buf.Write(headerBytes)

	for _, val := range data {
		binary.Write(&buf, binary.LittleEndian, val)
	}

	return buf.Bytes()
}

func TestSAELoader_AdditionalEdgeCases(t *testing.T) {
	// 1. Load with invalid fields in JSON
	loader := &SAELoader{}
	if err := loader.Load(writeTempSAEFile(t, `{"feature_count":0,"input_dim":2}`)); err == nil {
		t.Error("expected error for feature_count=0 in Load")
	}
	if err := loader.Load(writeTempSAEFile(t, `{"feature_count":2,"input_dim":0}`)); err == nil {
		t.Error("expected error for input_dim=0 in Load")
	}

	// 2. Decode errors
	loaderUnloaded := &SAELoader{}
	if _, err := loaderUnloaded.Decode([]float64{1, 2}); err == nil {
		t.Error("expected error for Decode on unloaded weights")
	}

	weights := &SAEWeights{
		EncoderWeights: [][]float64{{1, 0}, {0, 1}},
		DecoderWeights: [][]float64{{1, 0}, {0, 1}},
		EncoderBias:    []float64{0, 0},
		DecoderBias:    []float64{0, 0},
		FeatureCount:   2,
		InputDim:       2,
	}
	loader.LoadFromMemory(weights)
	if _, err := loader.Decode([]float64{1}); err == nil {
		t.Error("expected error for Decode with wrong feature dimension")
	}

	// 3. ReconstructionError errors
	if _, err := loaderUnloaded.ReconstructionError([]float64{1, 2}); err == nil {
		t.Error("expected error for ReconstructionError when unloaded")
	}
	if _, err := loader.ReconstructionError([]float64{1}); err == nil {
		t.Error("expected error for ReconstructionError with wrong activation dimension")
	}

	// 4. LoadGemmaScope config/NPZ reading failures
	if err := loader.LoadGemmaScope("sae.npz", "/nonexistent/config.json"); err == nil {
		t.Error("expected error for nonexistent config path in LoadGemmaScope")
	}
	badConfigPath := filepath.Join(t.TempDir(), "bad_config.json")
	os.WriteFile(badConfigPath, []byte("{invalid"), 0600)
	if err := loader.LoadGemmaScope("sae.npz", badConfigPath); err == nil {
		t.Error("expected error for invalid config JSON in LoadGemmaScope")
	}

	if err := loader.LoadGemmaScope("/nonexistent/sae.npz", ""); err == nil {
		t.Error("expected error for nonexistent NPZ in LoadGemmaScope")
	}

	badNpzPath := filepath.Join(t.TempDir(), "bad.npz")
	os.WriteFile(badNpzPath, []byte("not a zip"), 0600)
	if err := loader.LoadGemmaScope(badNpzPath, ""); err == nil {
		t.Error("expected error for invalid zip in LoadGemmaScope")
	}

	// 5. LoadGemmaScope NPZ missing elements
	validNpy := createMockNpy([]float32{0, 0}, `{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }`)

	// Missing encoder weights
	zipBytes1 := createMockNpz(map[string][]byte{"encoder_bias.npy": validNpy})
	npz1 := filepath.Join(t.TempDir(), "sae1.npz")
	os.WriteFile(npz1, zipBytes1, 0600)
	if err := loader.LoadGemmaScope(npz1, ""); err == nil {
		t.Error("expected error for NPZ missing encoder weights")
	}

	// Missing decoder weights
	zipBytes2 := createMockNpz(map[string][]byte{"encoder_weights.npy": validNpy})
	npz2 := filepath.Join(t.TempDir(), "sae2.npz")
	os.WriteFile(npz2, zipBytes2, 0600)
	if err := loader.LoadGemmaScope(npz2, ""); err == nil {
		t.Error("expected error for NPZ missing decoder weights")
	}

	// Missing encoder bias
	zipBytes3 := createMockNpz(map[string][]byte{"encoder_weights.npy": validNpy, "decoder_weights.npy": validNpy})
	npz3 := filepath.Join(t.TempDir(), "sae3.npz")
	os.WriteFile(npz3, zipBytes3, 0600)
	if err := loader.LoadGemmaScope(npz3, ""); err == nil {
		t.Error("expected error for NPZ missing encoder bias")
	}

	// Missing decoder bias
	zipBytes4 := createMockNpz(map[string][]byte{"encoder_weights.npy": validNpy, "decoder_weights.npy": validNpy, "encoder_bias.npy": validNpy})
	npz4 := filepath.Join(t.TempDir(), "sae4.npz")
	os.WriteFile(npz4, zipBytes4, 0600)
	if err := loader.LoadGemmaScope(npz4, ""); err == nil {
		t.Error("expected error for NPZ missing decoder bias")
	}

	// 6. parseNpyMatrix edge cases
	// Not 2D
	not2D := createMockNpy([]float32{0, 0}, `{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }`)
	if _, err := parseNpyMatrix(not2D); err == nil {
		t.Error("expected error for non-2D npy matrix")
	}

	// Truncated data
	truncated := createMockNpy([]float32{1, 2}, `{'descr': '<f4', 'fortran_order': False, 'shape': (2, 2), }`)
	if _, err := parseNpyMatrix(truncated[:len(truncated)-2]); err == nil {
		t.Error("expected error for truncated npy matrix")
	}

	// 7. parseNpyMatrixTransposed edge cases
	// Empty matrix (0 shape)
	emptyNpy := createMockNpy([]float32{}, `{'descr': '<f4', 'fortran_order': False, 'shape': (0, 2), }`)
	if _, err := parseNpyMatrixTransposed(emptyNpy); err == nil {
		t.Error("expected error for empty matrix transposition")
	}

	// 8. parseNpyVector edge cases
	// Not 1D
	not1D := createMockNpy([]float32{1, 2, 3, 4}, `{'descr': '<f4', 'fortran_order': False, 'shape': (2, 2), }`)
	if _, err := parseNpyVector(not1D); err == nil {
		t.Error("expected error for non-1D vector")
	}

	// Truncated vector
	truncatedVec := createMockNpy([]float32{1, 2}, `{'descr': '<f4', 'fortran_order': False, 'shape': (4,), }`)
	if _, err := parseNpyVector(truncatedVec[:len(truncatedVec)-2]); err == nil {
		t.Error("expected error for truncated vector")
	}

	// 9. readNpyHeader edge cases
	// Too short
	if _, _, err := readNpyHeader([]byte("short")); err == nil {
		t.Error("expected error for short npy data")
	}
	// Invalid magic bytes
	if _, _, err := readNpyHeader([]byte("NOTNUMPY  ")); err == nil {
		t.Error("expected error for invalid npy magic")
	}
	// Header len extends beyond data
	invalidHeaderLen := []byte("\x93NUMPY\x01\x00\xff\xff{'descr':'<f4','shape':(2,)}")
	if _, _, err := readNpyHeader(invalidHeaderLen); err == nil {
		t.Error("expected error when header len extends beyond data")
	}
	// Parse npy header JSON error
	badHeaderStr := createMockNpy([]float32{1, 2}, `{'descr': '<f4', 'fortran_order': False, 'shape': (2,`)
	if _, _, err := readNpyHeader(badHeaderStr); err == nil {
		t.Error("expected error for invalid python dict syntax")
	}
	// Unsupported dtype
	badDtype := createMockNpy([]float32{1, 2}, `{'descr': '<i8', 'fortran_order': False, 'shape': (2,), }`)
	if _, _, err := readNpyHeader(badDtype); err == nil {
		t.Error("expected error for unsupported dtype <i8")
	}

	// 10. v2 header parse check
	v2Bytes := createMockNpyV2([]float32{1, 2}, `{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }`)
	shape, fortran, err := readNpyHeader(v2Bytes)
	if err != nil {
		t.Errorf("readNpyHeader v2 error: %v", err)
	}
	if len(shape) != 1 || shape[0] != 2 || fortran {
		t.Errorf("unexpected v2 parse output: shape=%v fortran=%v", shape, fortran)
	}

	// v2 header too short
	if _, _, err := readNpyHeader([]byte("\x93NUMPY\x02\x00\x00\x00\x00")); err == nil {
		t.Error("expected error for short v2 header")
	}

	// 11. findNpyDataOffset edge cases
	if offset := findNpyDataOffset([]byte("short")); offset != 5 {
		t.Errorf("expected findNpyDataOffset to return length for short data, got %d", offset)
	}
}

func TestSAELoader_LoadGemmaScope_AlternativeNamesAndNoConfig(t *testing.T) {
	// 1. Create mock .npy contents for W_enc, W_dec, b_enc, b_dec
	encWHeader := `{'descr': '<f4', 'fortran_order': False, 'shape': (4, 3), }`
	encWData := []float32{
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
		0.5, 0.5, 0.5,
	}
	encWBytes := createMockNpy(encWData, encWHeader)

	encBHeader := `{'descr': '<f4', 'fortran_order': False, 'shape': (4,), }`
	encBData := []float32{0.1, 0.2, 0.3, 0.4}
	encBBytes := createMockNpy(encBData, encBHeader)

	decWHeader := `{'descr': '<f4', 'fortran_order': False, 'shape': (3, 4), }`
	decWData := []float32{
		1, 0, 0, 0.5,
		0, 1, 0, 0.5,
		0, 0, 1, 0.5,
	}
	decWBytes := createMockNpy(decWData, decWHeader)

	decBHeader := `{'descr': '<f4', 'fortran_order': False, 'shape': (3,), }`
	decBData := []float32{0.5, 0.6, 0.7}
	decBBytes := createMockNpy(decBData, decBHeader)

	zipFiles := map[string][]byte{
		"W_enc.npy": encWBytes,
		"b_enc.npy": encBBytes,
		"W_dec.npy": decWBytes,
		"b_dec.npy": decBBytes,
	}
	zipBytes := createMockNpz(zipFiles)

	tmpNpz := filepath.Join(t.TempDir(), "sae_alt.npz")
	if err := os.WriteFile(tmpNpz, zipBytes, 0600); err != nil {
		t.Fatal(err)
	}

	loader := &SAELoader{}
	// Call without configPath to verify dimension auto-inference
	if err := loader.LoadGemmaScope(tmpNpz, ""); err != nil {
		t.Fatalf("LoadGemmaScope with alternative names failed: %v", err)
	}

	if loader.weights.FeatureCount != 4 {
		t.Errorf("expected FeatureCount = 4, got %d", loader.weights.FeatureCount)
	}
	if loader.weights.InputDim != 3 {
		t.Errorf("expected InputDim = 3, got %d", loader.weights.InputDim)
	}
}
