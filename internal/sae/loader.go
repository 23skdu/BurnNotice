package sae

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"path/filepath"
	"sort"

	"os"
)

// SAEWeights holds sparse autoencoder weight matrices.
type SAEWeights struct {
	EncoderWeights  [][]float64 `json:"encoder_weights"`
	DecoderWeights  [][]float64 `json:"decoder_weights"`
	EncoderBias     []float64   `json:"encoder_bias"`
	DecoderBias     []float64   `json:"decoder_bias"`
	FeatureCount    int         `json:"feature_count"`
	InputDim        int         `json:"input_dim"`
	LayerIndex      int         `json:"layer_index"`
}

// SAELoader loads and manages SAE weights.
type SAELoader struct {
	weights *SAEWeights
}

// Load reads SAE weights from a JSON file.
func (l *SAELoader) Load(path string) error {
	// #nosec G304 -- CLI tool accepts user-provided SAE paths
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read SAE file: %w", err)
	}

	var w SAEWeights
	if err := json.Unmarshal(data, &w); err != nil {
		return fmt.Errorf("parse SAE file: %w", err)
	}

	if w.FeatureCount <= 0 {
		return fmt.Errorf("invalid feature count: %d", w.FeatureCount)
	}
	if w.InputDim <= 0 {
		return fmt.Errorf("invalid input dim: %d", w.InputDim)
	}

	l.weights = &w
	return nil
}

// LoadFromMemory loads SAE weights from an in-memory struct.
func (l *SAELoader) LoadFromMemory(w *SAEWeights) error {
	if w.FeatureCount <= 0 {
		return fmt.Errorf("invalid feature count: %d", w.FeatureCount)
	}
	if w.InputDim <= 0 {
		return fmt.Errorf("invalid input dim: %d", w.InputDim)
	}
	l.weights = w
	return nil
}

// Encode runs the SAE encoder: ReLU(W_enc * x + b_enc).
func (l *SAELoader) Encode(activation []float64) ([]float64, error) {
	if l.weights == nil {
		return nil, fmt.Errorf("SAE weights not loaded")
	}
	if len(activation) != l.weights.InputDim {
		return nil, fmt.Errorf("activation dim %d != input dim %d", len(activation), l.weights.InputDim)
	}

	features := make([]float64, l.weights.FeatureCount)
	for i := 0; i < l.weights.FeatureCount; i++ {
		var sum float64
		for j := 0; j < l.weights.InputDim; j++ {
			sum += l.weights.EncoderWeights[i][j] * activation[j]
		}
		sum += l.weights.EncoderBias[i]
		features[i] = math.Max(0, sum) // ReLU
	}
	return features, nil
}

// Decode runs the SAE decoder: W_dec * f + b_dec.
func (l *SAELoader) Decode(features []float64) ([]float64, error) {
	if l.weights == nil {
		return nil, fmt.Errorf("SAE weights not loaded")
	}
	if len(features) != l.weights.FeatureCount {
		return nil, fmt.Errorf("features dim %d != feature count %d", len(features), l.weights.FeatureCount)
	}

	reconstructed := make([]float64, l.weights.InputDim)
	for i := 0; i < l.weights.InputDim; i++ {
		var sum float64
		for j := 0; j < l.weights.FeatureCount; j++ {
			sum += l.weights.DecoderWeights[j][i] * features[j]
		}
		sum += l.weights.DecoderBias[i]
		reconstructed[i] = sum
	}
	return reconstructed, nil
}

// TopKFeatures returns the indices and values of the top-K active features.
func (l *SAELoader) TopKFeatures(features []float64, k int) ([]int, []float64) {
	type featureIdx struct {
		idx   int
		value float64
	}

	sorted := make([]featureIdx, len(features))
	for i, v := range features {
		sorted[i] = featureIdx{i, v}
	}
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].value > sorted[j].value
	})

	if k > len(sorted) {
		k = len(sorted)
	}

	indices := make([]int, k)
	values := make([]float64, k)
	for i := 0; i < k; i++ {
		indices[i] = sorted[i].idx
		values[i] = sorted[i].value
	}
	return indices, values
}

// Attribution returns a map of feature index to activation value for non-zero features.
func (l *SAELoader) Attribution(features []float64) map[int]float64 {
	result := make(map[int]float64)
	for i, v := range features {
		if v > 0 {
			result[i] = v
		}
	}
	return result
}

// GemmaScopeConfig holds metadata for Gemma Scope SAEs.
type GemmaScopeConfig struct {
	ModelID     string `json:"model_id"`
	LayerIndex  int    `json:"layer_index"`
	HiddenDim   int    `json:"hidden_dim"`
	FeatureDim  int    `json:"feature_dim"`
	L1Coefficient float64 `json:"l1_coefficient"`
}

// LoadGemmaScope loads SAE weights from Gemma Scope format (NPZ + config JSON).
// The npzPath points to a .npz file (ZIP of .npy arrays) containing:
//   - "encoder_weights" or "W_enc" (feature_dim × hidden_dim)
//   - "encoder_bias" or "b_enc" (feature_dim,)
//   - "decoder_weights" or "W_dec" (hidden_dim × feature_dim)
//   - "decoder_bias" or "b_dec" (hidden_dim,)
//
	// The optional configPath can point to a JSON file with Gemma Scope metadata.
func (l *SAELoader) LoadGemmaScope(npzPath, configPath string) error {
	// #nosec G304 -- CLI tool accepts user-provided SAE paths
	w := &SAEWeights{}

	if configPath != "" {
		// #nosec G304 -- CLI tool accepts user-provided SAE paths
		data, err := os.ReadFile(configPath)
		if err != nil {
			return fmt.Errorf("read gemma scope config: %w", err)
		}
		var cfg GemmaScopeConfig
		if err := json.Unmarshal(data, &cfg); err != nil {
			return fmt.Errorf("parse gemma scope config: %w", err)
		}
		w.FeatureCount = cfg.FeatureDim
		w.InputDim = cfg.HiddenDim
		w.LayerIndex = cfg.LayerIndex
	}

	// #nosec G304 -- CLI tool accepts user-provided NPZ paths
	npzData, err := os.ReadFile(npzPath)
	if err != nil {
		return fmt.Errorf("read npz file: %w", err)
	}

	reader, err := zip.NewReader(bytes.NewReader(npzData), int64(len(npzData)))
	if err != nil {
		return fmt.Errorf("open npz archive: %w", err)
	}

	npyFiles := make(map[string][]byte)
	for _, f := range reader.File {
		rc, err := f.Open()
		if err != nil {
			continue
		}
		data, err := io.ReadAll(rc)
		_ = rc.Close()
		if err != nil {
			continue
		}
		name := filepath.Base(f.Name)
		name = name[:len(name)-len(".npy")]
		npyFiles[name] = data
	}

	if encW, ok := npyFiles["encoder_weights"]; ok {
		w.EncoderWeights, err = parseNpyMatrix(encW)
	} else if encW, ok := npyFiles["W_enc"]; ok {
		w.EncoderWeights, err = parseNpyMatrix(encW)
	} else {
		return fmt.Errorf("npz missing encoder_weights or W_enc")
	}
	if err != nil {
		return fmt.Errorf("parse encoder_weights: %w", err)
	}

	if decW, ok := npyFiles["decoder_weights"]; ok {
		w.DecoderWeights, err = parseNpyMatrixTransposed(decW)
	} else if decW, ok := npyFiles["W_dec"]; ok {
		w.DecoderWeights, err = parseNpyMatrixTransposed(decW)
	} else {
		return fmt.Errorf("npz missing decoder_weights or W_dec")
	}
	if err != nil {
		return fmt.Errorf("parse decoder_weights: %w", err)
	}

	if encB, ok := npyFiles["encoder_bias"]; ok {
		w.EncoderBias, err = parseNpyVector(encB)
	} else if encB, ok := npyFiles["b_enc"]; ok {
		w.EncoderBias, err = parseNpyVector(encB)
	} else {
		return fmt.Errorf("npz missing encoder_bias or b_enc")
	}
	if err != nil {
		return fmt.Errorf("parse encoder_bias: %w", err)
	}

	if decB, ok := npyFiles["decoder_bias"]; ok {
		w.DecoderBias, err = parseNpyVector(decB)
	} else if decB, ok := npyFiles["b_dec"]; ok {
		w.DecoderBias, err = parseNpyVector(decB)
	} else {
		return fmt.Errorf("npz missing decoder_bias or b_dec")
	}
	if err != nil {
		return fmt.Errorf("parse decoder_bias: %w", err)
	}

	if w.FeatureCount == 0 {
		w.FeatureCount = len(w.EncoderWeights)
	}
	if w.InputDim == 0 && len(w.EncoderWeights) > 0 {
		w.InputDim = len(w.EncoderWeights[0])
	}

	l.weights = w
	return nil
}

// parseNpyMatrix reads a 2D float32 matrix from .npy format and returns it as [][]float64.
func parseNpyMatrix(data []byte) ([][]float64, error) {
	shape, _, err := readNpyHeader(data)
	if err != nil {
		return nil, err
	}
	if len(shape) != 2 {
		return nil, fmt.Errorf("expected 2D array, got %dD", len(shape))
	}
	rows, cols := shape[0], shape[1]
	headerSize := int64(findNpyDataOffset(data))
	floatData := data[headerSize:]

	mat := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		mat[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			offset := (i*cols + j) * 4
			if offset+4 > len(floatData) {
				return nil, fmt.Errorf("npy data truncated at (%d,%d)", i, j)
			}
			bits := binary.LittleEndian.Uint32(floatData[offset : offset+4])
			mat[i][j] = float64(math.Float32frombits(bits))
		}
	}
	return mat, nil
}

// parseNpyMatrixTransposed reads a 2D float32 matrix from .npy and returns
// it transposed as [][]float64 (feature_dim × hidden_dim → rows are features).
func parseNpyMatrixTransposed(data []byte) ([][]float64, error) {
	mat, err := parseNpyMatrix(data)
	if err != nil {
		return nil, err
	}
	if len(mat) == 0 {
		return nil, fmt.Errorf("empty matrix")
	}
	rows := len(mat[0])
	cols := len(mat)
	t := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		t[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			t[i][j] = mat[j][i]
		}
	}
	return t, nil
}

// parseNpyVector reads a 1D float32 vector from .npy format.
func parseNpyVector(data []byte) ([]float64, error) {
	shape, _, err := readNpyHeader(data)
	if err != nil {
		return nil, err
	}
	if len(shape) != 1 {
		return nil, fmt.Errorf("expected 1D array, got %dD", len(shape))
	}
	n := shape[0]
	headerSize := findNpyDataOffset(data)
	floatData := data[headerSize:]

	vec := make([]float64, n)
	for i := 0; i < n; i++ {
		offset := i * 4
		if offset+4 > len(floatData) {
			return nil, fmt.Errorf("npy data truncated at index %d", i)
		}
		bits := binary.LittleEndian.Uint32(floatData[offset : offset+4])
		vec[i] = float64(math.Float32frombits(bits))
	}
	return vec, nil
}

// readNpyHeader parses the .npy format header and returns (shape, fortranOrder, error).
func readNpyHeader(data []byte) ([]int, bool, error) {
	if len(data) < 10 {
		return nil, false, fmt.Errorf("npy data too short")
	}
	if string(data[:6]) != "\x93NUMPY" {
		return nil, false, fmt.Errorf("invalid npy magic bytes")
	}
	majorVer := int(data[6])
	_ = majorVer

	var headerLen int64
	if data[6] == 1 {
		if len(data) < 10 {
			return nil, false, fmt.Errorf("npy v1 header too short")
		}
		headerLen = int64(binary.LittleEndian.Uint16(data[8:10]))
	} else if data[6] == 2 {
		if len(data) < 12 {
			return nil, false, fmt.Errorf("npy v2 header too short")
		}
		headerLen = int64(binary.LittleEndian.Uint32(data[8:12]))
	} else {
		return nil, false, fmt.Errorf("unsupported npy version: %d", data[6])
	}

	headerEnd := 10 + int(headerLen)
	if data[6] == 2 {
		headerEnd = 12 + int(headerLen)
	}
	if headerEnd > len(data) {
		return nil, false, fmt.Errorf("npy header extends beyond data")
	}

	headerStr := string(data[10:headerEnd])
	if data[6] == 2 {
		headerStr = string(data[12:headerEnd])
	}

	jsonStr := npyHeaderToJSON(headerStr)
	var header struct {
		Descr        string `json:"descr"`
		FortranOrder bool   `json:"fortran_order"`
		Shape        []int  `json:"shape"`
	}
	if err := json.Unmarshal([]byte(jsonStr), &header); err != nil {
		return nil, false, fmt.Errorf("parse npy header (sanitized JSON: %s): %w", jsonStr, err)
	}

	if header.Descr != "<f4" && header.Descr != "float32" {
		return nil, false, fmt.Errorf("unsupported npy dtype: %s (want <f4)", header.Descr)
	}

	return header.Shape, header.FortranOrder, nil
}

func npyHeaderToJSON(s string) string {
	var buf bytes.Buffer
	inString := false
	var strChar rune

	runes := []rune(s)
	n := len(runes)
	for i := 0; i < n; i++ {
		r := runes[i]
		if inString {
			if r == strChar {
				buf.WriteRune('"')
				inString = false
			} else {
				if r == '"' {
					buf.WriteString("\\\"")
				} else {
					buf.WriteRune(r)
				}
			}
			continue
		}

		switch r {
		case '\'', '"':
			inString = true
			strChar = r
			buf.WriteRune('"')
		case '(':
			buf.WriteRune('[')
		case ')':
			buf.WriteRune(']')
		case 'T', 'F':
			if i+3 < n && string(runes[i:i+4]) == "True" {
				buf.WriteString("true")
				i += 3
			} else if i+4 < n && string(runes[i:i+5]) == "False" {
				buf.WriteString("false")
				i += 4
			} else {
				buf.WriteRune(r)
			}
		default:
			buf.WriteRune(r)
		}
	}

	return cleanTrailingCommas(buf.String())
}

func cleanTrailingCommas(s string) string {
	runes := []rune(s)
	n := len(runes)
	var out []rune
	for i := 0; i < n; i++ {
		if runes[i] == ',' {
			nextNonWS := -1
			for j := i + 1; j < n; j++ {
				r := runes[j]
				if r != ' ' && r != '\t' && r != '\r' && r != '\n' {
					nextNonWS = j
					break
				}
			}
			if nextNonWS != -1 && (runes[nextNonWS] == ']' || runes[nextNonWS] == '}') {
				continue
			}
		}
		out = append(out, runes[i])
	}
	return string(out)
}

func findNpyDataOffset(data []byte) int {
	if len(data) < 10 {
		return len(data)
	}
	if data[6] == 1 {
		return 10 + int(binary.LittleEndian.Uint16(data[8:10]))
	}
	return 12 + int(binary.LittleEndian.Uint32(data[8:12]))
}

// ReconstructionError computes L2 norm between activation and SAE reconstruction.
func (l *SAELoader) ReconstructionError(activation []float64) (float64, error) {
	features, err := l.Encode(activation)
	if err != nil {
		return 0, err
	}
	reconstructed, err := l.Decode(features)
	if err != nil {
		return 0, err
	}

	var sum float64
	for i := range activation {
		diff := activation[i] - reconstructed[i]
		sum += diff * diff
	}
	return math.Sqrt(sum), nil
}
