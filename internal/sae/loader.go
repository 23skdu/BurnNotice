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
	"strings"

	"os"
)

// SAEFamily identifies the architecture-specific Sparse Autoencoder dictionary format.
type SAEFamily string

const (
	SAEGemmaScope SAEFamily = "gemma-scope"
	SAELlamaScope SAEFamily = "llama-scope"
	SAEQwenSAE    SAEFamily = "qwen-sae"
	SAEGeneric    SAEFamily = "generic"
)

// SAEDictionary defines the common interface across Gemma Scope, Llama-Scope, and Qwen-SAE dictionaries.
type SAEDictionary interface {
	Family() SAEFamily
	Weights() *SAEWeights
	Encode(activation []float64) ([]float64, error)
	Decode(features []float64) ([]float64, error)
	TopKFeatures(features []float64, k int) ([]int, []float64)
	Attribution(features []float64) map[int]float64
	ReconstructionError(activation []float64) (float64, error)
}

// SAEWeights holds sparse autoencoder weight matrices.
type SAEWeights struct {
	EncoderWeights [][]float64 `json:"encoder_weights"`
	DecoderWeights [][]float64 `json:"decoder_weights"`
	EncoderBias    []float64   `json:"encoder_bias"`
	DecoderBias    []float64   `json:"decoder_bias"`
	PreBias        []float64   `json:"pre_bias,omitempty"`       // Optional pre-encoder centering bias (Llama-Scope / Qwen-SAE)
	JumpThreshold  []float64   `json:"jump_threshold,omitempty"` // Optional JumpReLU threshold per feature
	FeatureCount   int         `json:"feature_count"`
	InputDim       int         `json:"input_dim"`
	LayerIndex     int         `json:"layer_index"`
	Family         SAEFamily   `json:"family,omitempty"`
}

// SAELoader loads and manages SAE weights.
type SAELoader struct {
	weights *SAEWeights
}

// Family returns the detected or configured SAEFamily for the loaded dictionary.
func (l *SAELoader) Family() SAEFamily {
	if l.weights == nil || l.weights.Family == "" {
		return SAEGeneric
	}
	return l.weights.Family
}

// Weights returns the underlying SAEWeights.
func (l *SAELoader) Weights() *SAEWeights {
	return l.weights
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

// Encode runs the SAE encoder: ReLU(W_enc * (x - b_pre) + b_enc) with optional JumpReLU threshold.
func (l *SAELoader) Encode(activation []float64) ([]float64, error) {
	if l.weights == nil {
		return nil, fmt.Errorf("SAE weights not loaded")
	}
	if len(activation) != l.weights.InputDim {
		return nil, fmt.Errorf("activation dim %d != input dim %d", len(activation), l.weights.InputDim)
	}

	hasPreBias := len(l.weights.PreBias) == l.weights.InputDim
	hasJump := len(l.weights.JumpThreshold) == l.weights.FeatureCount

	features := make([]float64, l.weights.FeatureCount)
	for i := 0; i < l.weights.FeatureCount; i++ {
		var sum float64
		for j := 0; j < l.weights.InputDim; j++ {
			x := activation[j]
			if hasPreBias {
				x -= l.weights.PreBias[j]
			}
			sum += l.weights.EncoderWeights[i][j] * x
		}
		if i < len(l.weights.EncoderBias) {
			sum += l.weights.EncoderBias[i]
		}
		val := math.Max(0, sum) // ReLU
		if hasJump && val <= l.weights.JumpThreshold[i] {
			val = 0
		}
		features[i] = val
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

// GemmaScopeConfig holds metadata for Gemma Scope, Llama-Scope, and Qwen-SAE dictionaries.
type GemmaScopeConfig struct {
	ModelID       string  `json:"model_id"`
	SAEFamily     string  `json:"sae_family,omitempty"`
	Architecture  string  `json:"architecture,omitempty"`
	HookPoint     string  `json:"hook_point,omitempty"`
	LayerIndex    int     `json:"layer_index"`
	HiddenDim     int     `json:"hidden_dim"`
	FeatureDim    int     `json:"feature_dim"`
	L1Coefficient float64 `json:"l1_coefficient"`
}

// DetectSAEArchitecture detects the SAE dictionary family from model ID, config JSON, or tensor names.
func DetectSAEArchitecture(modelID string, configJSON []byte, tensorNames []string) SAEFamily {
	if len(configJSON) > 0 {
		var cfg GemmaScopeConfig
		if err := json.Unmarshal(configJSON, &cfg); err == nil {
			combined := strings.ToLower(cfg.SAEFamily + " " + cfg.ModelID + " " + cfg.Architecture + " " + cfg.HookPoint)
			if strings.Contains(combined, "llama") {
				return SAELlamaScope
			}
			if strings.Contains(combined, "qwen") || strings.Contains(combined, "qwq") {
				return SAEQwenSAE
			}
			if strings.Contains(combined, "gemma") {
				return SAEGemmaScope
			}
		}
	}

	idLower := strings.ToLower(modelID)
	if strings.Contains(idLower, "llama") {
		return SAELlamaScope
	}
	if strings.Contains(idLower, "qwen") || strings.Contains(idLower, "qwq") {
		return SAEQwenSAE
	}
	if strings.Contains(idLower, "gemma") {
		return SAEGemmaScope
	}

	for _, tName := range tensorNames {
		switch tName {
		case "encoder.weight", "decoder.weight", "b_pre":
			return SAELlamaScope
		case "w_enc", "w_dec", "pre_bias":
			return SAEQwenSAE
		case "W_enc", "W_dec", "threshold":
			return SAEGemmaScope
		}
	}

	return SAEGeneric
}

// MapSAEToModelFamily maps an SAEFamily to the canonical model family identifier.
func MapSAEToModelFamily(family SAEFamily) string {
	switch family {
	case SAEGemmaScope:
		return "gemma"
	case SAELlamaScope:
		return "llama"
	case SAEQwenSAE:
		return "qwen"
	default:
		return "unknown"
	}
}

// LoadGemmaScope loads SAE weights from Gemma Scope format (NPZ + config JSON).
func (l *SAELoader) LoadGemmaScope(npzPath, configPath string) error {
	return l.loadScopedDictionary(npzPath, configPath, SAEGemmaScope)
}

// LoadLlamaScope loads SAE weights from Llama-Scope format (NPZ or JSON + optional config JSON).
func (l *SAELoader) LoadLlamaScope(path, configPath string) error {
	return l.loadScopedDictionary(path, configPath, SAELlamaScope)
}

// LoadQwenSAE loads SAE weights from Qwen-SAE format (NPZ or JSON + optional config JSON).
func (l *SAELoader) LoadQwenSAE(path, configPath string) error {
	return l.loadScopedDictionary(path, configPath, SAEQwenSAE)
}

// LoadDictionary auto-detects the SAE dictionary family (Gemma Scope, Llama-Scope, Qwen-SAE, or generic JSON)
// and loads the weights into the SAELoader.
func (l *SAELoader) LoadDictionary(path, configPath string) error {
	var cfgBytes []byte
	if configPath != "" {
		// #nosec G304 -- CLI tool accepts user-provided SAE paths
		data, err := os.ReadFile(configPath)
		if err != nil {
			return fmt.Errorf("read sae config: %w", err)
		}
		cfgBytes = data
	}
	fam := DetectSAEArchitecture(filepath.Base(path), cfgBytes, nil)
	return l.loadScopedDictionary(path, configPath, fam)
}

func (l *SAELoader) loadScopedDictionary(path, configPath string, defaultFamily SAEFamily) error {
	w := &SAEWeights{Family: defaultFamily}
	var cfgBytes []byte

	if configPath != "" {
		// #nosec G304 -- CLI tool accepts user-provided SAE paths
		data, err := os.ReadFile(configPath)
		if err != nil {
			return fmt.Errorf("read sae config: %w", err)
		}
		cfgBytes = data
		var cfg GemmaScopeConfig
		if err := json.Unmarshal(data, &cfg); err != nil {
			return fmt.Errorf("parse sae config: %w", err)
		}
		w.FeatureCount = cfg.FeatureDim
		w.InputDim = cfg.HiddenDim
		w.LayerIndex = cfg.LayerIndex
	}

	if strings.HasSuffix(strings.ToLower(path), ".json") {
		if err := l.Load(path); err != nil {
			return err
		}
		if l.weights.Family == "" {
			detected := DetectSAEArchitecture(filepath.Base(path), cfgBytes, nil)
			if detected != SAEGeneric {
				l.weights.Family = detected
			} else {
				l.weights.Family = defaultFamily
			}
		}
		return nil
	}

	// #nosec G304 -- CLI tool accepts user-provided NPZ paths
	npzData, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read npz file: %w", err)
	}

	reader, err := zip.NewReader(bytes.NewReader(npzData), int64(len(npzData)))
	if err != nil {
		return fmt.Errorf("open npz archive: %w", err)
	}

	npyFiles := make(map[string][]byte)
	tensorNames := make([]string, 0, len(reader.File))
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
		name := strings.TrimSuffix(filepath.Base(f.Name), ".npy")
		npyFiles[name] = data
		tensorNames = append(tensorNames, name)
	}

	if defaultFamily == SAEGeneric {
		w.Family = DetectSAEArchitecture(filepath.Base(path), cfgBytes, tensorNames)
		if w.Family == SAEGeneric {
			w.Family = SAEGemmaScope
		}
	}

	encW, ok := firstPresentTensor(npyFiles, "encoder_weights", "W_enc", "encoder.weight", "w_enc")
	if !ok {
		return fmt.Errorf("npz missing encoder_weights, W_enc, encoder.weight, or w_enc")
	}
	w.EncoderWeights, err = parseNpyMatrix(encW)
	if err != nil {
		return fmt.Errorf("parse encoder_weights: %w", err)
	}

	decW, ok := firstPresentTensor(npyFiles, "decoder_weights", "W_dec", "decoder.weight", "w_dec")
	if !ok {
		return fmt.Errorf("npz missing decoder_weights, W_dec, decoder.weight, or w_dec")
	}
	w.DecoderWeights, err = parseNpyMatrixTransposed(decW)
	if err != nil {
		return fmt.Errorf("parse decoder_weights: %w", err)
	}

	encB, ok := firstPresentTensor(npyFiles, "encoder_bias", "b_enc", "encoder.bias")
	if !ok {
		return fmt.Errorf("npz missing encoder_bias, b_enc, or encoder.bias")
	}
	w.EncoderBias, err = parseNpyVector(encB)
	if err != nil {
		return fmt.Errorf("parse encoder_bias: %w", err)
	}

	decB, ok := firstPresentTensor(npyFiles, "decoder_bias", "b_dec", "decoder.bias")
	if !ok {
		return fmt.Errorf("npz missing decoder_bias, b_dec, or decoder.bias")
	}
	w.DecoderBias, err = parseNpyVector(decB)
	if err != nil {
		return fmt.Errorf("parse decoder_bias: %w", err)
	}

	if preB, hasPre := firstPresentTensor(npyFiles, "pre_bias", "b_pre"); hasPre {
		if parsedPre, pErr := parseNpyVector(preB); pErr == nil {
			w.PreBias = parsedPre
		}
	}
	if jumpT, hasJump := firstPresentTensor(npyFiles, "threshold", "jump_threshold"); hasJump {
		if parsedJump, jErr := parseNpyVector(jumpT); jErr == nil {
			w.JumpThreshold = parsedJump
		}
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

func firstPresentTensor(files map[string][]byte, keys ...string) ([]byte, bool) {
	for _, k := range keys {
		if data, ok := files[k]; ok {
			return data, true
		}
	}
	return nil, false
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
	var headerLen int64
	switch data[6] {
	case 1:
		if len(data) < 10 {
			return nil, false, fmt.Errorf("npy v1 header too short")
		}
		headerLen = int64(binary.LittleEndian.Uint16(data[8:10]))
	case 2:
		if len(data) < 12 {
			return nil, false, fmt.Errorf("npy v2 header too short")
		}
		headerLen = int64(binary.LittleEndian.Uint32(data[8:12]))
	default:
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
