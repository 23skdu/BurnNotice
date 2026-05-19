package sae

import (
	"encoding/json"
	"fmt"
	"math"
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
