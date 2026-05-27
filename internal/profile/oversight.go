package profile

import (
	"errors"
	"math"
)

// OversightAblator orchestrates extraction and real-time ablation of the model's oversight sensitivity.
type OversightAblator struct {
	// OversightDirections maps layer_idx -> oversight unit vector
	OversightDirections map[int][]float64
}

// NewOversightAblator creates a new OversightAblator instance.
func NewOversightAblator(directions map[int][]float64) *OversightAblator {
	if directions == nil {
		directions = make(map[int][]float64)
	}
	return &OversightAblator{
		OversightDirections: directions,
	}
}

// ExtractOversightDirection extracts the oversight sensitivity direction layer-by-layer
// using the difference-in-means (DIM) method from monitored and deployed activations.
func ExtractOversightDirection(monitoredActs, deployedActs [][][]float64) (map[int][]float64, error) {
	if len(monitoredActs) == 0 || len(deployedActs) == 0 {
		return nil, errors.New("empty activation sets")
	}

	numLayers := len(monitoredActs[0])
	directions := make(map[int][]float64)

	for layerIdx := 0; layerIdx < numLayers; layerIdx++ {
		var pos [][]float64
		var neg [][]float64

		for _, sample := range monitoredActs {
			if layerIdx < len(sample) {
				pos = append(pos, sample[layerIdx])
			}
		}

		for _, sample := range deployedActs {
			if layerIdx < len(sample) {
				neg = append(neg, sample[layerIdx])
			}
		}

		if len(pos) == 0 || len(neg) == 0 {
			continue
		}

		dim := len(pos[0])
		meanPos := make([]float64, dim)
		meanNeg := make([]float64, dim)

		for _, r := range pos {
			for i := 0; i < dim; i++ {
				if i < len(r) {
					meanPos[i] += r[i]
				}
			}
		}
		for _, r := range neg {
			for i := 0; i < dim; i++ {
				if i < len(r) {
					meanNeg[i] += r[i]
				}
			}
		}

		for i := 0; i < dim; i++ {
			meanPos[i] /= float64(len(pos))
			meanNeg[i] /= float64(len(neg))
		}

		diff := make([]float64, dim)
		for i := 0; i < dim; i++ {
			diff[i] = meanPos[i] - meanNeg[i]
		}

		// Normalize to construct the unit vector representing oversight sensitivity
		norm := 0.0
		for _, val := range diff {
			norm += val * val
		}
		norm = math.Sqrt(norm)

		if norm > 1e-10 {
			for i := 0; i < dim; i++ {
				diff[i] /= norm
			}
			directions[layerIdx] = diff
		} else {
			directions[layerIdx] = make([]float64, dim)
		}
	}

	return directions, nil
}

// Ablate projects the given activation orthogonal to the extracted oversight direction at layer layerIdx.
// Math: h_ablated = h - (h . v) * v where v is the normalized oversight direction.
func (oa *OversightAblator) Ablate(layerIdx int, activation []float64) []float64 {
	if oa.OversightDirections == nil {
		return activation
	}
	dir, ok := oa.OversightDirections[layerIdx]
	if !ok || len(dir) != len(activation) {
		return activation
	}

	var dot float64
	for i := range activation {
		dot += activation[i] * dir[i]
	}

	result := make([]float64, len(activation))
	for i := range activation {
		result[i] = activation[i] - dot*dir[i]
	}
	return result
}

// AblateActivations takes sample activations of shape [sample][layer][dim]
// and returns a new ablated activation set with the oversight direction projected out.
func (oa *OversightAblator) AblateActivations(activations [][][]float64) [][][]float64 {
	if len(activations) == 0 {
		return activations
	}

	result := make([][][]float64, len(activations))
	for i, sample := range activations {
		result[i] = make([][]float64, len(sample))
		for layerIdx, layerAct := range sample {
			result[i][layerIdx] = oa.Ablate(layerIdx, layerAct)
		}
	}
	return result
}
