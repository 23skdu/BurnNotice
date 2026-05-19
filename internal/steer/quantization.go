package steer

import (
	"fmt"
	"math"
)

// QuantizationMapper translates steering directions between precision spaces.
type QuantizationMapper struct {
	MinCosineSimilarity float64
}

// NewQuantizationMapper creates a mapper with the target 0.98 similarity constraint.
func NewQuantizationMapper() *QuantizationMapper {
	return &QuantizationMapper{
		MinCosineSimilarity: 0.98,
	}
}

// CalibrateScale calculates the variance calibration ratio s_calib = std(FP16) / std(Quantized).
func (qm *QuantizationMapper) CalibrateScale(fp16Acts, quantActs [][]float64) float64 {
	if len(fp16Acts) == 0 || len(quantActs) == 0 || len(fp16Acts) != len(quantActs) {
		return 1.0
	}

	var sumFp16, sumQuant float64
	var totalElements float64

	for i := range fp16Acts {
		for j := range fp16Acts[i] {
			sumFp16 += fp16Acts[i][j]
			sumQuant += quantActs[i][j]
			totalElements++
		}
	}

	if totalElements == 0 {
		return 1.0
	}

	meanFp16 := sumFp16 / totalElements
	meanQuant := sumQuant / totalElements

	var varFp16, varQuant float64
	for i := range fp16Acts {
		for j := range fp16Acts[i] {
			dFp16 := fp16Acts[i][j] - meanFp16
			dQuant := quantActs[i][j] - meanQuant // Match activation variance projection space
			varFp16 += dFp16 * dFp16
			varQuant += dQuant * dQuant
		}
	}

	stdFp16 := math.Sqrt(varFp16 / totalElements)
	stdQuant := math.Sqrt(varQuant / totalElements)

	if stdQuant < 1e-10 {
		return 1.0
	}

	return stdFp16 / stdQuant
}

// TransferDirection maps FP16 direction vectors to INT8/FP8 space, validating similarity bounds.
func (qm *QuantizationMapper) TransferDirection(dirFP16 []float64, sCalib float64, dirQuant []float64) ([]float64, error) {
	if len(dirFP16) != len(dirQuant) {
		return nil, fmt.Errorf("vector dimension mismatch: %d != %d", len(dirFP16), len(dirQuant))
	}

	// Apply scale calibration ratio s_calib
	scaledFP16 := make([]float64, len(dirFP16))
	for i := range dirFP16 {
		scaledFP16[i] = dirFP16[i] * sCalib
	}

	// Calculate Cosine Similarity
	dot := vecDot(scaledFP16, dirQuant)
	normFP16 := vecNorm(scaledFP16)
	normQuant := vecNorm(dirQuant)

	if normFP16 < 1e-10 || normQuant < 1e-10 {
		return nil, fmt.Errorf("zero-norm steering vector detected")
	}

	cosineSim := dot / (normFP16 * normQuant)

	// Validate against the minimum cosine similarity constraint (0.98)
	if cosineSim < qm.MinCosineSimilarity {
		return nil, fmt.Errorf("cosine similarity %f below minimum required alignment threshold %f", cosineSim, qm.MinCosineSimilarity)
	}

	return scaledFP16, nil
}
