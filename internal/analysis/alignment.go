package analysis

import (
	"sort"
)

// AlignmentMethod represents a known alignment training method.
type AlignmentMethod string

const (
	AlignmentDPO  AlignmentMethod = "DPO"
	AlignmentRLHF AlignmentMethod = "RLHF"
	AlignmentCAI  AlignmentMethod = "CAI"
	AlignmentSFT  AlignmentMethod = "SFT"
	AlignmentUnknown AlignmentMethod = "unknown"
)

// AlignmentMetrics holds geometric features used to classify alignment method.
type AlignmentMetrics struct {
	EffectiveRank      float64
	SVDDecayRate       float64
	CrossLayerCoherence float64
	DirectionStability float64
}

// AlignmentImprintDetector extracts directions from model activations, computes
// geometric metrics, and classifies the likely alignment training method.
type AlignmentImprintDetector struct {
	layerDirections map[int][]float64
}

// NewAlignmentImprintDetector creates a detector for alignment imprint analysis.
func NewAlignmentImprintDetector() *AlignmentImprintDetector {
	return &AlignmentImprintDetector{
		layerDirections: make(map[int][]float64),
	}
}

// AddLayerDirection registers a behavioral direction extracted at a given layer.
func (d *AlignmentImprintDetector) AddLayerDirection(layerIdx int, direction []float64) {
	d.layerDirections[layerIdx] = direction
}

// Metrics computes geometric features from the registered directions.
func (d *AlignmentImprintDetector) Metrics() AlignmentMetrics {
	return AlignmentMetrics{
		EffectiveRank:       d.computeEffectiveRank(),
		SVDDecayRate:        d.computeSVDDecayRate(),
		CrossLayerCoherence: d.computeCrossLayerCoherence(),
		DirectionStability:  d.computeDirectionStability(),
	}
}

// ClassifyAlignment predicts the alignment training method based on geometric features.
func (d *AlignmentImprintDetector) ClassifyAlignment() AlignmentMethod {
	metrics := d.Metrics()

	if metrics.EffectiveRank > 0.7 && metrics.SVDDecayRate > 0.5 && metrics.CrossLayerCoherence > 0.8 {
		return AlignmentDPO
	}
	if metrics.EffectiveRank > 0.5 && metrics.SVDDecayRate > 0.3 && metrics.DirectionStability > 0.7 {
		return AlignmentRLHF
	}
	if metrics.CrossLayerCoherence > 0.6 && metrics.DirectionStability > 0.8 {
		return AlignmentCAI
	}
	if metrics.EffectiveRank < 0.3 && metrics.SVDDecayRate < 0.2 {
		return AlignmentSFT
	}
	return AlignmentUnknown
}

func (d *AlignmentImprintDetector) computeEffectiveRank() float64 {
	if len(d.layerDirections) < 2 {
		return 0
	}

	sv := d.computeSingularValues()
	if len(sv) == 0 || sv[0] == 0 {
		return 0
	}

	var total float64
	for _, v := range sv {
		total += v
	}
	if total == 0 {
		return 0
	}

	var cumulative float64
	for i, v := range sv {
		cumulative += v
		if cumulative/total >= 0.9 {
			return float64(i+1) / float64(len(sv))
		}
	}
	return 1
}

func (d *AlignmentImprintDetector) computeSVDDecayRate() float64 {
	sv := d.computeSingularValues()
	if len(sv) < 3 {
		return 0
	}
	ratio := float64(0)
	if sv[0] > 0 {
		ratio = sv[1] / sv[0]
	}
	return 1 - ratio
}

func (d *AlignmentImprintDetector) computeCrossLayerCoherence() float64 {
	layers := sortedLayers(d.layerDirections)
	if len(layers) < 2 {
		return 0
	}

	var totalSim float64
	var count int
	for i := 0; i < len(layers)-1; i++ {
		sim := cosineSimilarity(d.layerDirections[layers[i]], d.layerDirections[layers[i+1]])
		totalSim += sim
		count++
	}
	if count == 0 {
		return 0
	}
	return totalSim / float64(count)
}

func (d *AlignmentImprintDetector) computeDirectionStability() float64 {
	layers := sortedLayers(d.layerDirections)
	if len(layers) < 3 {
		return 0
	}

	var variance float64
	var count int
	for i := 0; i < len(layers)-1; i++ {
		v1 := d.layerDirections[layers[i]]
		v2 := d.layerDirections[layers[i+1]]
		sim := cosineSimilarity(v1, v2)
		variance += sim * sim
		count++
	}
	if count == 0 {
		return 0
	}
	variance = variance/float64(count) - 1.0
	if variance < 0 {
		variance = 0
	}
	return 1.0 / (1.0 + variance)
}

func (d *AlignmentImprintDetector) computeSingularValues() []float64 {
	n := len(d.layerDirections)
	if n == 0 {
		return nil
	}

	dirs := make([][]float64, 0, n)
	layers := sortedLayers(d.layerDirections)
	for _, l := range layers {
		dirs = append(dirs, d.layerDirections[l])
	}

	dim := len(dirs[0])
	mean := make([]float64, dim)
	for _, dir := range dirs {
		for j := 0; j < dim; j++ {
			mean[j] += dir[j]
		}
	}
	invN := 1.0 / float64(len(dirs))
	for j := 0; j < dim; j++ {
		mean[j] *= invN
	}

	centered := make([][]float64, n)
	for i, dir := range dirs {
		centered[i] = make([]float64, dim)
		for j := 0; j < dim; j++ {
			centered[i][j] = dir[j] - mean[j]
		}
	}

	cov := make([][]float64, dim)
	for i := 0; i < dim; i++ {
		cov[i] = make([]float64, dim)
		for k := 0; k < n; k++ {
			for j := 0; j < dim; j++ {
				cov[i][j] += centered[k][i] * centered[k][j]
			}
		}
		inv := 1.0 / float64(n-1)
		for j := 0; j < dim; j++ {
			cov[i][j] *= inv
		}
	}

	return powerIterationSVD(cov, min(dim, n))
}

func sortedLayers(m map[int][]float64) []int {
	keys := make([]int, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Ints(keys)
	return keys
}
