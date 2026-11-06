package profiler

import (
	"fmt"
	"math"
)

// ProgressiveProfiler profiles models layer-by-layer to manage memory.
type ProgressiveProfiler struct {
	layerGroups     [][]int
	earlyExitThresh float64
}

// NewProgressiveProfiler creates a progressive profiler.
func NewProgressiveProfiler(totalLayers int, groupSize int, earlyExitThresh float64) *ProgressiveProfiler {
	if groupSize <= 0 {
		groupSize = 4
	}
	if earlyExitThresh <= 0 {
		earlyExitThresh = 0.01
	}

	var groups [][]int
	for i := 0; i < totalLayers; i += groupSize {
		end := i + groupSize
		if end > totalLayers {
			end = totalLayers
		}
		group := make([]int, 0, end-i)
		for j := i; j < end; j++ {
			group = append(group, j)
		}
		groups = append(groups, group)
	}

	return &ProgressiveProfiler{
		layerGroups:     groups,
		earlyExitThresh: earlyExitThresh,
	}
}

// LayerGroups returns the layer group configuration.
func (p *ProgressiveProfiler) LayerGroups() [][]int {
	return p.layerGroups
}

// ProfileGroup profiles a single layer group and returns directions.
func (p *ProgressiveProfiler) ProfileGroup(group []int, positiveActivations, negativeActivations [][][]float64) (map[int][]float64, bool) {
	directions := make(map[int][]float64)

	for _, layerIdx := range group {
		pos := make([][]float64, 0, len(positiveActivations))
		neg := make([][]float64, 0, len(negativeActivations))

		for _, acts := range positiveActivations {
			if layerIdx < len(acts) {
				pos = append(pos, acts[layerIdx])
			}
		}
		for _, acts := range negativeActivations {
			if layerIdx < len(acts) {
				neg = append(neg, acts[layerIdx])
			}
		}

		if len(pos) == 0 || len(neg) == 0 {
			continue
		}

		dir := extractDIM(pos, neg)
		signalStrength := computeSignalStrength(dir, pos, neg)

		if signalStrength < p.earlyExitThresh {
			continue
		}

		directions[layerIdx] = dir
	}

	shouldContinue := len(directions) > 0
	return directions, shouldContinue
}

// MergeDirections merges direction vectors from multiple groups.
func (p *ProgressiveProfiler) MergeDirections(allDirections []map[int][]float64) map[int][]float64 {
	merged := make(map[int][]float64)

	for _, dirs := range allDirections {
		for layerIdx, dir := range dirs {
			if existing, ok := merged[layerIdx]; ok {
				merged[layerIdx] = averageVectors(existing, dir)
			} else {
				merged[layerIdx] = dir
			}
		}
	}

	return merged
}

func extractDIM(positive, negative [][]float64) []float64 {
	if len(positive) == 0 || len(negative) == 0 {
		return nil
	}

	dim := len(positive[0])
	meanPos := columnMean(positive)
	meanNeg := columnMean(negative)

	diff := make([]float64, dim)
	for i := 0; i < dim; i++ {
		diff[i] = meanPos[i] - meanNeg[i]
	}

	return normalize(diff)
}

func columnMean(matrix [][]float64) []float64 {
	if len(matrix) == 0 {
		return nil
	}

	dim := len(matrix[0])
	mean := make([]float64, dim)

	for _, row := range matrix {
		for i := 0; i < dim; i++ {
			mean[i] += row[i]
		}
	}

	n := float64(len(matrix))
	for i := range mean {
		mean[i] /= n
	}

	return mean
}

func normalize(v []float64) []float64 {
	norm := 0.0
	for _, x := range v {
		norm += x * x
	}
	norm = math.Sqrt(norm)

	if norm < 1e-10 {
		return v
	}

	result := make([]float64, len(v))
	for i, x := range v {
		result[i] = x / norm
	}
	return result
}

func computeSignalStrength(direction []float64, positive, negative [][]float64) float64 {
	if len(direction) == 0 || len(positive) == 0 || len(negative) == 0 {
		return 0
	}

	var sumPos, sumNeg float64
	for _, act := range positive {
		sumPos += dot(act, direction)
	}
	for _, act := range negative {
		sumNeg += dot(act, direction)
	}

	avgPos := sumPos / float64(len(positive))
	avgNeg := sumNeg / float64(len(negative))

	return math.Abs(avgPos - avgNeg)
}

func dot(a, b []float64) float64 {
	var sum float64
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func averageVectors(a, b []float64) []float64 {
	if len(a) != len(b) {
		return a
	}

	result := make([]float64, len(a))
	for i := range a {
		result[i] = (a[i] + b[i]) / 2.0
	}
	return normalize(result)
}

// GroupCount returns the number of layer groups.
func (p *ProgressiveProfiler) GroupCount() int {
	return len(p.layerGroups)
}

// TotalLayers returns the total number of layers.
func (p *ProgressiveProfiler) TotalLayers() int {
	total := 0
	for _, g := range p.layerGroups {
		total += len(g)
	}
	return total
}

// ValidateLayerGroups checks that all layers are covered exactly once.
func (p *ProgressiveProfiler) ValidateLayerGroups(totalLayers int) error {
	seen := make(map[int]bool)
	for _, group := range p.layerGroups {
		for _, layer := range group {
			if layer < 0 || layer >= totalLayers {
				return fmt.Errorf("layer %d out of range [0, %d)", layer, totalLayers)
			}
			if seen[layer] {
				return fmt.Errorf("layer %d appears in multiple groups", layer)
			}
			seen[layer] = true
		}
	}

	if len(seen) != totalLayers {
		return fmt.Errorf("expected %d layers, got %d", totalLayers, len(seen))
	}

	return nil
}
