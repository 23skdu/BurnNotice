package analysis

import (
	"math"
	"sort"
)

// CrossLayerAlignmentAnalyzer extracts directions per layer and analyzes
// direction similarity, clusters, and decision-layer identification.
type CrossLayerAlignmentAnalyzer struct {
	layerDirections map[int][]float64
}

// NewCrossLayerAlignmentAnalyzer creates an analyzer for cross-layer direction alignment.
func NewCrossLayerAlignmentAnalyzer() *CrossLayerAlignmentAnalyzer {
	return &CrossLayerAlignmentAnalyzer{
		layerDirections: make(map[int][]float64),
	}
}

// AddLayerDirection registers a direction vector for a specific layer.
func (c *CrossLayerAlignmentAnalyzer) AddLayerDirection(layerIdx int, direction []float64) {
	c.layerDirections[layerIdx] = direction
}

// PairwiseSimilarityMatrix returns a matrix of cosine similarities between all layer pairs.
func (c *CrossLayerAlignmentAnalyzer) PairwiseSimilarityMatrix() [][]float64 {
	layers := sortedLayers(c.layerDirections)
	n := len(layers)
	mat := make([][]float64, n)
	for i := 0; i < n; i++ {
		mat[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			mat[i][j] = cosineSimilarity(c.layerDirections[layers[i]], c.layerDirections[layers[j]])
		}
	}
	return mat
}

// DecisionLayer identifies the layer where the direction first becomes discriminative.
// This is the first layer where cosine similarity to the mean of all subsequent layers
// exceeds a threshold (default 0.8).
func (c *CrossLayerAlignmentAnalyzer) DecisionLayer(threshold float64) (int, bool) {
	layers := sortedLayers(c.layerDirections)
	if len(layers) < 3 {
		return 0, false
	}
	if threshold <= 0 {
		threshold = 0.8
	}

	mean := make([]float64, len(c.layerDirections[layers[0]]))
	if len(mean) == 0 {
		return layers[len(layers)-1], false
	}
	for _, l := range layers {
		dir := c.layerDirections[l]
		for j := 0; j < len(dir) && j < len(mean); j++ {
			mean[j] += dir[j]
		}
	}
	inv := 1.0 / float64(len(layers))
	for j := range mean {
		mean[j] *= inv
	}
	mean = normalize(mean)

	for _, l := range layers {
		sim := cosineSimilarity(c.layerDirections[l], mean)
		if sim >= threshold {
			return l, true
		}
	}

	return layers[len(layers)-1], false
}

// DirectionClusters groups layers by direction similarity using a simple
// threshold-based clustering approach. Returns a map of cluster ID to layer indices.
func (c *CrossLayerAlignmentAnalyzer) DirectionClusters(similarityThreshold float64) map[int][]int {
	layers := sortedLayers(c.layerDirections)
	n := len(layers)
	if n == 0 {
		return nil
	}
	if similarityThreshold <= 0 {
		similarityThreshold = 0.85
	}

	clusters := make(map[int][]int)
	assigned := make([]bool, n)
	clusterID := 0

	for i := 0; i < n; i++ {
		if assigned[i] {
			continue
		}
		clusterID++
		clusters[clusterID] = append(clusters[clusterID], layers[i])
		assigned[i] = true

		for j := i + 1; j < n; j++ {
			if assigned[j] {
				continue
			}
			sim := cosineSimilarity(c.layerDirections[layers[i]], c.layerDirections[layers[j]])
			if sim >= similarityThreshold {
				clusters[clusterID] = append(clusters[clusterID], layers[j])
				assigned[j] = true
			}
		}
	}

	for id := range clusters {
		sort.Ints(clusters[id])
	}
	return clusters
}

// CKA computes Centered Kernel Alignment between two sets of layer activations.
// Returns a value in [0,1] where 1 = identical representations.
func CKA(activationsA, activationsB [][]float64) float64 {
	if len(activationsA) != len(activationsB) || len(activationsA) == 0 {
		return 0
	}
	n := len(activationsA)
	if len(activationsA[0]) == 0 || len(activationsB[0]) == 0 {
		return 0
	}

	center := func(X [][]float64) ([][]float64, bool) {
		cols := len(X[0])
		if cols == 0 {
			return nil, false
		}
		colMeans := make([]float64, cols)
		for i := 0; i < n; i++ {
			if len(X[i]) != cols || hasNonFinite(X[i]) {
				return nil, false
			}
			for j := 0; j < cols; j++ {
				colMeans[j] += X[i][j]
			}
		}
		inv := 1.0 / float64(n)
		for j := range colMeans {
			colMeans[j] *= inv
		}
		centered := make([][]float64, n)
		for i := 0; i < n; i++ {
			centered[i] = make([]float64, cols)
			for j := 0; j < cols; j++ {
				centered[i][j] = X[i][j] - colMeans[j]
			}
		}
		return centered, true
	}

	centeredA, okA := center(activationsA)
	centeredB, okB := center(activationsB)
	if !okA || !okB {
		return 0
	}

	ka := make([][]float64, n)
	kb := make([][]float64, n)
	for i := 0; i < n; i++ {
		ka[i] = make([]float64, n)
		kb[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			for k := 0; k < len(centeredA[i]); k++ {
				ka[i][j] += centeredA[i][k] * centeredA[j][k]
			}
			for k := 0; k < len(centeredB[i]); k++ {
				kb[i][j] += centeredB[i][k] * centeredB[j][k]
			}
		}
	}

	var hsicXY, hsicXX, hsicYY float64
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			hsicXY += ka[i][j] * kb[i][j]
			hsicXX += ka[i][j] * ka[i][j]
			hsicYY += kb[i][j] * kb[i][j]
		}
	}

	if hsicXX <= 0 || hsicYY <= 0 || math.IsNaN(hsicXX) || math.IsNaN(hsicYY) || math.IsInf(hsicXX, 0) || math.IsInf(hsicYY, 0) {
		return 0
	}
	val := hsicXY / math.Sqrt(hsicXX*hsicYY)
	if math.IsNaN(val) || math.IsInf(val, 0) || val < 0 {
		return 0
	}
	if val > 1.0 {
		return 1.0
	}
	return val
}
