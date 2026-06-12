package analysis

import (
	"math"
	"testing"
)

func TestCrossLayerAlignmentAnalyzer_Basic(t *testing.T) {
	analyzer := NewCrossLayerAlignmentAnalyzer()

	// 1. Edge cases
	if _, ok := analyzer.DecisionLayer(0.8); ok {
		t.Error("expected ok=false for DecisionLayer with less than 3 layers")
	}
	if analyzer.DirectionClusters(0.8) != nil {
		t.Error("expected nil clusters for empty analyzer")
	}

	// 2. Add layer directions
	// Layer 0: [1, 0]
	// Layer 1: [0.99, 0.1]
	// Layer 2: [0.8, 0.6]
	analyzer.AddLayerDirection(0, []float64{1.0, 0.0})
	analyzer.AddLayerDirection(1, []float64{0.99, 0.1})
	analyzer.AddLayerDirection(2, []float64{0.8, 0.6})

	// Pairwise similarity matrix
	mat := analyzer.PairwiseSimilarityMatrix()
	if len(mat) != 3 || len(mat[0]) != 3 {
		t.Errorf("unexpected pairwise matrix dimensions: %dx%d", len(mat), len(mat[0]))
	}

	// Decision layer
	dl, ok := analyzer.DecisionLayer(0.9)
	if !ok {
		t.Error("expected DecisionLayer to return true")
	}
	// Layer 0 similarity to mean should be high enough or Layer 1
	_ = dl

	// Clusters
	clusters := analyzer.DirectionClusters(0.95)
	if len(clusters) == 0 {
		t.Error("expected non-empty clusters")
	}
}

func TestCKA(t *testing.T) {
	// Identical datasets
	actA := [][]float64{
		{1.0, 2.0},
		{3.0, 4.0},
		{5.0, 6.0},
	}
	actB := [][]float64{
		{1.0, 2.0},
		{3.0, 4.0},
		{5.0, 6.0},
	}

	ckaVal := CKA(actA, actB)
	if math.Abs(ckaVal-1.0) > 1e-5 {
		t.Errorf("CKA for identical activations = %f, want 1.0", ckaVal)
	}

	// Mismatched row count
	actC := [][]float64{
		{1.0, 2.0},
	}
	if CKA(actA, actC) != 0 {
		t.Error("expected CKA to return 0 for row count mismatch")
	}

	// Empty activations
	if CKA(nil, nil) != 0 {
		t.Error("expected CKA to return 0 for nil inputs")
	}
}
