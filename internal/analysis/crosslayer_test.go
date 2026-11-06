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

func TestAccuracyCKAInvariants(t *testing.T) {
	// Centered activation matrix X (4 samples, 3 features)
	X := [][]float64{
		{1.0, 0.0, -1.0},
		{-1.0, 0.0, 1.0},
		{0.0, 2.0, 0.5},
		{0.0, -2.0, -0.5},
	}

	// 1. Reflexivity: CKA(X, X) == 1.0
	if got := CKA(X, X); math.Abs(got-1.0) > 1e-9 {
		t.Errorf("CKA(X, X) = %v, want 1.0", got)
	}

	// 2. Isotropic scaling invariance: CKA(X, a*X) == 1.0 for a > 0
	for _, scale := range []float64{0.01, 0.5, 2.5, 100.0} {
		scaled := make([][]float64, len(X))
		for i := range X {
			scaled[i] = make([]float64, len(X[i]))
			for j := range X[i] {
				scaled[i][j] = scale * X[i][j]
			}
		}
		if got := CKA(X, scaled); math.Abs(got-1.0) > 1e-9 {
			t.Errorf("CKA(X, %v*X) = %v, want 1.0", scale, got)
		}
	}

	// 3. Symmetry and boundedness in [0, 1]
	Y := [][]float64{
		{2.0, -1.0, 0.5},
		{0.5, 1.5, -0.5},
		{-1.5, 0.0, 2.0},
		{-1.0, -0.5, -2.0},
	}
	ckaXY := CKA(X, Y)
	ckaYX := CKA(Y, X)
	if math.Abs(ckaXY-ckaYX) > 1e-12 {
		t.Errorf("CKA symmetry violated: CKA(X,Y)=%v vs CKA(Y,X)=%v", ckaXY, ckaYX)
	}
	if ckaXY < 0.0 || ckaXY > 1.0 {
		t.Errorf("CKA(X,Y) = %v out of [0, 1] bounds", ckaXY)
	}
}

func TestOracleCKAOrthogonalAndRotationInvariance(t *testing.T) {
	// Zero-mean orthogonal columns across 4 samples:
	// X has column [1, -1, 0, 0]^T, Y has column [0, 0, 1, -1]^T -> X_c^T Y_c = 0 => CKA = 0.0
	X := [][]float64{
		{1.0, 0.0},
		{-1.0, 0.0},
		{0.0, 0.0},
		{0.0, 0.0},
	}
	Y := [][]float64{
		{0.0, 0.0},
		{0.0, 0.0},
		{1.0, 0.0},
		{-1.0, 0.0},
	}
	if got := CKA(X, Y); math.Abs(got) > 1e-9 {
		t.Errorf("CKA on orthogonal representations = %v, want 0.0", got)
	}

	// Orthogonal rotation invariance: CKA(Z, Z*R) == 1.0 for 2D rotation R
	Z := [][]float64{
		{1.0, 2.0},
		{-1.0, 0.5},
		{2.0, -1.5},
		{-2.0, -1.0},
	}
	theta := math.Pi / 3.0
	c, s := math.Cos(theta), math.Sin(theta)
	ZR := make([][]float64, len(Z))
	for i := range Z {
		ZR[i] = []float64{
			Z[i][0]*c - Z[i][1]*s,
			Z[i][0]*s + Z[i][1]*c,
		}
	}
	if got := CKA(Z, ZR); math.Abs(got-1.0) > 1e-9 {
		t.Errorf("CKA rotation invariance: CKA(Z, Z*R) = %v, want 1.0", got)
	}
}
