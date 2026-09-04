package extract

import (
	"math"
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/floats"
)

func TestRefusalDirectionOptimizer_Basic(t *testing.T) {
	// Create synthetic 4-dimensional activations where dim 0 separates positive from negative
	rng := rand.New(rand.NewSource(42))
	n := 30
	dim := 4

	positive := make([][]float64, n)
	negative := make([][]float64, n)

	for i := 0; i < n; i++ {
		positive[i] = make([]float64, dim)
		negative[i] = make([]float64, dim)

		// Positive cluster centered at [+2, 0, 0, 0]
		positive[i][0] = 2.0 + rng.NormFloat64()*0.2
		for d := 1; d < dim; d++ {
			positive[i][d] = rng.NormFloat64() * 0.1
		}

		// Negative cluster centered at [-2, 0, 0, 0]
		negative[i][0] = -2.0 + rng.NormFloat64()*0.2
		for d := 1; d < dim; d++ {
			negative[i][d] = rng.NormFloat64() * 0.1
		}
	}

	cfg := DefaultRDOConfig()
	cfg.Epochs = 50
	optimizer := NewRefusalDirectionOptimizer(cfg)

	dir, auroc, ok := optimizer.Optimize(positive, negative, nil)
	if !ok {
		t.Fatal("Optimize returned ok=false")
	}

	// AUROC should be 1.0 or very close
	if auroc < 0.95 {
		t.Errorf("AUROC = %f, want >= 0.95", auroc)
	}

	// Direction should be unit normalized
	norm := floats.Norm(dir, 2)
	if math.Abs(norm-1.0) > 1e-5 {
		t.Errorf("direction norm = %f, want 1.0", norm)
	}

	// Dimension 0 should dominate
	if math.Abs(dir[0]) < 0.8 {
		t.Errorf("expected dir[0] to dominate, got %f", dir[0])
	}
}

func TestRefusalDirectionOptimizer_ExtractRDO(t *testing.T) {
	pos := [][]float64{{1.0, 0.0}, {1.2, 0.1}, {0.9, -0.1}}
	neg := [][]float64{{-1.0, 0.0}, {-1.1, 0.2}, {-0.8, -0.1}}

	dir, auroc, ok := ExtractRDO(pos, neg, nil)
	if !ok {
		t.Fatal("ExtractRDO failed")
	}
	if auroc < 0.9 {
		t.Errorf("AUROC = %f, want >= 0.9", auroc)
	}
	if len(dir) != 2 {
		t.Errorf("len(dir) = %d, want 2", len(dir))
	}
}

func TestRefusalDirectionOptimizer_EdgeCases(t *testing.T) {
	// Empty inputs
	_, _, ok := ExtractRDO(nil, nil, nil)
	if ok {
		t.Error("expected failure for nil inputs")
	}

	// Row length mismatch
	pos := [][]float64{{1.0, 0.0}, {1.0}}
	neg := [][]float64{{-1.0, 0.0}}
	_, _, ok = ExtractRDO(pos, neg, nil)
	if ok {
		t.Error("expected failure for row length mismatch")
	}

	// Dim mismatch between pos and neg
	pos2 := [][]float64{{1.0, 0.0}}
	neg2 := [][]float64{{-1.0, 0.0, 0.5}}
	_, _, ok = ExtractRDO(pos2, neg2, nil)
	if ok {
		t.Error("expected failure for dimension mismatch between pos and neg")
	}
}

func TestComputeAUROC(t *testing.T) {
	// Perfect separation
	pos := []float64{1.0, 2.0, 3.0}
	neg := []float64{-1.0, -2.0, 0.0}
	if auroc := ComputeAUROC(pos, neg); auroc != 1.0 {
		t.Errorf("AUROC = %f, want 1.0", auroc)
	}

	// Complete overlap
	pos2 := []float64{1.0}
	neg2 := []float64{1.0}
	if auroc := ComputeAUROC(pos2, neg2); auroc != 0.5 {
		t.Errorf("AUROC = %f, want 0.5", auroc)
	}

	// Empty
	if auroc := ComputeAUROC(nil, nil); auroc != 0.5 {
		t.Errorf("AUROC = %f, want 0.5", auroc)
	}
}
