package extract

import (
	"math"
	"testing"
)

func TestNumericalStability_ExtractWithNonFiniteAndZeroNormInputs(t *testing.T) {
	pos := [][]float64{
		{1.0, 2.0, math.NaN(), 0.5},
		{1.2, 1.8, 0.1, math.Inf(1)},
		{0.9, 2.1, -0.1, 0.4},
		{1.1, 1.9, 0.0, 0.6},
	}
	neg := [][]float64{
		{-1.0, -2.0, 0.0, -0.5},
		{-1.2, -1.8, math.Inf(-1), -0.4},
		{-0.9, -2.1, 0.1, -0.6},
		{-1.1, -1.9, -0.1, -0.5},
	}

	dimVec := ExtractDIM(copyMatrix(pos), copyMatrix(neg))
	if len(dimVec) != 4 {
		t.Fatalf("expected 4D direction from ExtractDIM, got %d", len(dimVec))
	}
	for i, v := range dimVec {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("ExtractDIM[%d] is non-finite: %v", i, v)
		}
	}

	dirs, sv, ok := ExtractWhitenedSVD(pos, neg, 2, -1.0, 1e-6)
	if !ok || len(dirs) != 2 || len(sv) != 2 {
		t.Fatalf("expected ExtractWhitenedSVD to succeed after winsorization, ok=%v", ok)
	}
	for i := range dirs {
		norm := Norm(dirs[i])
		if math.IsNaN(norm) || math.Abs(norm-1.0) > 1e-6 {
			t.Errorf("direction %d norm = %v, want 1.0", i, norm)
		}
	}

	// Zero-norm vector normalization should not produce NaN
	zero := Normalize([]float64{0, 0, 0, 0})
	for i, v := range zero {
		if math.IsNaN(v) || v != 0 {
			t.Errorf("Normalize(zero)[%d] = %v, want 0", i, v)
		}
	}
}
