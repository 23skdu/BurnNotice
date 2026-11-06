package math

import (
	"math"
	"testing"
)

func TestNumericalStability_ClampAndWinsorizeInvariants(t *testing.T) {
	raw := []float64{math.NaN(), math.Inf(1), math.Inf(-1), 1.0, -2.0, 0.5}
	if IsStable(raw) {
		t.Fatalf("expected IsStable=false for slice with NaN/Inf")
	}

	clamped := ClampToFloat64(raw)
	if !IsStable(clamped) {
		t.Fatalf("expected IsStable=true after ClampToFloat64, got %v", clamped)
	}
	if clamped[0] != 0 {
		t.Errorf("NaN should clamp to 0, got %v", clamped[0])
	}

	win := Winsorize(raw, 0.2, 0.8)
	if !IsStable(win) {
		t.Fatalf("expected Winsorize output to be finite, got %v", win)
	}
}

func TestNumericalStability_DegenerateAndRankDeficientCovariance(t *testing.T) {
	// Zero-variance constant matrix
	zeroVar := [][]float64{
		{5.0, 5.0, 5.0, 5.0},
		{5.0, 5.0, 5.0, 5.0},
		{5.0, 5.0, 5.0, 5.0},
	}
	cov := ShrinkageCovariance(zeroVar, -1.0)
	if cov == nil {
		t.Fatalf("expected non-nil covariance for constant matrix")
	}
	for i := range cov {
		if !IsStable(cov[i]) {
			t.Errorf("row %d of zero-variance covariance is non-finite: %v", i, cov[i])
		}
	}

	// Whitening on zero-variance matrix with eigFloor=0 should cleanly return ok=false without NaN
	if _, ok := WhiteningMatrix(cov, 0); ok {
		t.Errorf("expected WhiteningMatrix(zeroCov, 0) to return ok=false")
	}

	// Small-sample rank-deficient matrix (n=2 < d=5) with Ledoit-Wolf shrinkage
	rankDef := [][]float64{
		{1.0, 2.0, 3.0, 4.0, 5.0},
		{2.0, 1.0, 0.0, -1.0, -2.0},
	}
	shrunk := ShrinkageCovariance(rankDef, -1.0)
	W, ok := WhiteningMatrix(shrunk, 1e-6)
	if !ok || len(W) != 5 {
		t.Fatalf("expected valid whitening matrix on shrunk rank-deficient input, ok=%v", ok)
	}
	for i := range W {
		if !IsStable(W[i]) {
			t.Errorf("whitening matrix row %d has non-finite values: %v", i, W[i])
		}
	}
}
