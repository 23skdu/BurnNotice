package analysis

import (
	"math"
	"testing"
)

func TestNumericalStability_AnalysisEdgeCases(t *testing.T) {
	// 1. Extreme logits in LogSoftmax and ComputeKLDivergence should not overflow/NaN
	extremeP := []float64{1000.0, -1000.0, 999.5, 0.0}
	extremeQ := []float64{-500.0, 500.0, 499.0, 0.0}

	logP, err := LogSoftmax(extremeP)
	if err != nil {
		t.Fatalf("LogSoftmax failed on extreme logits: %v", err)
	}
	for i, v := range logP {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("LogSoftmax[%d] = %v is non-finite", i, v)
		}
	}

	kl, err := ComputeKLDivergence(extremeP, extremeQ)
	if err != nil || math.IsNaN(kl) || math.IsInf(kl, 0) || kl < 0 {
		t.Errorf("ComputeKLDivergence on extreme logits returned %v, err=%v", kl, err)
	}

	// 2. Zero-norm / degenerate CKA should return 0 instead of NaN
	zeroActs := [][]float64{
		{0, 0, 0},
		{0, 0, 0},
	}
	if cka := CKA(zeroActs, zeroActs); math.IsNaN(cka) || cka != 0 {
		t.Errorf("CKA(zero, zero) = %v, want 0", cka)
	}

	// 3. ProjectLatentSpace rejects NaN/Inf vectors cleanly
	badPts := []LatentPoint{
		{ID: "1", Vector: []float64{1.0, math.NaN(), 2.0}},
		{ID: "2", Vector: []float64{2.0, 3.0, 4.0}},
	}
	if _, err := ProjectLatentSpace(badPts, nil, nil); err == nil {
		t.Errorf("expected error for NaN vector in ProjectLatentSpace")
	}
}
