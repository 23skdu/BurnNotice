package analysis

import (
	"math"
	"testing"
)

func TestAccuracy_CKAAndCrossLayerInvariants(t *testing.T) {
	actA := [][]float64{
		{1.0, 2.0, 3.0, 4.0},
		{2.0, 4.0, 1.0, 0.0},
		{-1.0, 0.0, 2.0, 3.0},
		{0.5, -2.0, 1.5, -1.0},
	}

	// 1. Identical matrix CKA == 1.0
	ckaSelf := CKA(actA, actA)
	if math.Abs(ckaSelf-1.0) > 1e-9 {
		t.Errorf("CKA(A, A) = %.12f, want 1.0", ckaSelf)
	}

	// 2. Isotropic scaling invariance: CKA(A, c*A) == 1.0
	actScaled := make([][]float64, len(actA))
	for i := range actA {
		actScaled[i] = make([]float64, len(actA[i]))
		for j := range actA[i] {
			actScaled[i][j] = actA[i][j] * 3.5
		}
	}
	ckaScaled := CKA(actA, actScaled)
	if math.Abs(ckaScaled-1.0) > 1e-9 {
		t.Errorf("CKA(A, 3.5*A) = %.12f, want 1.0", ckaScaled)
	}

	// 3. Orthogonal centered representations yield CKA == 0.0
	// After double-centering (column mean + grand mean), patterns u=[1,-1,0,0] and v=[0,0,1,-1]
	// have orthogonal Gram matrices K_u and K_v (<K_u, K_v>_F == 0).
	orthoX := [][]float64{
		{1.0, -1.0},
		{-1.0, 1.0},
		{0.0, 0.0},
		{0.0, 0.0},
	}
	orthoY := [][]float64{
		{0.0, 0.0},
		{0.0, 0.0},
		{1.0, -1.0},
		{-1.0, 1.0},
	}
	ckaOrtho := CKA(orthoX, orthoY)
	if math.Abs(ckaOrtho) > 1e-9 {
		t.Errorf("CKA(orthoX, orthoY) = %.12f, want 0.0", ckaOrtho)
	}

	// 4. Range bound [0, 1] and symmetry CKA(A, B) == CKA(B, A)
	actB := [][]float64{
		{0.2, -1.0, 0.5, 2.0},
		{1.5, 0.3, -0.5, 1.0},
		{-0.8, 2.1, 0.0, -1.5},
		{0.0, -0.4, 1.2, 0.8},
	}
	ckaAB := CKA(actA, actB)
	ckaBA := CKA(actB, actA)
	if ckaAB < -1e-12 || ckaAB > 1.0+1e-12 {
		t.Errorf("CKA(A, B) = %f out of [0, 1]", ckaAB)
	}
	if math.Abs(ckaAB-ckaBA) > 1e-12 {
		t.Errorf("CKA symmetry violated: %f != %f", ckaAB, ckaBA)
	}
}

func TestAccuracy_KLDivergenceCalibrationBounds(t *testing.T) {
	p := []float64{2.0, 1.0, 0.1, -1.0}
	q := []float64{1.8, 1.1, 0.0, -0.8}

	// Identity KL == 0
	klSelf, err := ComputeKLDivergence(p, p)
	if err != nil || math.Abs(klSelf) > 1e-12 {
		t.Fatalf("ComputeKLDivergence(p, p) = %v, err=%v; want 0", klSelf, err)
	}

	// Shift invariance of logits: KL(p + c, q + d) == KL(p, q)
	pShift := []float64{102.0, 101.0, 100.1, 99.0}
	qShift := []float64{-98.2, -98.9, -100.0, -100.8}
	kl1, err := ComputeKLDivergence(p, q)
	if err != nil {
		t.Fatalf("ComputeKLDivergence failed: %v", err)
	}
	kl2, err := ComputeKLDivergence(pShift, qShift)
	if err != nil {
		t.Fatalf("ComputeKLDivergence shifted failed: %v", err)
	}
	if kl1 < 0 {
		t.Errorf("Gibbs inequality violated: KL = %f < 0", kl1)
	}
	if math.Abs(kl1-kl2) > 1e-12 {
		t.Errorf("shift invariance violated: %v vs %v", kl1, kl2)
	}
}
