package math

import (
	"math"
	"testing"
)

func TestRidgeRegularizedProjection_Success(t *testing.T) {
	// 3D vector, 2D basis
	W := []float64{1.0, 2.0, 3.0}
	V_k := [][]float64{
		{1.0, 0.0, 0.0},
		{0.0, 1.0, 0.0},
	}

	// For alpha = 0, this is a standard orthogonal projection onto x-y plane.
	// Expected projection is {1.0, 2.0, 0.0}
	proj, err := RidgeRegularizedProjection(W, V_k, 0.0)
	if err != nil {
		t.Fatalf("projection failed: %v", err)
	}
	if len(proj) != 3 {
		t.Fatalf("expected 3D vector, got %dD", len(proj))
	}
	if math.Abs(proj[0]-1.0) > 1e-9 || math.Abs(proj[1]-2.0) > 1e-9 || math.Abs(proj[2]) > 1e-9 {
		t.Errorf("expected {1.0, 2.0, 0.0}, got %v", proj)
	}

	// For non-zero alpha, say alpha = 1.0
	// Gram matrix is V_k^T · V_k = [[1, 0], [0, 1]].
	// Regularized Gram matrix: V_k^T · V_k + alpha * I = [[2, 0], [0, 2]].
	// Inverse is [[0.5, 0], [0, 0.5]].
	// V_k^T · W = {1.0, 2.0}.
	// x = {0.5, 1.0}.
	// W_proj = V_k · x = {0.5, 1.0, 0.0}.
	projReg, err := RidgeRegularizedProjection(W, V_k, 1.0)
	if err != nil {
		t.Fatalf("regularized projection failed: %v", err)
	}
	if math.Abs(projReg[0]-0.5) > 1e-9 || math.Abs(projReg[1]-1.0) > 1e-9 || math.Abs(projReg[2]) > 1e-9 {
		t.Errorf("expected {0.5, 1.0, 0.0}, got %v", projReg)
	}
}

func TestRidgeRegularizedProjection_Errors(t *testing.T) {
	// Empty basis
	_, err := RidgeRegularizedProjection([]float64{1, 2}, [][]float64{}, 0.1)
	if err == nil {
		t.Error("expected error for empty basis")
	}

	// Dimension mismatch
	W := []float64{1.0, 2.0}
	V_k := [][]float64{
		{1.0, 0.0, 0.0},
	}
	_, err = RidgeRegularizedProjection(W, V_k, 0.1)
	if err == nil {
		t.Error("expected error for basis vector dimension mismatch")
	}

	// Non-positive definite Gram matrix (with alpha=0 and singular basis)
	W2 := []float64{1.0}
	V_k2 := [][]float64{
		{1.0},
		{2.0}, // singular in 1D space
	}
	_, err = RidgeRegularizedProjection(W2, V_k2, 0.0)
	if err == nil {
		t.Error("expected cholesky failure for linearly dependent basis with alpha=0")
	}
}

func TestAlphaScheduler(t *testing.T) {
	s := NewAlphaScheduler(10)
	if s.AlphaHigh != 1.0 || s.AlphaLow != 0.0 || s.TotalPasses != 10 {
		t.Errorf("unexpected defaults: %+v", s)
	}

	// pass = 0 should return AlphaHigh
	a0 := s.Alpha(0)
	if math.Abs(a0-1.0) > 1e-9 {
		t.Errorf("Alpha(0) = %f, want 1.0", a0)
	}

	// pass >= TotalPasses should return AlphaLow
	a10 := s.Alpha(10)
	if math.Abs(a10-0.0) > 1e-9 {
		t.Errorf("Alpha(10) = %f, want 0.0", a10)
	}

	// pass in between should decay
	a5 := s.Alpha(5)
	if a5 >= 1.0 || a5 <= 0.0 {
		t.Errorf("Alpha(5) = %f should be in (0, 1)", a5)
	}

	// test invalid passes clamp in NewAlphaScheduler
	s2 := NewAlphaScheduler(0)
	if s2.TotalPasses != 1 {
		t.Errorf("expected TotalPasses = 1, got %d", s2.TotalPasses)
	}
}

func TestCholeskyDecomp_Errors(t *testing.T) {
	// Singular diagonal element during division
	// Under Cholesky, A = L * L^T. If diagonal of L becomes 0, it means matrix is singular.
	A := [][]float64{
		{1.0, 2.0},
		{2.0, 1.0},
	}
	_, err := choleskyDecomp(A)
	if err == nil {
		t.Error("expected error for non-positive definite matrix")
	}
}
