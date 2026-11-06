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

func TestAccuracy_RidgeProjection_OrthogonalAndDegenerate(t *testing.T) {
	// Property 1: Idempotence when alpha = 0 on orthonormal basis (P(P(W)) == P(W))
	W := []float64{2.5, -1.75, 4.25, 0.8}
	Vk := [][]float64{
		{1, 0, 0, 0},
		{0, 0.6, 0.8, 0},
	}

	p1, err := RidgeRegularizedProjection(W, Vk, 0.0)
	if err != nil {
		t.Fatalf("projection 1 failed: %v", err)
	}
	p2, err := RidgeRegularizedProjection(p1, Vk, 0.0)
	if err != nil {
		t.Fatalf("projection 2 failed: %v", err)
	}
	for i := range p1 {
		if math.Abs(p1[i]-p2[i]) > 1e-12 {
			t.Errorf("idempotence violated at %d: %f != %f", i, p1[i], p2[i])
		}
	}

	// Property 2: Residual W - P(W) is orthogonal to every basis vector when alpha = 0
	for bIdx, bVec := range Vk {
		var dot float64
		for i := range W {
			dot += (W[i] - p1[i]) * bVec[i]
		}
		if math.Abs(dot) > 1e-12 {
			t.Errorf("residual not orthogonal to basis %d: dot = %e", bIdx, dot)
		}
	}

	// Property 3: Zero-norm vector W produces exact zero vector
	zeroW := []float64{0, 0, 0, 0}
	pZero, err := RidgeRegularizedProjection(zeroW, Vk, 0.1)
	if err != nil {
		t.Fatalf("zero vector projection failed: %v", err)
	}
	for i, v := range pZero {
		if v != 0 {
			t.Errorf("expected 0 at %d, got %f", i, v)
		}
	}

	// Property 4: Zero-norm basis vector with positive ridge alpha stays finite and well-conditioned
	zeroBasis := [][]float64{{0, 0, 0, 0}}
	pZeroBasis, err := RidgeRegularizedProjection(W, zeroBasis, 0.5)
	if err != nil {
		t.Fatalf("zero basis with alpha > 0 should succeed: %v", err)
	}
	for i, v := range pZeroBasis {
		if v != 0 || math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("expected 0 at %d for zero basis projection, got %f", i, v)
		}
	}
}

func TestInvariant_NumericalStability_NaNInfAndDegenerateCov(t *testing.T) {
	// 1. ClampToFloat64 & Winsorize guarantee IsStable on arbitrary NaN/Inf combinations
	adversarialInputs := [][]float64{
		{math.NaN(), math.NaN(), math.NaN()},
		{math.Inf(1), math.Inf(-1), math.NaN(), 0.0, 42.0},
		{1e300, -1e300, math.NaN(), math.Inf(1)},
	}
	for idx, inp := range adversarialInputs {
		clamped := ClampToFloat64(inp)
		if !IsStable(clamped) {
			t.Errorf("case %d: ClampToFloat64 produced unstable output: %v", idx, clamped)
		}
		win := Winsorize(inp, 0.1, 0.9)
		if !IsStable(win) {
			t.Errorf("case %d: Winsorize produced unstable output: %v", idx, win)
		}
	}

	// 2. Degenerate rank-0 (all-identical rows) covariance & shrinkage covariance stability
	constantData := [][]float64{
		{5.0, -3.0, 2.0},
		{5.0, -3.0, 2.0},
		{5.0, -3.0, 2.0},
		{5.0, -3.0, 2.0},
	}
	cov := Covariance(constantData)
	if cov == nil {
		t.Fatal("Covariance on constant data returned nil")
	}
	shrunk := ShrinkageCovariance(constantData, -1.0)
	if shrunk == nil {
		t.Fatal("ShrinkageCovariance on constant data returned nil")
	}
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if cov[i][j] != 0 || shrunk[i][j] != 0 {
				t.Errorf("expected 0 covariance on constant rows at [%d][%d], got cov=%f shrunk=%f", i, j, cov[i][j], shrunk[i][j])
			}
		}
	}

	// 3. Degenerate rank-1 collinear matrix with explicit positive floor in WhiteningMatrix
	collinearData := [][]float64{
		{1.0, 2.0, 3.0},
		{2.0, 4.0, 6.0},
		{3.0, 6.0, 9.0},
		{4.0, 8.0, 12.0},
	}
	shrunkCollinear := ShrinkageCovariance(collinearData, -1.0)
	W, ok := WhiteningMatrix(shrunkCollinear, 1e-8)
	if !ok || W == nil {
		t.Fatal("WhiteningMatrix failed on Ledoit-Wolf shrunk collinear data")
	}
	for i := range W {
		if !IsStable(W[i]) {
			t.Errorf("WhiteningMatrix row %d is not numerically stable: %v", i, W[i])
		}
	}
}

func BenchmarkRidgeRegularizedProjection(b *testing.B) {
	b.ReportAllocs()
	const d = 4096
	const k = 4
	W := make([]float64, d)
	for i := range W {
		W[i] = float64(i%17) * 0.1
	}
	Vk := make([][]float64, k)
	for i := 0; i < k; i++ {
		Vk[i] = make([]float64, d)
		Vk[i][i] = 1.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = RidgeRegularizedProjection(W, Vk, 0.01)
	}
}

func BenchmarkWinsorize(b *testing.B) {
	b.ReportAllocs()
	vec := make([]float64, 4096)
	for i := range vec {
		vec[i] = float64(i) - 2048.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Winsorize(vec, 0.005, 0.995)
	}
}

func BenchmarkShrinkageCovariance(b *testing.B) {
	b.ReportAllocs()
	const n = 32
	const d = 64
	data := make([][]float64, n)
	for i := 0; i < n; i++ {
		data[i] = make([]float64, d)
		for j := 0; j < d; j++ {
			data[i][j] = float64((i+1)*(j+1)%29) * 0.1
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ShrinkageCovariance(data, -1)
	}
}
