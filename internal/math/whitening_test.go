package math

import (
	"math"
	"testing"
)

func TestWhiteningMatrix_Identity(t *testing.T) {
	ident := [][]float64{
		{1, 0, 0},
		{0, 1, 0},
		{0, 0, 1},
	}
	W, ok := WhiteningMatrix(ident, 1e-10)
	if !ok {
		t.Fatal("WhiteningMatrix on identity returned !ok")
	}
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			want := kroneckerDelta(i, j)
			if math.Abs(W[i][j]-want) > 1e-9 {
				t.Errorf("W[%d][%d] = %f, want %f", i, j, W[i][j], want)
			}
		}
	}
}

func TestWhiteningMatrix_Diagonal(t *testing.T) {
	diag := [][]float64{
		{4, 0, 0},
		{0, 9, 0},
		{0, 0, 16},
	}
	W, ok := WhiteningMatrix(diag, 1e-12)
	if !ok {
		t.Fatal("WhiteningMatrix on diagonal returned !ok")
	}
	expected := [][]float64{
		{0.5, 0, 0},
		{0, 1.0 / 3, 0},
		{0, 0, 0.25},
	}
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if math.Abs(W[i][j]-expected[i][j]) > 1e-9 {
				t.Errorf("W[%d][%d] = %f, want %f", i, j, W[i][j], expected[i][j])
			}
		}
	}
}

func TestWhiteningMatrix_RoundTrip(t *testing.T) {
	cov := [][]float64{
		{2.5, 0.7, -0.3},
		{0.7, 1.8, 0.4},
		{-0.3, 0.4, 1.2},
	}
	W, ok := WhiteningMatrix(cov, 1e-10)
	if !ok {
		t.Fatal("WhiteningMatrix returned !ok")
	}

	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			prod := 0.0
			for k := 0; k < 3; k++ {
				prod += W[i][k] * W[j][k]
			}
			inv := 0.0
			for k := 0; k < 3; k++ {
				inv += W[i][k] * cov[k][j]
			}
			_ = inv
		}
	}

	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			got := 0.0
			for k := 0; k < 3; k++ {
				mw := 0.0
				for l := 0; l < 3; l++ {
					mw += cov[k][l] * W[l][j]
				}
				got += W[i][k] * mw
			}
			want := kroneckerDelta(i, j)
			if math.Abs(got-want) > 1e-6 {
				t.Errorf("(W @ cov @ W)[%d][%d] = %f, want %f", i, j, got, want)
			}
		}
	}
}

func TestWhiteningMatrix_Symmetric(t *testing.T) {
	cov := [][]float64{
		{3, 1, 0.5},
		{1, 2, 0.3},
		{0.5, 0.3, 1.5},
	}
	W, ok := WhiteningMatrix(cov, 1e-12)
	if !ok {
		t.Fatal("WhiteningMatrix returned !ok")
	}
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if math.Abs(W[i][j]-W[j][i]) > 1e-9 {
				t.Errorf("W not symmetric: W[%d][%d]=%f vs W[%d][%d]=%f",
					i, j, W[i][j], j, i, W[j][i])
			}
		}
	}
}

func TestWhiteningMatrix_AsymmetricRejected(t *testing.T) {
	bad := [][]float64{
		{1, 0.5, 0},
		{0.3, 1, 0},
		{0, 0, 1},
	}
	if _, ok := WhiteningMatrix(bad, 1e-10); ok {
		t.Error("WhiteningMatrix should reject asymmetric covariance")
	}
}

func TestWhiteningMatrix_HandlesRankDeficient(t *testing.T) {
	rank1 := [][]float64{
		{1, 2, 3},
		{2, 4, 6},
		{3, 6, 9},
	}
	W, ok := WhiteningMatrix(rank1, 1e-3)
	if !ok {
		t.Fatal("WhiteningMatrix on rank-1 cov returned !ok")
	}
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if math.IsNaN(W[i][j]) || math.IsInf(W[i][j], 0) {
				t.Errorf("W[%d][%d] = %f, want finite (rank-1 input)", i, j, W[i][j])
			}
		}
	}
}

func TestUnwhiteningMatrix_WhiteningInverse(t *testing.T) {
	cov := [][]float64{
		{2.5, 0.7, -0.3},
		{0.7, 1.8, 0.4},
		{-0.3, 0.4, 1.2},
	}
	W, ok := WhiteningMatrix(cov, 1e-10)
	if !ok {
		t.Fatal("WhiteningMatrix returned !ok")
	}
	U, ok := UnwhiteningMatrix(cov, 1e-10)
	if !ok {
		t.Fatal("UnwhiteningMatrix returned !ok")
	}
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			prod := 0.0
			for k := 0; k < 3; k++ {
				prod += W[i][k] * U[k][j]
			}
			if math.Abs(prod-kroneckerDelta(i, j)) > 1e-6 {
				t.Errorf("(W @ U)[%d][%d] = %f, want %f", i, j, prod, kroneckerDelta(i, j))
			}
		}
	}
}

func TestApplyWhitening_BasicMatrixMultiply(t *testing.T) {
	W := [][]float64{
		{2, 0},
		{0, 3},
	}
	X := [][]float64{
		{1, 0},
		{0, 1},
		{1, 1},
	}
	got := ApplyWhitening(X, W)
	want := [][]float64{{2, 0}, {0, 3}, {2, 3}}
	if len(got) != 3 {
		t.Fatalf("len(got) = %d, want 3", len(got))
	}
	for i := range got {
		for j := 0; j < 2; j++ {
			if math.Abs(got[i][j]-want[i][j]) > 1e-12 {
				t.Errorf("got[%d][%d] = %f, want %f", i, j, got[i][j], want[i][j])
			}
		}
	}
}

func TestApplyWhitening_DimMismatch(t *testing.T) {
	X := [][]float64{{1, 2, 3}}
	W := [][]float64{{1, 0}, {0, 1}}
	if got := ApplyWhitening(X, W); got != nil {
		t.Errorf("ApplyWhitening with mismatched dims should return nil, got %v", got)
	}
}

func TestApplyWhitening_Empty(t *testing.T) {
	if got := ApplyWhitening(nil, [][]float64{{1}}); got != nil {
		t.Errorf("ApplyWhitening(nil, ...) should return nil, got %v", got)
	}
}

func TestZCAWhitening_ProducesIdentityCovariance(t *testing.T) {
	rng := newSeededRNG(0xBEEF)

	cov := [][]float64{
		{2.5, 0.7, -0.3},
		{0.7, 1.8, 0.4},
		{-0.3, 0.4, 1.2},
	}
	L := cholesky(cov, 3)

	const n = 5000
	data := make([][]float64, n)
	for i := 0; i < n; i++ {
		z := []float64{rng.NormFloat64(), rng.NormFloat64(), rng.NormFloat64()}
		row := make([]float64, 3)
		for r := 0; r < 3; r++ {
			var s float64
			for c := 0; c < 3; c++ {
				s += L[r][c] * z[c]
			}
			row[r] = s
		}
		data[i] = row
	}

	Xwhite, W, _, ok := ZCAWhitening(data, 0.0, 0.0)
	if !ok {
		t.Fatal("ZCAWhitening returned !ok")
	}
	if len(Xwhite) != n {
		t.Fatalf("len(Xwhite) = %d, want %d", len(Xwhite), n)
	}

	got := Covariance(Xwhite)
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			want := kroneckerDelta(i, j)
			if math.Abs(got[i][j]-want) > 0.1 {
				t.Errorf("whitened cov[%d][%d] = %f, want ~%f", i, j, got[i][j], want)
			}
		}
	}

	if W == nil || len(W) != 3 {
		t.Fatalf("W has wrong shape: %v", W)
	}
}

func TestZCAWhitening_RankDeficientInput(t *testing.T) {
	const d = 50
	const n = 5
	rng := newSeededRNG(7)
	data := make([][]float64, n)
	for i := 0; i < n; i++ {
		row := make([]float64, d)
		for j := 0; j < d; j++ {
			row[j] = rng.NormFloat64()
		}
		data[i] = row
	}
	Xwhite, W, _, ok := ZCAWhitening(data, -1.0, 0.0)
	if !ok {
		t.Fatal("ZCAWhitening returned !ok on n<d input")
	}
	if len(Xwhite) != n {
		t.Fatalf("len(Xwhite) = %d, want %d", len(Xwhite), n)
	}
	for i := 0; i < n; i++ {
		for j := 0; j < d; j++ {
			if math.IsNaN(Xwhite[i][j]) || math.IsInf(Xwhite[i][j], 0) {
				t.Errorf("Xwhite[%d][%d] = %f, want finite", i, j, Xwhite[i][j])
			}
		}
	}
	for i := 0; i < d; i++ {
		for j := 0; j < d; j++ {
			if math.IsNaN(W[i][j]) || math.IsInf(W[i][j], 0) {
				t.Errorf("W[%d][%d] = %f, want finite", i, j, W[i][j])
			}
		}
	}
}

func TestZCAWhitening_RejectsBadInput(t *testing.T) {
	if _, _, _, ok := ZCAWhitening(nil, 0, 0); ok {
		t.Error("ZCAWhitening(nil) should return ok=false")
	}
	if _, _, _, ok := ZCAWhitening([][]float64{{1, 2}}, 0, 0); ok {
		t.Error("ZCAWhitening(n=1) should return ok=false")
	}
	ragged := [][]float64{{1, 2}, {3, 4, 5}}
	if _, _, _, ok := ZCAWhitening(ragged, 0, 0); ok {
		t.Error("ZCAWhitening(ragged) should return ok=false")
	}
}
