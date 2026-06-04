package math

import (
	"math"
	"testing"
)

func TestCovariance_KnownGaussian(t *testing.T) {
	rng := newSeededRNG(0xC0FFEE)

	const d = 4
	const n = 5000

	target := [][]float64{
		{1.0, 0.3, -0.2, 0.0},
		{0.3, 2.0, 0.1, -0.4},
		{-0.2, 0.1, 0.5, 0.25},
		{0.0, -0.4, 0.25, 1.5},
	}
	choleskyL := cholesky(target, d)

	data := make([][]float64, n)
	for i := 0; i < n; i++ {
		z := make([]float64, d)
		for j := 0; j < d; j++ {
			z[j] = rng.NormFloat64()
		}
		row := make([]float64, d)
		for r := 0; r < d; r++ {
			var s float64
			for c := 0; c < d; c++ {
				s += choleskyL[r][c] * z[c]
			}
			row[r] = s
		}
		data[i] = row
	}

	got := Covariance(data)
	if got == nil {
		t.Fatal("Covariance returned nil")
	}
	if len(got) != d || len(got[0]) != d {
		t.Fatalf("cov shape = %dx%d, want %dx%d", len(got), len(got[0]), d, d)
	}
	for i := 0; i < d; i++ {
		for j := 0; j < d; j++ {
			want := target[i][j]
			if math.Abs(got[i][j]-want) > 0.1 {
				t.Errorf("cov[%d][%d] = %f, want ~%f (tol 0.1)", i, j, got[i][j], want)
			}
		}
	}
}

func TestCovariance_SymmetricAndFinite(t *testing.T) {
	data := [][]float64{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
		{10, 11, 12},
	}
	cov := Covariance(data)
	if cov == nil {
		t.Fatal("Covariance returned nil")
	}
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if math.Abs(cov[i][j]-cov[j][i]) > 1e-12 {
				t.Errorf("cov not symmetric: cov[%d][%d]=%f, cov[%d][%d]=%f",
					i, j, cov[i][j], j, i, cov[j][i])
			}
			if math.IsNaN(cov[i][j]) || math.IsInf(cov[i][j], 0) {
				t.Errorf("cov[%d][%d] = %f, want finite", i, j, cov[i][j])
			}
		}
	}
}

func TestCovariance_DegenerateInputs(t *testing.T) {
	if Covariance(nil) != nil {
		t.Error("Covariance(nil) should be nil")
	}
	if Covariance([][]float64{{1, 2}}) != nil {
		t.Error("Covariance(n=1) should be nil")
	}
	if Covariance([][]float64{}) != nil {
		t.Error("Covariance(empty) should be nil")
	}
	if Covariance([][]float64{{1, 2}, {3}}) != nil {
		t.Error("Covariance(ragged) should be nil")
	}
}

func TestShrinkageCovariance_AutoFullRankWhenNLessThanD(t *testing.T) {
	const d = 50
	const n = 5

	rng := newSeededRNG(42)
	data := make([][]float64, n)
	for i := 0; i < n; i++ {
		row := make([]float64, d)
		for j := 0; j < d; j++ {
			row[j] = rng.NormFloat64()
		}
		data[i] = row
	}

	plain := Covariance(data)
	if plain != nil {
		if isFullRank(plain, 1e-6) {
			t.Log("plain Covariance happened to be full rank with n<d (rare but acceptable)")
		}
	}

	shrunk := ShrinkageCovariance(data, -1)
	if shrunk == nil {
		t.Fatal("ShrinkageCovariance returned nil")
	}
	if !isFullRank(shrunk, 1e-6) {
		t.Fatal("auto-shrunk covariance is not full rank — rank safety violated")
	}
	for i := 0; i < d; i++ {
		if shrunk[i][i] <= 0 {
			t.Errorf("shrunk diagonal[%d] = %f, want > 0", i, shrunk[i][i])
		}
	}
	for i := 0; i < d; i++ {
		for j := 0; j < d; j++ {
			if math.Abs(shrunk[i][j]-shrunk[j][i]) > 1e-9 {
				t.Errorf("shrunk not symmetric: [%d][%d] vs [%d][%d]", i, j, j, i)
			}
		}
	}
}

func TestShrinkageCovariance_ExplicitIntensity(t *testing.T) {
	data := [][]float64{
		{1, 0},
		{0, 1},
		{2, 0},
		{0, 2},
	}

	zero := ShrinkageCovariance(data, 0.0)
	mu := AverageVariance(zero)

	shrunk1 := ShrinkageCovariance(data, 1.0)
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			want := mu * kroneckerDelta(i, j)
			if math.Abs(shrunk1[i][j]-want) > 1e-12 {
				t.Errorf("full-shrink [%d][%d] = %f, want %f", i, j, shrunk1[i][j], want)
			}
		}
	}

	half := ShrinkageCovariance(data, 0.5)
	for i := 0; i < 2; i++ {
		want := 0.5*zero[i][i] + 0.5*mu
		if math.Abs(half[i][i]-want) > 1e-12 {
			t.Errorf("half-shrink diagonal[%d] = %f, want %f", i, half[i][i], want)
		}
		wantOff := 0.5 * zero[i][1]
		if i == 1 {
			continue
		}
		if math.Abs(half[i][1]-wantOff) > 1e-12 {
			t.Errorf("half-shrink off-diagonal[%d][1] = %f, want %f", i, half[i][1], wantOff)
		}
	}
}

func TestShrinkageCovariance_ClipsIntensity(t *testing.T) {
	data := [][]float64{{1, 2}, {3, 4}, {5, 6}}
	zero := ShrinkageCovariance(data, 0.0)
	mu := AverageVariance(zero)
	over := ShrinkageCovariance(data, 5.0)
	under := ShrinkageCovariance(data, -2.0)

	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			want := mu * kroneckerDelta(i, j)
			if math.Abs(over[i][j]-want) > 1e-12 {
				t.Errorf("shrinkage=5 (clipped to 1) should equal μ·I: [%d][%d] %f vs %f",
					i, j, over[i][j], want)
			}
		}
	}

	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			if math.IsNaN(under[i][j]) || math.IsInf(under[i][j], 0) {
				t.Errorf("auto shrinkage produced non-finite value at [%d][%d] = %f", i, j, under[i][j])
			}
		}
	}
}

func TestShrinkageCovariance_ReducesToMLWhenPLentyOfData(t *testing.T) {
	const d = 3
	const n = 2000
	rng := newSeededRNG(7)
	data := make([][]float64, n)
	for i := 0; i < n; i++ {
		row := make([]float64, d)
		for j := 0; j < d; j++ {
			row[j] = rng.NormFloat64()
		}
		data[i] = row
	}
	shrunk := ShrinkageCovariance(data, -1)
	plain := Covariance(data)
	for i := 0; i < d; i++ {
		for j := 0; j < d; j++ {
			if math.Abs(shrunk[i][j]-plain[i][j]) > 0.2 {
				t.Errorf("with n>>d, shrunk should be near plain: [%d][%d] shrunk=%f plain=%f",
					i, j, shrunk[i][j], plain[i][j])
			}
		}
	}
}

func TestShrinkageCovariance_NaNInfSafe(t *testing.T) {
	data := [][]float64{
		{math.NaN(), 1, 2},
		{3, math.Inf(1), 5},
		{6, 7, math.Inf(-1)},
		{9, 10, 11},
	}
	cov := ShrinkageCovariance(data, 0.3)
	if cov == nil {
		t.Fatal("ShrinkageCovariance returned nil")
	}
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if math.IsNaN(cov[i][j]) || math.IsInf(cov[i][j], 0) {
				t.Errorf("cov[%d][%d] = %f, want finite", i, j, cov[i][j])
			}
		}
	}
}

func TestAverageVariance(t *testing.T) {
	cov := [][]float64{
		{2, 0, 0},
		{0, 4, 0},
		{0, 0, 6},
	}
	if got := AverageVariance(cov); math.Abs(got-4.0) > 1e-12 {
		t.Errorf("AverageVariance = %f, want 4.0", got)
	}
	if got := AverageVariance(nil); got != 0 {
		t.Errorf("AverageVariance(nil) = %f, want 0", got)
	}
	if got := AverageVariance([][]float64{}); got != 0 {
		t.Errorf("AverageVariance(empty) = %f, want 0", got)
	}
}

func isFullRank(m [][]float64, tol float64) bool {
	d := len(m)
	if d == 0 {
		return true
	}
	aug := make([][]float64, d)
	for i := range aug {
		aug[i] = make([]float64, 2*d)
		copy(aug[i], m[i])
	}
	pivots := make([]int, d)
	for i := range pivots {
		pivots[i] = -1
	}
	row := 0
	for col := 0; col < d && row < d; col++ {
		pivot := -1
		maxVal := tol
		for r := row; r < d; r++ {
			if math.Abs(aug[r][col]) > maxVal {
				maxVal = math.Abs(aug[r][col])
				pivot = r
			}
		}
		if pivot == -1 {
			return false
		}
		aug[row], aug[pivot] = aug[pivot], aug[row]
		pivots[row] = col
		pv := aug[row][col]
		for j := 0; j < 2*d; j++ {
			aug[row][j] /= pv
		}
		for r := 0; r < d; r++ {
			if r == row {
				continue
			}
			factor := aug[r][col]
			if factor == 0 {
				continue
			}
			for j := 0; j < 2*d; j++ {
				aug[r][j] -= factor * aug[row][j]
			}
		}
		row++
	}
	return row == d
}
