package math

import (
	"math"
	"testing"
)

func FuzzWhiteningMatrix(f *testing.F) {
	f.Add(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
	f.Add(4.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 16.0)
	f.Add(2.0, 0.5, 0.0, 0.5, 2.0, 0.0, 0.0, 0.0, 2.0)
	f.Add(1e-10, 0.0, 0.0, 0.0, 1e-10, 0.0, 0.0, 0.0, 1e-10)

	f.Fuzz(func(t *testing.T, a, b, c, d, e, g, h, i, j float64) {
		cov := [][]float64{
			{a, b, c},
			{d, e, g},
			{h, i, j},
		}
		// Force symmetry: the function rejects asymmetric inputs.
		for r := 0; r < 3; r++ {
			for c2 := r + 1; c2 < 3; c2++ {
				mid := (cov[r][c2] + cov[c2][r]) / 2
				cov[r][c2] = mid
				cov[c2][r] = mid
			}
		}
		// Reject non-finite input; the function returns !ok in that case.
		for r := 0; r < 3; r++ {
			for c2 := 0; c2 < 3; c2++ {
				if math.IsNaN(cov[r][c2]) || math.IsInf(cov[r][c2], 0) {
					return
				}
			}
		}
		W, ok := WhiteningMatrix(cov, 1e-8)
		if !ok {
			return
		}
		for r := 0; r < 3; r++ {
			for c2 := 0; c2 < 3; c2++ {
				if math.IsNaN(W[r][c2]) || math.IsInf(W[r][c2], 0) {
					t.Errorf("W[%d][%d] = %f, want finite", r, c2, W[r][c2])
				}
			}
		}
	})
}

func FuzzApplyWhitening(f *testing.F) {
	f.Add(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

	f.Fuzz(func(t *testing.T, x1, x2, x3, y1, y2, y3, w11, w12, w13, w21, w22, w23, w31, w32, w33 float64) {
		X := [][]float64{{x1, x2, x3}, {y1, y2, y3}}
		W := [][]float64{{w11, w12, w13}, {w21, w22, w23}, {w31, w32, w33}}
		out := ApplyWhitening(X, W)
		if out == nil {
			t.Fatal("ApplyWhitening returned nil for matching dims")
		}
		for i := range out {
			for j := 0; j < 3; j++ {
				if math.IsNaN(out[i][j]) || math.IsInf(out[i][j], 0) {
					return
				}
			}
		}
	})
}
