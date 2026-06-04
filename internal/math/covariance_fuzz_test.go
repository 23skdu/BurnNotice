package math

import (
	"math"
	"testing"
)

func FuzzCovariance(f *testing.F) {
	f.Add(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
	f.Add(-1.0, -2.0, -3.0, 1.0, 2.0, 3.0)
	f.Add(1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10)
	f.Add(1e10, 0.0, -1e10, 1e10, 0.0, -1e10)

	f.Fuzz(func(t *testing.T, a1, a2, b1, b2, c1, c2 float64) {
		data := [][]float64{{a1, a2}, {b1, b2}, {c1, c2}}
		cov := Covariance(data)
		if cov == nil {
			t.Fatal("Covariance returned nil on 3x2 data")
		}
		for i := 0; i < 2; i++ {
			for j := 0; j < 2; j++ {
				if math.IsNaN(cov[i][j]) || math.IsInf(cov[i][j], 0) {
					t.Errorf("cov[%d][%d] = %f, want finite", i, j, cov[i][j])
				}
				if math.Abs(cov[i][j]-cov[j][i]) > 1e-9 {
					t.Errorf("cov not symmetric: [%d][%d]=%f vs [%d][%d]=%f",
						i, j, cov[i][j], j, i, cov[j][i])
				}
			}
		}
	})
}

func FuzzShrinkageCovariance(f *testing.F) {
	f.Add(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0)
	f.Add(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5)
	f.Add(1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, -1.0)
	f.Add(1e10, -1e10, 1e10, -1e10, 1e10, -1e10, 1.0)
	f.Add(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0)

	f.Fuzz(func(t *testing.T, a1, a2, b1, b2, c1, c2, shrink float64) {
		data := [][]float64{{a1, a2}, {b1, b2}, {c1, c2}}
		cov := ShrinkageCovariance(data, shrink)
		if cov == nil {
			t.Fatal("ShrinkageCovariance returned nil on 3x2 data")
		}
		for i := 0; i < 2; i++ {
			for j := 0; j < 2; j++ {
				if math.IsNaN(cov[i][j]) || math.IsInf(cov[i][j], 0) {
					t.Errorf("cov[%d][%d] = %f, want finite (shrink=%f)", i, j, cov[i][j], shrink)
				}
				if math.Abs(cov[i][j]-cov[j][i]) > 1e-9 {
					t.Errorf("cov not symmetric: [%d][%d]=%f vs [%d][%d]=%f",
						i, j, cov[i][j], j, i, cov[j][i])
				}
			}
		}
	})
}
