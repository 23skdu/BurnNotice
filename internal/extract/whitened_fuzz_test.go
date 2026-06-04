package extract

import (
	"math"
	"testing"
)

func FuzzExtractWhitenedSVD(f *testing.F) {
	f.Add(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, -1.0, -2.0, -3.0)
	f.Add(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
	f.Add(1e10, -1e10, 1e10, -1e10, 1e10, -1e10, 1e10, -1e10, 1.0, 2.0, 3.0, 4.0)
	f.Add(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
	f.Add(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, -1.0, -2.0, -3.0, -4.0)

	f.Fuzz(func(t *testing.T, p1, p2, p3, p4, p5, p6, p7, p8, n1, n2, n3, n4 float64) {
		pos := [][]float64{{p1, p2, p3, p4}, {p5, p6, p7, p8}}
		neg := [][]float64{{n1, n2, n3, n4}, {n1, n2, n3, n4}}
		dirs, sv, ok := ExtractWhitenedSVD(pos, neg, 2, 0.3, 1e-8)
		if !ok {
			return
		}
		if len(dirs) != 2 || len(sv) != 2 {
			t.Fatalf("len mismatch: dirs=%d sv=%d", len(dirs), len(sv))
		}
		for i, d := range dirs {
			if len(d) != 4 {
				t.Errorf("dirs[%d] dim = %d, want 4", i, len(d))
			}
			for _, v := range d {
				if math.IsNaN(v) || math.IsInf(v, 0) {
					return
				}
			}
			if n := Norm(d); math.Abs(n-1.0) > 1e-4 {
				t.Errorf("dirs[%d] norm = %f, want 1.0", i, n)
			}
		}
		for _, s := range sv {
			if math.IsNaN(s) || math.IsInf(s, 0) {
				return
			}
		}
	})
}
