package engine

import (
	"math"
	"testing"
)

func FuzzMoEDetectRouting(f *testing.F) {
	f.Add(1.0, 2.0, 3.0, 4.0)
	f.Add(0.0, 0.0, 0.0, 0.0)
	f.Add(-5.0, 10.0, -2.0, 0.5)

	f.Fuzz(func(t *testing.T, a, b, c, d float64) {
		if math.IsNaN(a) || math.IsNaN(b) || math.IsNaN(c) || math.IsNaN(d) ||
			math.IsInf(a, 0) || math.IsInf(b, 0) || math.IsInf(c, 0) || math.IsInf(d, 0) {
			t.Skip()
		}
		eng := NewMoEEngine(4, 2, []int{1, 1, 1, 1}, nil)
		idx, w := eng.DetectRouting(0, []float64{a, b, c, d})
		if len(idx) != 2 || len(w) != 2 {
			t.Fatalf("expected 2 results, got %d, %d", len(idx), len(w))
		}
		idxLogits, wLogits := eng.DetectRoutingFromLogits([]float64{a, b, c, d})
		if len(idxLogits) != 2 || len(wLogits) != 2 {
			t.Fatalf("expected 2 logit results, got %d, %d", len(idxLogits), len(wLogits))
		}
	})
}
