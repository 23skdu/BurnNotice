package scoring

import (
	"math"
	"testing"
)

func FuzzScorer_Score(f *testing.F) {
	f.Add(1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
	f.Add(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
	f.Add(-1.0, -2.0, -3.0, 1.0, 2.0, 3.0)

	f.Fuzz(func(t *testing.T, a1, a2, a3, d1, d2, d3 float64) {
		for _, v := range []float64{a1, a2, a3, d1, d2, d3} {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Skip()
			}
		}

		s := NewScorer()
		s.RegisterDirection("test", []float64{d1, d2, d3})

		scores := s.Score([]float64{a1, a2, a3})

		if scores == nil {
			return
		}

		score := scores["test"]
		if score < 0 || score > 1 {
			t.Errorf("score = %f, want [0, 1]", score)
		}
	})
}

func FuzzScorer_Subspace(f *testing.F) {
	f.Add(1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
	f.Add(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

	f.Fuzz(func(t *testing.T, a1, a2, a3, b1, b2, b3 float64) {
		for _, v := range []float64{a1, a2, a3, b1, b2, b3} {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Skip()
			}
		}

		s := NewScorer()
		basis := [][]float64{{b1, b2, b3}}

		energy := s.ScoreSubspace([]float64{a1, a2, a3}, basis)

		if energy < 0 {
			t.Errorf("energy = %f, want >= 0", energy)
		}
		if math.IsNaN(energy) || math.IsInf(energy, 0) {
			t.Errorf("energy is NaN or Inf")
		}
	})
}
