package hallucination

import (
	"math"
	"testing"
)

func FuzzComputeEnergy(f *testing.F) {
	f.Add(0.0, 0.0, 0.0)
	f.Add(1.0, 2.0, 3.0)
	f.Add(-1.5, 4.2, -0.8)

	f.Fuzz(func(t *testing.T, a, b, c float64) {
		if math.IsNaN(a) || math.IsNaN(b) || math.IsNaN(c) ||
			math.IsInf(a, 0) || math.IsInf(b, 0) || math.IsInf(c, 0) {
			t.Skip()
		}
		d := NewHallucinationDetector(nil, 2.0, 3)
		energy := d.ComputeEnergy([]float64{a, b, c})
		if energy < 0 {
			t.Errorf("energy = %f, want >= 0", energy)
		}
	})
}

func FuzzRiskScoreAndThreshold(f *testing.F) {
	f.Add(0, 10.0, 1.0, 2.0)
	f.Add(1, 0.5, 0.2, 0.3)
	f.Add(2, 0.0, 1.0, 1.0)

	f.Fuzz(func(t *testing.T, layerIdx int, threshold, f1, f2 float64) {
		if math.IsNaN(threshold) || math.IsNaN(f1) || math.IsNaN(f2) ||
			math.IsInf(threshold, 0) || math.IsInf(f1, 0) || math.IsInf(f2, 0) {
			t.Skip()
		}
		d := NewHallucinationDetector(map[int]float64{layerIdx: threshold}, 2.0, 4)
		score := d.RiskScore(layerIdx, []float64{f1, f2})
		if !math.IsNaN(score) && (score < 0.0 || score > 1.0) {
			t.Errorf("RiskScore = %f, want in [0.0, 1.0]", score)
		}
		_ = d.DetectHallucination(layerIdx, []float64{f1, f2})
	})
}

func FuzzDetectSpikeRollingWindow(f *testing.F) {
	f.Add(2.0, 3, 1.0, 2.0, 5.0)
	f.Add(1.5, 2, 0.5, 0.5, 0.5)

	f.Fuzz(func(t *testing.T, multiplier float64, window int, v1, v2, v3 float64) {
		if window < 1 || window > 32 {
			t.Skip()
		}
		if math.IsNaN(multiplier) || math.IsInf(multiplier, 0) || multiplier <= 1.0 {
			t.Skip()
		}
		if math.IsNaN(v1) || math.IsNaN(v2) || math.IsNaN(v3) ||
			math.IsInf(v1, 0) || math.IsInf(v2, 0) || math.IsInf(v3, 0) {
			t.Skip()
		}
		d := NewHallucinationDetector(nil, multiplier, window)
		if err := d.Validate(); err != nil {
			t.Fatalf("unexpected Validate error: %v", err)
		}
		for i := 0; i < window; i++ {
			_ = d.DetectSpike(0, []float64{v1, v2})
		}
		_ = d.DetectSpike(0, []float64{v3})
		if b := d.BaselineEnergy(0); b < 0 {
			t.Errorf("BaselineEnergy = %f, want >= 0", b)
		}
	})
}
