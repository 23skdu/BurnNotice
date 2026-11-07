package hallucination

import (
	"math"
	"testing"
)

func TestCalibration_HallucinationRollingWindowAndSpikes(t *testing.T) {
	thresholds := map[int]float64{
		0: 10.0,
		1: 25.0,
	}
	det := NewHallucinationDetector(thresholds, 2.0, 4)
	if err := det.Validate(); err != nil {
		t.Fatalf("Validate failed: %v", err)
	}

	// Warm up 4-step window on layer 0 with energy = 4.0 (features = [2.0, 0.0])
	baseFeat := []float64{2.0, 0.0}
	for i := 0; i < 4; i++ {
		if det.DetectSpike(0, baseFeat) {
			t.Fatalf("unexpected spike during window warmup at step %d", i)
		}
	}

	if math.Abs(det.BaselineEnergy(0)-4.0) > 1e-12 {
		t.Errorf("BaselineEnergy(0) = %f, want 4.0", det.BaselineEnergy(0))
	}

	// Exact boundary: energy = 8.0 == avg(4.0) * multiplier(2.0) -> strict '>' should be false
	boundaryFeat := []float64{2.0, 2.0}
	if det.DetectSpike(0, boundaryFeat) {
		t.Errorf("expected false at exact spike multiplier boundary (energy=8.0, threshold=8.0)")
	}

	// Rolling window now has [4, 4, 4, 8] -> avg = 5.0, spike threshold = 10.0
	// Energy = 10.24 (features = [3.2, 0.0]) > 10.0 -> spike detected AND exceeds layer 0 threshold (10.0)
	spikeFeat := []float64{3.2, 0.0}
	if !det.DetectSpike(0, spikeFeat) {
		t.Errorf("expected spike detection for energy 10.24 > 10.0")
	}
	if !det.DetectHallucination(0, spikeFeat) {
		t.Errorf("expected hallucination threshold breach for energy 10.24 > 10.0")
	}

	// RiskScore clamped in [0, 1]
	if score := det.RiskScore(0, spikeFeat); score != 1.0 {
		t.Errorf("RiskScore above threshold = %f, want 1.0", score)
	}
	if score := det.RiskScore(1, baseFeat); math.Abs(score-4.0/25.0) > 1e-12 {
		t.Errorf("RiskScore(1) = %f, want 0.16", score)
	}

	// Verify multi-family SAE energy ingestion (Gemma Scope, Llama-Scope, Qwen-SAE, generic)
	for _, fam := range []string{"gemma-scope", "llama-scope", "qwen-sae", "generic"} {
		det.ResetHistory()
		for i := 0; i < 4; i++ {
			r, sp, hal := det.IngestFamilyFeatures(0, baseFeat, fam)
			if sp || hal || r <= 0 {
				t.Errorf("family %s warmup step %d unexpected: r=%f, sp=%v, hal=%v", fam, i, r, sp, hal)
			}
		}
		rSpike, sp, hal := det.IngestFamilyFeatures(0, []float64{5.0, 0.0}, fam)
		if !sp || !hal || rSpike != 1.0 {
			t.Errorf("family %s expected spike & hallucination on [5,0], got r=%f, sp=%v, hal=%v", fam, rSpike, sp, hal)
		}
	}
}
