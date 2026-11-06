package policy

import (
	"math"
	"testing"
)

func TestCalibration_ThresholdDriftAndEscalationBounds(t *testing.T) {
	policies := []Policy{
		{
			Name:            "deception_guard",
			Metric:          "deception",
			Threshold:       0.70,
			Operator:        "greater",
			Action:          ActionSteer,
			EscalationCount: 3,
		},
		{
			Name:      "coherence_floor",
			Metric:    "coherence",
			Threshold: 0.35,
			Operator:  "less",
			Action:    ActionTerminate,
		},
	}

	ks, _ := NewKillSwitch(nil)
	eng := NewEngine(policies)
	eng.RegisterKillSwitch(ks)

	// Boundary check: score == threshold should NOT violate strict inequality
	vExact := eng.Evaluate(map[string]float64{"deception": 0.70, "coherence": 0.35})
	if len(vExact) != 0 {
		t.Fatalf("expected 0 violations at exact threshold boundary, got %d", len(vExact))
	}

	// Drift calibration: shift deception threshold by +0.05 -> 0.75
	eng.AdjustThresholdsForDrift(map[string]float64{"deception": 0.05})
	vBelowDrifted := eng.Evaluate(map[string]float64{"deception": 0.74})
	if len(vBelowDrifted) != 0 {
		t.Fatalf("expected 0 violations below drifted threshold 0.75, got %d", len(vBelowDrifted))
	}

	// Trigger 3 violations above 0.75 to verify escalation boundary
	for i := 1; i <= 3; i++ {
		v := eng.Evaluate(map[string]float64{"deception": 0.75 + float64(i)*1e-4})
		if len(v) != 1 || math.Abs(v[0].Threshold-0.75) > 1e-12 {
			t.Fatalf("step %d: expected violation against threshold 0.75, got %v", i, v)
		}
		wantEscalate := i >= 3
		if eng.ShouldEscalate("deception_guard") != wantEscalate {
			t.Errorf("step %d: ShouldEscalate = %v, want %v", i, eng.ShouldEscalate("deception_guard"), wantEscalate)
		}
	}

	// Coherence below floor triggers emergency kill-switch
	if !ks.IsArmed() {
		t.Fatalf("kill switch should be armed before coherence violation")
	}
	vFloor := eng.Evaluate(map[string]float64{"coherence": 0.3499})
	if len(vFloor) != 1 || ks.IsArmed() {
		t.Errorf("expected coherence floor violation to trigger kill switch, violations=%v armed=%v", vFloor, ks.IsArmed())
	}
}
