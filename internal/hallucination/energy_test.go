package hallucination

import (
	"math"
	"testing"
)

func TestNewHallucinationDetector(t *testing.T) {
	thresholds := map[int]float64{0: 1.0, 1: 2.0}
	d := NewHallucinationDetector(thresholds, 3.0, 5)

	if d.spikeMultiplier != 3.0 {
		t.Errorf("spikeMultiplier = %f, want 3.0", d.spikeMultiplier)
	}
	if d.windowSize != 5 {
		t.Errorf("windowSize = %d, want 5", d.windowSize)
	}
}

func TestComputeEnergy(t *testing.T) {
	d := NewHallucinationDetector(nil, 2.0, 3)

	features := []float64{1, 2, 3}
	energy := d.ComputeEnergy(features)

	expected := 1.0 + 4.0 + 9.0 // 14
	if math.Abs(energy-expected) > 1e-9 {
		t.Errorf("energy = %f, want %f", energy, expected)
	}
}

func TestComputeEnergy_ZeroFeatures(t *testing.T) {
	d := NewHallucinationDetector(nil, 2.0, 3)

	energy := d.ComputeEnergy([]float64{0, 0, 0})
	if energy != 0 {
		t.Errorf("energy = %f, want 0", energy)
	}
}

func TestDetectHallucination(t *testing.T) {
	thresholds := map[int]float64{0: 10.0}
	d := NewHallucinationDetector(thresholds, 2.0, 3)

	// Below threshold
	if d.DetectHallucination(0, []float64{1, 2}) {
		t.Error("should not detect hallucination below threshold")
	}

	// Above threshold
	if !d.DetectHallucination(0, []float64{3, 4}) {
		t.Error("should detect hallucination above threshold")
	}
}

func TestDetectHallucination_DefaultThreshold(t *testing.T) {
	d := NewHallucinationDetector(nil, 2.0, 3)

	// Default threshold is 1.0
	if !d.DetectHallucination(99, []float64{2, 0}) {
		t.Error("should detect with default threshold")
	}
}

func TestDetectSpike(t *testing.T) {
	d := NewHallucinationDetector(nil, 3.0, 3)

	// Fill history
	d.UpdateHistory(0, []float64{1, 0})
	d.UpdateHistory(0, []float64{1, 0})
	d.UpdateHistory(0, []float64{1, 0})

	// Spike: energy = 100, avg = 1, multiplier = 3
	if !d.DetectSpike(0, []float64{10, 0}) {
		t.Error("should detect spike")
	}
}

func TestDetectSpike_NoSpike(t *testing.T) {
	d := NewHallucinationDetector(nil, 3.0, 3)

	d.UpdateHistory(0, []float64{5, 0})
	d.UpdateHistory(0, []float64{5, 0})
	d.UpdateHistory(0, []float64{5, 0})

	if d.DetectSpike(0, []float64{5, 0}) {
		t.Error("should not detect spike when energy equals average")
	}
}

func TestDetectSpike_NotEnoughHistory(t *testing.T) {
	d := NewHallucinationDetector(nil, 3.0, 3)

	// Not enough history yet
	if d.DetectSpike(0, []float64{100, 0}) {
		t.Error("should not detect spike with insufficient history")
	}
}

func TestBaselineEnergy(t *testing.T) {
	d := NewHallucinationDetector(nil, 2.0, 3)

	// UpdateHistory computes energy = sum(features^2)
	// For [2, 0], energy = 4
	// For [4, 0], energy = 16
	d.UpdateHistory(0, []float64{2, 0})
	d.UpdateHistory(0, []float64{4, 0})

	baseline := d.BaselineEnergy(0)
	expected := (4.0 + 16.0) / 2.0 // 10.0
	if math.Abs(baseline-expected) > 1e-9 {
		t.Errorf("baseline = %f, want %f", baseline, expected)
	}
}

func TestBaselineEnergy_NoHistory(t *testing.T) {
	d := NewHallucinationDetector(nil, 2.0, 3)

	baseline := d.BaselineEnergy(0)
	if baseline != 0 {
		t.Errorf("baseline = %f, want 0", baseline)
	}
}

func TestRiskScore(t *testing.T) {
	thresholds := map[int]float64{0: 10.0}
	d := NewHallucinationDetector(thresholds, 2.0, 3)

	// Half threshold
	score := d.RiskScore(0, []float64{math.Sqrt(5), 0}) // energy = 5
	if math.Abs(score-0.5) > 1e-9 {
		t.Errorf("risk score = %f, want 0.5", score)
	}

	// Above threshold (capped at 1.0)
	score = d.RiskScore(0, []float64{4, 0}) // energy = 16
	if score != 1.0 {
		t.Errorf("risk score = %f, want 1.0", score)
	}
}

func TestRiskScore_ZeroThreshold(t *testing.T) {
	thresholds := map[int]float64{0: 0}
	d := NewHallucinationDetector(thresholds, 2.0, 3)

	score := d.RiskScore(0, []float64{1, 0})
	if score != 1.0 {
		t.Errorf("risk score = %f, want 1.0 for zero threshold", score)
	}
}

func TestResetHistory(t *testing.T) {
	d := NewHallucinationDetector(nil, 2.0, 3)

	d.UpdateHistory(0, []float64{1, 0})
	d.UpdateHistory(1, []float64{2, 0})

	d.ResetHistory()

	if d.BaselineEnergy(0) != 0 {
		t.Error("history should be cleared")
	}
}

func TestValidate(t *testing.T) {
	d := NewHallucinationDetector(nil, 2.0, 3)
	if err := d.Validate(); err != nil {
		t.Errorf("valid config should pass: %v", err)
	}
}

func TestValidate_InvalidSpikeMultiplier(t *testing.T) {
	d := NewHallucinationDetector(nil, 1.0, 3)
	if err := d.Validate(); err == nil {
		t.Fatal("expected error for spikeMultiplier <= 1.0")
	}
}

func TestValidate_InvalidWindowSize(t *testing.T) {
	d := NewHallucinationDetector(nil, 2.0, 0)
	if err := d.Validate(); err == nil {
		t.Fatal("expected error for windowSize < 1")
	}
}

func TestHallucinationCalibration_BaselineAndRollingWindow(t *testing.T) {
	windowSize := 4
	thresholds := map[int]float64{0: 25.0, 1: 50.0}
	d := NewHallucinationDetector(thresholds, 2.5, windowSize)
	if err := d.Validate(); err != nil {
		t.Fatalf("unexpected validation error: %v", err)
	}

	// Feed 6 samples into layer 0 to exercise rolling window eviction in UpdateHistory
	// Energies: 1^2=1, 2^2=4, 3^2=9, 4^2=16, 5^2=25, 6^2=36
	for i := 1; i <= 6; i++ {
		d.UpdateHistory(0, []float64{float64(i), 0})
	}

	// Window of size 4 should hold the last 4 energies: 9, 16, 25, 36 -> mean = 86 / 4 = 21.5
	gotBaseline := d.BaselineEnergy(0)
	wantBaseline := (9.0 + 16.0 + 25.0 + 36.0) / 4.0
	if math.Abs(gotBaseline-wantBaseline) > 1e-9 {
		t.Errorf("calibrated rolling baseline = %f, want %f", gotBaseline, wantBaseline)
	}

	// Layer 1 should remain independent with 0 baseline
	if d.BaselineEnergy(1) != 0 {
		t.Errorf("layer 1 baseline = %f, want 0", d.BaselineEnergy(1))
	}
}

func TestHallucinationThreshold_BoundaryAndRiskCalibration(t *testing.T) {
	thresholds := map[int]float64{
		0: 16.0,
		1: -5.0, // non-positive threshold edge case
	}
	d := NewHallucinationDetector(thresholds, 2.0, 3)

	// Exact boundary: energy == threshold (4^2 = 16.0) should NOT trigger (> threshold required)
	if d.DetectHallucination(0, []float64{4.0, 0.0}) {
		t.Error("expected false when energy equals threshold exactly")
	}
	if score := d.RiskScore(0, []float64{4.0, 0.0}); math.Abs(score-1.0) > 1e-9 {
		t.Errorf("RiskScore at exact threshold = %f, want 1.0", score)
	}

	// Slightly above threshold
	if !d.DetectHallucination(0, []float64{4.0001, 0.0}) {
		t.Error("expected true when energy exceeds threshold")
	}

	// Unconfigured layer uses default threshold 1.0 in RiskScore and DetectHallucination
	if score := d.RiskScore(99, []float64{0.5, 0.5}); math.Abs(score-0.5) > 1e-9 {
		t.Errorf("RiskScore with default threshold = %f, want 0.5", score)
	}

	// Negative threshold returns max risk 1.0
	if score := d.RiskScore(1, []float64{0.1}); score != 1.0 {
		t.Errorf("RiskScore for negative threshold = %f, want 1.0", score)
	}
}

func TestHallucinationOracle_SpikeDetectionTrajectory(t *testing.T) {
	d := NewHallucinationDetector(map[int]float64{0: 100.0}, 2.0, 3)

	// Warmup phase (< windowSize): DetectSpike appends to history and returns false
	for i := 0; i < 3; i++ {
		if d.DetectSpike(0, []float64{2.0, 0.0}) { // energy = 4.0
			t.Fatalf("unexpected spike during warmup at step %d", i)
		}
	}
	// Baseline is now 4.0; spike threshold is 4.0 * 2.0 = 8.0
	// Energy = 8.0 (exact boundary) -> not a spike
	if d.DetectSpike(0, []float64{2.0, 2.0}) { // 4 + 4 = 8.0
		t.Error("expected no spike at exact multiplier boundary (energy == avg * multiplier)")
	}

	// History is now [4.0, 4.0, 8.0], avg = 16/3 = 5.333..., threshold = 10.666...
	// Feed energy = 25.0 -> spike!
	if !d.DetectSpike(0, []float64{3.0, 4.0}) {
		t.Error("expected oracle spike detection for energy 25.0 > 10.67")
	}

	// Zero/negative windowSize guard in DetectSpike and UpdateHistory
	dZero := NewHallucinationDetector(nil, 2.0, 0)
	dZero.UpdateHistory(0, []float64{10.0})
	if dZero.DetectSpike(0, []float64{10.0}) {
		t.Error("expected false for DetectSpike when windowSize <= 0")
	}
}
