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
