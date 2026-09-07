package steer

import (
	"math"
	"testing"
)

func TestBehavioralCommitmentDetector_Analyze(t *testing.T) {
	cfg := DefaultCommitmentConfig("gemma")
	detector := NewBehavioralCommitmentDetector(cfg)

	// Simulated scores across 20 tokens with a sharp decision commitment at token 8
	scores := []float64{
		0.05, 0.06, 0.05, 0.07, 0.08, 0.10, 0.12, 0.15,
		0.75, // sharp jump at token 8 (derivative = 0.60)
		0.80, 0.82, 0.83, 0.84, 0.85, 0.85, 0.86,
		0.87, 0.87, 0.88, 0.88,
	}

	prefillLen := 14
	res, err := detector.AnalyzeCommitmentWindow(scores, prefillLen)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if res.PeakPosition != 8 {
		t.Errorf("expected peak position 8, got %d", res.PeakPosition)
	}

	if math.Abs(res.MaxDerivative-0.60) > 1e-6 {
		t.Errorf("expected max derivative 0.60, got %f", res.MaxDerivative)
	}

	if res.PrefillEnergyRatio < 0.80 {
		t.Errorf("expected prefill energy ratio > 0.80, got %f", res.PrefillEnergyRatio)
	}

	// Inside commitment window (e.g. token 8): full strength
	sPeak := detector.ComputeSteeringMultiplier(8, prefillLen, res, 1.0)
	if sPeak != 1.0 {
		t.Errorf("expected full strength 1.0 at peak, got %f", sPeak)
	}

	// Post-prefill token (token 18) for Gemma (prefill-dominant): attenuated to PostCommitDecay (0.10)
	sPost := detector.ComputeSteeringMultiplier(18, prefillLen, res, 1.0)
	if math.Abs(sPost-0.10) > 1e-6 {
		t.Errorf("expected Gemma post-commit decay ~0.10, got %f", sPost)
	}
}

func TestBehavioralCommitmentDetector_LlamaStandard(t *testing.T) {
	cfg := DefaultCommitmentConfig("llama")
	detector := NewBehavioralCommitmentDetector(cfg)

	scores := []float64{0.1, 0.2, 0.6, 0.7, 0.8}
	res, err := detector.AnalyzeCommitmentWindow(scores, 3)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Llama is not prefill-dominant; post-prefill decoding retains active steering
	sDec := detector.ComputeSteeringMultiplier(4, 3, res, 1.0)
	if sDec < 0.5 {
		t.Errorf("expected Llama to retain active steering (>=0.5), got %f", sDec)
	}
}
