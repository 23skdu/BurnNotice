package profile

import (
	"testing"
)

func TestMLPProbe_Predict(t *testing.T) {
	probe := NewMLPProbe(4, 2)
	act := []float64{0.5, -0.5, 0.2, 0.9}

	score := probe.Predict(act)
	if score < 0 || score > 1 {
		t.Errorf("expected score in range [0, 1], got %f", score)
	}
}

func TestAdversarialPerturber_Perturb(t *testing.T) {
	probe := NewMLPProbe(3, 2)
	act := []float64{0.1, 0.2, 0.3}

	ap := &AdversarialPerturber{Epsilon: 0.1}
	perturbed := ap.Perturb(act, probe, 0.0)

	initialScore := probe.Predict(act)
	finalScore := probe.Predict(perturbed)

	if initialScore > 0.0 && finalScore >= initialScore {
		t.Errorf("expected adversarial perturbation to successfully downgrade prediction score, got %f -> %f", initialScore, finalScore)
	}
}

func TestOODAnomalyDetector(t *testing.T) {
	baseline := [][]float64{
		{1.0, 1.0, 1.0},
		{1.1, 0.9, 1.0},
		{0.9, 1.1, 1.0},
		{1.0, 1.0, 0.9},
	}

	detector, err := NewOODAnomalyDetector(baseline, 3.0)
	if err != nil {
		t.Fatalf("failed to create OOD detector: %v", err)
	}

	// Normal in-distribution vector
	normalAct := []float64{1.0, 1.05, 1.0}
	if detector.IsAnomaly(normalAct) {
		t.Error("expected normal activation to NOT be flagged as anomaly")
	}

	// High-obfuscation OOD anomaly vector
	anomalyAct := []float64{8.0, -5.0, 12.0}
	if !detector.IsAnomaly(anomalyAct) {
		t.Error("expected highly perturbed OOD vector to be flagged as anomaly")
	}
}

func TestMLPProbe_Predict_EdgeCases(t *testing.T) {
	probe := &MLPProbe{
		Weights1: [][]float64{},
	}
	if score := probe.Predict([]float64{1.0}); score != 0.5 {
		t.Errorf("expected 0.5 for empty hidden layers, got %f", score)
	}

	probe2 := NewMLPProbe(4, 2)
	if score := probe2.Predict([]float64{}); score != 0.5 {
		t.Errorf("expected 0.5 for empty input activation, got %f", score)
	}
}

func TestOODAnomalyDetector_EmptyBaseline(t *testing.T) {
	_, err := NewOODAnomalyDetector(nil, 3.0)
	if err == nil {
		t.Error("expected error for nil baseline")
	}

	_, err2 := NewOODAnomalyDetector([][]float64{}, 3.0)
	if err2 == nil {
		t.Error("expected error for empty baseline")
	}
}

func TestOODAnomalyDetector_IsAnomaly_MismatchedDimension(t *testing.T) {
	detector, _ := NewOODAnomalyDetector([][]float64{{1.0, 2.0}}, 3.0)
	if !detector.IsAnomaly([]float64{1.0}) {
		t.Error("expected mismatched vector to be flagged as anomaly")
	}
}

