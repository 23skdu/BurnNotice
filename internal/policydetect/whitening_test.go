package policydetect

import (
	"math"
	"testing"
)

func TestNewWhiteningDetector(t *testing.T) {
	d := NewWhiteningDetector(1e-6)
	if d.epsilon != 1e-6 {
		t.Errorf("epsilon = %e, want 1e-6", d.epsilon)
	}
}

func TestNewWhiteningDetector_DefaultEpsilon(t *testing.T) {
	d := NewWhiteningDetector(0)
	if d.epsilon != 1e-8 {
		t.Errorf("epsilon = %e, want 1e-8", d.epsilon)
	}
}

func TestWhiteningDetector_Train(t *testing.T) {
	d := NewWhiteningDetector(1e-8)

	compliant := [][]float64{
		{1, 0, 0},
		{0, 1, 0},
		{0, 0, 1},
		{1, 1, 0},
	}

	if err := d.Train(compliant, nil); err != nil {
		t.Fatalf("Train error: %v", err)
	}

	if !d.IsTrained() {
		t.Error("detector should be trained")
	}
	if d.Dimension() != 3 {
		t.Errorf("dimension = %d, want 3", d.Dimension())
	}
}

func TestWhiteningDetector_Train_InsufficientSamples(t *testing.T) {
	d := NewWhiteningDetector(1e-8)

	compliant := [][]float64{{1, 0}}

	if err := d.Train(compliant, nil); err == nil {
		t.Fatal("expected error for insufficient samples")
	}
}

func TestWhiteningDetector_Train_InconsistentDims(t *testing.T) {
	d := NewWhiteningDetector(1e-8)

	compliant := [][]float64{
		{1, 0},
		{1, 0, 0},
	}

	if err := d.Train(compliant, nil); err == nil {
		t.Fatal("expected error for inconsistent dims")
	}
}

func TestWhiteningDetector_Detect(t *testing.T) {
	d := NewWhiteningDetector(1e-8)

	compliant := [][]float64{
		{1, 0},
		{0, 1},
		{1, 1},
		{0.5, 0.5},
	}

	if err := d.Train(compliant, nil); err != nil {
		t.Fatalf("Train error: %v", err)
	}

	// Compliant activation (near mean)
	score1, err := d.Detect([]float64{0.625, 0.625})
	if err != nil {
		t.Fatalf("Detect error: %v", err)
	}

	// Should have high compliance score
	if score1 < 0.5 {
		t.Errorf("compliant score = %f, want > 0.5", score1)
	}
}

func TestWhiteningDetector_Detect_NotTrained(t *testing.T) {
	d := NewWhiteningDetector(1e-8)

	_, err := d.Detect([]float64{1, 0})
	if err == nil {
		t.Fatal("expected error when not trained")
	}
}

func TestWhiteningDetector_Detect_WrongDim(t *testing.T) {
	d := NewWhiteningDetector(1e-8)

	compliant := [][]float64{
		{1, 0},
		{0, 1},
	}
	_ = d.Train(compliant, nil)

	_, err := d.Detect([]float64{1, 0, 0})
	if err == nil {
		t.Fatal("expected error for wrong dimension")
	}
}

func TestWhiteningDetector_ComplianceScoreRange(t *testing.T) {
	d := NewWhiteningDetector(1e-8)

	compliant := [][]float64{
		{1, 0},
		{0, 1},
		{1, 1},
		{0.5, 0.5},
	}
	_ = d.Train(compliant, nil)

	// Test multiple activations
	testCases := []struct {
		name       string
		activation []float64
	}{
		{"near_mean", []float64{0.625, 0.625}},
		{"far_from_mean", []float64{10, 10}},
		{"origin", []float64{0, 0}},
	}

	for _, tc := range testCases {
		score, err := d.Detect(tc.activation)
		if err != nil {
			t.Fatalf("Detect(%s) error: %v", tc.name, err)
		}
		if score < 0 || score > 1 {
			t.Errorf("Detect(%s) score = %f, want [0, 1]", tc.name, score)
		}
	}
}

func TestWhiteningDetector_CompliantVsViolating(t *testing.T) {
	d := NewWhiteningDetector(1e-8)

	compliant := [][]float64{
		{0.1, 0.1},
		{0.2, 0.2},
		{0.15, 0.15},
		{0.18, 0.18},
	}

	if err := d.Train(compliant, nil); err != nil {
		t.Fatalf("Train error: %v", err)
	}

	// Near-compliant should score higher than far-violating
	nearScore, _ := d.Detect([]float64{0.15, 0.15})
	farScore, _ := d.Detect([]float64{10, 10})

	if nearScore <= farScore {
		t.Errorf("near-compliant score (%f) should be > far-violating score (%f)", nearScore, farScore)
	}
}

func TestWhiteningDetector_LargeDataset(t *testing.T) {
	d := NewWhiteningDetector(1e-8)

	// Generate 100 compliant samples
	compliant := make([][]float64, 100)
	for i := 0; i < 100; i++ {
		compliant[i] = []float64{
			math.Sin(float64(i) * 0.1),
			math.Cos(float64(i) * 0.1),
		}
	}

	if err := d.Train(compliant, nil); err != nil {
		t.Fatalf("Train error: %v", err)
	}

	score, err := d.Detect([]float64{0.5, 0.5})
	if err != nil {
		t.Fatalf("Detect error: %v", err)
	}
	if score < 0 || score > 1 {
		t.Errorf("score = %f, want [0, 1]", score)
	}
}
