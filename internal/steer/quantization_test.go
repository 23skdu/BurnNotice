package steer

import (
	"testing"
)

func TestQuantizationMapper_CalibrateScale(t *testing.T) {
	qm := NewQuantizationMapper()

	fp16 := [][]float64{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
	}
	quant := [][]float64{
		{0.5, 1.0, 1.5},
		{2.0, 2.5, 3.0},
	}

	scale := qm.CalibrateScale(fp16, quant)
	if scale < 1.9 || scale > 2.1 {
		t.Errorf("expected calibration scale around 2.0, got %f", scale)
	}
}

func TestQuantizationMapper_TransferDirection(t *testing.T) {
	qm := NewQuantizationMapper()

	dirFP16 := []float64{1.0, 0.0, 0.0}
	dirQuant := []float64{0.99, 0.01, 0.0} // close similarity

	t.Run("Valid Alignment", func(t *testing.T) {
		trans, err := qm.TransferDirection(dirFP16, 1.0, dirQuant)
		if err != nil {
			t.Fatalf("unexpected transfer failure: %v", err)
		}
		if len(trans) != 3 {
			t.Errorf("expected vector size 3")
		}
	})

	t.Run("Similarity Failure", func(t *testing.T) {
		badQuant := []float64{0.5, 0.5, 0.5} // distinct direction
		_, err := qm.TransferDirection(dirFP16, 1.0, badQuant)
		if err == nil {
			t.Errorf("expected alignment error for low similarity vector, got nil")
		}
	})
}
