package policydetect

import (
	"math"
	"testing"
)

func FuzzWhiteningDetect(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4}, 1e-8)
	f.Add([]byte{0, 0, 0, 0}, 1e-6)

	f.Fuzz(func(t *testing.T, data []byte, epsilon float64) {
		if epsilon <= 0 || math.IsNaN(epsilon) || math.IsInf(epsilon, 0) {
			return
		}
		if len(data) < 4 {
			return
		}
		dim := len(data) / 2
		if dim < 1 {
			return
		}
		// Build activation from data
		act := make([]float64, dim)
		for i := 0; i < dim; i++ {
			act[i] = float64(data[i]) / 255.0
		}
		det := NewWhiteningDetector(epsilon)
		// Train with simple data
		compliant := [][]float64{act}
		violating := [][]float64{make([]float64, dim)}
		_ = det.Train(compliant, violating)
		if det.IsTrained() {
			score, err := det.Detect(act)
			if err != nil {
				return
			}
			if math.IsNaN(score) || math.IsInf(score, 0) {
				t.Errorf("NaN/Inf score: %f", score)
			}
			if score < 0 {
				t.Errorf("negative score: %f", score)
			}
		}
	})
}

func FuzzWhiteningTrain(f *testing.F) {
	f.Add(2, 2, 1e-8)

	f.Fuzz(func(t *testing.T, nCompliant, nViolating int, epsilon float64) {
		if epsilon <= 0 || math.IsNaN(epsilon) || math.IsInf(epsilon, 0) {
			return
		}
		if nCompliant < 1 || nCompliant > 10 || nViolating < 1 || nViolating > 10 {
			return
		}
		dim := 4
		compliant := make([][]float64, nCompliant)
		violating := make([][]float64, nViolating)
		for i := 0; i < nCompliant; i++ {
			compliant[i] = make([]float64, dim)
			for j := 0; j < dim; j++ {
				compliant[i][j] = float64(i+j) / 10.0
			}
		}
		for i := 0; i < nViolating; i++ {
			violating[i] = make([]float64, dim)
			for j := 0; j < dim; j++ {
				violating[i][j] = float64(i-j) / 10.0
			}
		}
		det := NewWhiteningDetector(epsilon)
		err := det.Train(compliant, violating)
		if err != nil {
			return
		}
		if !det.IsTrained() {
			t.Error("expected trained after successful Train")
		}
		if det.Dimension() != dim {
			t.Errorf("Dimension = %d, want %d", det.Dimension(), dim)
		}
	})
}
