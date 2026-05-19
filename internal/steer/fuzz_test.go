package steer

import (
	"math"
	"testing"
)

func FuzzActAdd(f *testing.F) {
	f.Add(1.0, 2.0, 3.0, 0.5, 1.5)
	f.Add(0.0, 0.0, 0.0, 0.0, 0.0)
	f.Add(-1.0, -2.0, -3.0, -0.5, -1.5)

	f.Fuzz(func(t *testing.T, a1, a2, a3, d1, alpha float64) {
		activation := []float64{a1, a2, a3}
		direction := []float64{d1, 1.0, 0.0}

		result := ActAdd(activation, direction, alpha)

		if len(result) != len(activation) {
			t.Fatalf("len(result) = %d, want %d", len(result), len(activation))
		}

		for i := range result {
			expected := activation[i] + alpha*direction[i]
			if math.IsInf(result[i], 0) || math.IsNaN(result[i]) {
				t.Errorf("result[%d] = %f (invalid)", i, result[i])
			}
			_ = expected
		}
	})
}

func FuzzAngular(f *testing.F) {
	f.Add(1.0, 2.0, 3.0, 0.5, 0.1)
	f.Add(0.0, 0.0, 0.0, 1.0, 0.5)
	f.Add(1.0, 0.0, 0.0, 1.0, 0.0)

	f.Fuzz(func(t *testing.T, a1, a2, a3, d1, alpha float64) {
		activation := []float64{a1, a2, a3}
		direction := []float64{d1, 1.0, 0.0}

		result := Angular(activation, direction, alpha)

		if len(result) != len(activation) {
			t.Fatalf("len(result) = %d, want %d", len(result), len(activation))
		}

		inputNorm := vecNorm(activation)
		outputNorm := vecNorm(result)

		if inputNorm > 1e-10 && outputNorm > 1e-10 {
			ratio := outputNorm / inputNorm
			if math.Abs(ratio-1.0) > 1e-5 {
				t.Errorf("norm ratio = %f, want ~1.0", ratio)
			}
		}

		for i := range result {
			if math.IsInf(result[i], 0) || math.IsNaN(result[i]) {
				t.Errorf("result[%d] = %f (invalid)", i, result[i])
			}
		}
	})
}

func FuzzAblate(f *testing.F) {
	f.Add(3.0, 4.0, 5.0, 1.0, 0.0)
	f.Add(0.0, 0.0, 0.0, 0.0, 0.0)
	f.Add(1.0, 1.0, 1.0, 1.0, 1.0)

	f.Fuzz(func(t *testing.T, a1, a2, a3, d1, d2 float64) {
		activation := []float64{a1, a2, a3}
		direction := []float64{d1, d2, 1.0}

		result := Ablate(activation, direction)

		if len(result) != len(activation) {
			t.Fatalf("len(result) = %d, want %d", len(result), len(activation))
		}

		dot := vecDot(result, direction)
		if math.Abs(dot) > 1e-6 {
			t.Errorf("result not orthogonal to direction: dot = %f", dot)
		}

		for i := range result {
			if math.IsInf(result[i], 0) || math.IsNaN(result[i]) {
				t.Errorf("result[%d] = %f (invalid)", i, result[i])
			}
		}
	})
}

func FuzzDispatcher(f *testing.F) {
	f.Add(0, 1.0, 2.0, 3.0, 0.5, 1.0)
	f.Add(1, 1.0, 0.0, 0.0, 1.0, 0.5)
	f.Add(2, 3.0, 4.0, 5.0, 1.0, 0.0)

	f.Fuzz(func(t *testing.T, layer int, a1, a2, a3, d1, strength float64) {
		layers := []int{0, 1, 2, 3}
		methods := map[int]Method{0: MethodAddition, 1: MethodAngular, 2: MethodAblation}
		d := NewDispatcher(layers, methods)

		activation := []float64{a1, a2, a3}
		direction := []float64{d1, 1.0, 0.0}

		result := d.Apply(layer, activation, direction, strength)

		if len(result) != len(activation) {
			t.Fatalf("len(result) = %d, want %d", len(result), len(activation))
		}

		for i := range result {
			if math.IsInf(result[i], 0) || math.IsNaN(result[i]) {
				t.Errorf("result[%d] = %f (invalid)", i, result[i])
			}
		}
	})
}
