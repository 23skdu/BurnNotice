package math

import (
	"math"
	"testing"
)

func FuzzWinsorize(f *testing.F) {
	f.Add(float64(1), float64(2), float64(3), float64(4), float64(5))
	f.Add(float64(0), float64(0), float64(0), float64(0), float64(0))
	f.Add(float64(-1e10), float64(-1), float64(0), float64(1), float64(1e10))
	f.Add(float64(math.NaN()), float64(1), float64(2), float64(3), float64(4))
	f.Add(float64(math.Inf(1)), float64(math.Inf(-1)), float64(0), float64(0), float64(0))

	f.Fuzz(func(t *testing.T, a, b, c, d, e float64) {
		input := []float64{a, b, c, d, e}

		// Winsorize should never panic
		result := Winsorize(input, 0.1, 0.9)
		if len(result) != len(input) {
			t.Fatalf("len(result) = %d, want %d", len(result), len(input))
		}

		// Every output value must be non-NaN and non-Inf
		for i, v := range result {
			if math.IsNaN(v) {
				t.Errorf("result[%d] is NaN", i)
			}
			if math.IsInf(v, 0) {
				t.Errorf("result[%d] is Inf", i)
			}
		}
	})
}

func FuzzWinsorizeInPlace(f *testing.F) {
	f.Add(float64(1), float64(2), float64(3), float64(4), float64(5))
	f.Add(float64(-1e30), float64(-1), float64(0), float64(1), float64(1e30))
	f.Add(float64(math.NaN()), float64(math.NaN()), float64(math.NaN()), float64(math.NaN()), float64(math.NaN()))

	f.Fuzz(func(t *testing.T, a, b, c, d, e float64) {
		input := []float64{a, b, c, d, e}

		// Must not panic
		WinsorizeInPlace(input, 0.1, 0.9)

		// In-place should not produce NaN/Inf from valid inputs
		// (NaN in -> NaN out is acceptable for in-place)
	})
}

func FuzzIsStable(f *testing.F) {
	f.Add(float64(1), float64(2), float64(3))
	f.Add(float64(math.NaN()), float64(2), float64(3))
	f.Add(float64(math.Inf(1)), float64(2), float64(3))
	f.Add(float64(0), float64(0), float64(0))

	f.Fuzz(func(t *testing.T, a, b, c float64) {
		v := []float64{a, b, c}
		stable := IsStable(v)
		for _, x := range v {
			if math.IsNaN(x) || math.IsInf(x, 0) {
				if stable {
					t.Errorf("IsStable(%v) = true, expected false (contains NaN/Inf)", v)
				}
				return
			}
		}
		if !stable {
			t.Errorf("IsStable(%v) = false, expected true (all finite)", v)
		}
	})
}

func FuzzClampToFloat64(f *testing.F) {
	f.Add(float64(1), float64(2), float64(3))
	f.Add(float64(math.NaN()), float64(math.Inf(1)), float64(math.Inf(-1)))
	f.Add(float64(-1e308), float64(0), float64(1e308))

	f.Fuzz(func(t *testing.T, a, b, c float64) {
		input := []float64{a, b, c}
		result := ClampToFloat64(input)
		if len(result) != len(input) {
			t.Fatalf("len(result) = %d, want %d", len(result), len(input))
		}
		for i, v := range result {
			if math.IsNaN(v) {
				t.Errorf("result[%d] is NaN after clamp", i)
			}
			if math.IsInf(v, 0) {
				t.Errorf("result[%d] is Inf after clamp", i)
			}
			if math.IsNaN(input[i]) && v != 0 {
				t.Errorf("NaN clamped to %f, want 0", v)
			}
		}
	})
}
