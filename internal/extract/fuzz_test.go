package extract

import (
	"math"
	"testing"
)

func FuzzExtractDIM(f *testing.F) {
	f.Add(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
	f.Add(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
	f.Add(1.0, 2.0, 3.0, -1.0, -2.0, -3.0)
	f.Add(1e-10, 1e-10, 1e-10, -1e-10, -1e-10, -1e-10)

	f.Fuzz(func(t *testing.T, a1, a2, a3, b1, b2, b3 float64) {
		for _, v := range []float64{a1, a2, a3, b1, b2, b3} {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Skip()
			}
		}

		positive := [][]float64{{a1, a2, a3}}
		negative := [][]float64{{b1, b2, b3}}

		result := ExtractDIM(positive, negative)

		if result == nil {
			t.Fatal("ExtractDIM returned nil")
		}
		if len(result) != 3 {
			t.Fatalf("len(result) = %d, want 3", len(result))
		}

		norm := Norm(result)
		if norm > 1e-10 && math.Abs(norm-1.0) > 1e-6 {
			t.Errorf("result norm = %f, want 1.0", norm)
		}
	})
}

func FuzzNormalize(f *testing.F) {
	f.Add(1.0, 2.0, 3.0)
	f.Add(0.0, 0.0, 0.0)
	f.Add(-1.0, -2.0, -3.0)
	f.Add(1e-10, 1e-10, 1e-10)
	f.Add(1e10, 1e10, 1e10)

	f.Fuzz(func(t *testing.T, x, y, z float64) {
		for _, v := range []float64{x, y, z} {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Skip()
			}
		}

		vec := []float64{x, y, z}
		result := Normalize(vec)

		if len(result) != 3 {
			t.Fatalf("len(result) = %d, want 3", len(result))
		}

		norm := Norm(result)
		if norm > 1e-10 && math.Abs(norm-1.0) > 1e-6 {
			t.Errorf("Normalize(%v) norm = %f, want 1.0", vec, norm)
		}
	})
}

func FuzzSigmoid(f *testing.F) {
	f.Add(0.0)
	f.Add(1.0)
	f.Add(-1.0)
	f.Add(100.0)
	f.Add(-100.0)

	f.Fuzz(func(t *testing.T, x float64) {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			t.Skip()
		}

		result := Sigmoid(x)

		if result < 0 || result > 1 {
			t.Errorf("Sigmoid(%f) = %f, want [0, 1]", x, result)
		}

		if x == 0 && math.Abs(result-0.5) > 1e-9 {
			t.Errorf("Sigmoid(0) = %f, want 0.5", result)
		}

		if x > 20 && result != 1.0 {
			t.Errorf("Sigmoid(%f) = %f, want 1.0", x, result)
		}

		if x < -20 && result != 0.0 {
			t.Errorf("Sigmoid(%f) = %f, want 0.0", x, result)
		}
	})
}

func FuzzDot(f *testing.F) {
	f.Add(1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
	f.Add(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

	f.Fuzz(func(t *testing.T, a1, a2, a3, b1, b2, b3 float64) {
		for _, v := range []float64{a1, a2, a3, b1, b2, b3} {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Skip()
			}
		}

		a := []float64{a1, a2, a3}
		b := []float64{b1, b2, b3}

		result := Dot(a, b)
		expected := a1*b1 + a2*b2 + a3*b3

		if math.IsNaN(result) || math.IsInf(result, 0) {
			t.Skip()
		}

		if math.Abs(result-expected) > 1e-6 {
			t.Errorf("Dot(%v, %v) = %f, want %f", a, b, result, expected)
		}
	})
}
