package extract

import (
	"math"
	"testing"
)

func FuzzExtractDIM(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4}, []byte{5, 6, 7, 8})

	f.Fuzz(func(t *testing.T, posData, negData []byte) {
		if len(posData) < 4 || len(negData) < 4 {
			return
		}
		dim := 2
		nPos := len(posData) / (dim * 8)
		nNeg := len(negData) / (dim * 8)
		if nPos < 1 || nNeg < 1 || nPos > 10 || nNeg > 10 {
			return
		}
		positive := make([][]float64, nPos)
		negative := make([][]float64, nNeg)
		for i := 0; i < nPos; i++ {
			positive[i] = make([]float64, dim)
			for j := 0; j < dim; j++ {
				idx := (i*dim + j) * 8
				if idx+8 <= len(posData) {
					bits := uint64(posData[idx]) | uint64(posData[idx+1])<<8 | uint64(posData[idx+2])<<16 | uint64(posData[idx+3])<<24
					positive[i][j] = float64(bits) / 1e6
				}
			}
		}
		for i := 0; i < nNeg; i++ {
			negative[i] = make([]float64, dim)
			for j := 0; j < dim; j++ {
				idx := (i*dim + j) * 8
				if idx+8 <= len(negData) {
					bits := uint64(negData[idx]) | uint64(negData[idx+1])<<8 | uint64(negData[idx+2])<<16 | uint64(negData[idx+3])<<24
					negative[i][j] = float64(bits) / 1e6
				}
			}
		}
		result := ExtractDIM(positive, negative)
		if result != nil {
			if len(result) != dim {
				t.Errorf("expected dim %d, got %d", dim, len(result))
			}
			for _, v := range result {
				if math.IsNaN(v) || math.IsInf(v, 0) {
					t.Errorf("NaN/Inf in result")
				}
			}
		}
	})
}

func FuzzNormalize(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4})

	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) < 4 {
			return
		}
		v := make([]float64, len(data))
		for i := range v {
			v[i] = float64(data[i]) / 255.0
		}
		result := Normalize(v)
		if result != nil {
			// Check normalized (unit norm)
			var norm float64
			for _, x := range result {
				norm += x * x
			}
			norm = math.Sqrt(norm)
			if norm > 0 && math.Abs(norm-1.0) > 1e-6 {
				t.Errorf("not unit norm: %f", norm)
			}
		}
	})
}

func FuzzNorm(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4})

	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) < 1 {
			return
		}
		v := make([]float64, len(data))
		for i := range v {
			v[i] = float64(data[i]) / 255.0
		}
		n := Norm(v)
		if math.IsNaN(n) || math.IsInf(n, 0) {
			t.Errorf("NaN/Inf norm: %f", n)
		}
		if n < 0 {
			t.Errorf("negative norm: %f", n)
		}
	})
}

func FuzzSigmoid(f *testing.F) {
	f.Add(0.0)
	f.Add(1.0)

	f.Fuzz(func(t *testing.T, x float64) {
		if math.IsNaN(x) || math.IsInf(x, 0) || math.Abs(x) > 500 {
			return
		}
		result := Sigmoid(x)
		if math.IsNaN(result) || math.IsInf(result, 0) {
			t.Errorf("NaN/Inf result: %f", result)
		}
		if result < 0 || result > 1 {
			t.Errorf("sigmoid out of [0,1]: %f", result)
		}
	})
}
