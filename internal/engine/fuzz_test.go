package engine

import (
	"encoding/binary"
	"math"
	"testing"
)

func FuzzMoEDetectRouting(f *testing.F) {
	f.Add([]byte{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 2, 3)

	f.Fuzz(func(t *testing.T, data []byte, topK, numExperts int) {
		if len(data) < 8 || topK < 1 || numExperts < 1 || topK > numExperts || numExperts > 10 {
			return
		}
		dim := len(data) / 8
		if dim > 20 || dim < 1 {
			return
		}
		activation := make([]float64, dim)
		for i := 0; i < dim && i*8+8 <= len(data); i++ {
			bits := binary.LittleEndian.Uint64(data[i*8:])
			activation[i] = math.Float64frombits(bits)
			if math.IsNaN(activation[i]) || math.IsInf(activation[i], 0) || math.Abs(activation[i]) > 1e6 {
				return
			}
		}
		expertDims := make([]int, numExperts)
		for i := range expertDims {
			expertDims[i] = dim
		}
		e := NewMoEEngine(numExperts, topK, expertDims, nil)
		indices, weights := e.DetectRouting(0, activation)
		if indices == nil {
			return
		}
		_ = indices
		_ = weights
		var sum float64
		for _, w := range weights {
			if math.IsNaN(w) || math.IsInf(w, 0) {
				t.Errorf("NaN/Inf weight")
			}
			sum += w
		}
		if len(weights) > 0 {
			if sum > 1.0+1e-9 {
				t.Errorf("weights sum = %f > 1.0", sum)
			}
			for _, w := range weights {
				if w < 0 || w > 1 {
					t.Errorf("weight out of [0,1]: %f", w)
				}
			}
		}
	})
}

func FuzzMoEDetectRoutingFromLogits(f *testing.F) {
	f.Add([]byte{0, 0, 0, 0, 0, 0, 240, 63, 0, 0, 0, 0, 0, 0, 0, 64}, 2)

	f.Fuzz(func(t *testing.T, data []byte, topK int) {
		if len(data) < 8 || topK < 1 {
			return
		}
		n := len(data) / 8
		if n > 20 || n < 1 || topK > n {
			return
		}
		logits := make([]float64, n)
		for i := 0; i < n; i++ {
			bits := binary.LittleEndian.Uint64(data[i*8:])
			logits[i] = math.Float64frombits(bits)
			if math.IsNaN(logits[i]) || math.IsInf(logits[i], 0) {
				return
			}
		}
		expertDims := make([]int, n)
		for i := range expertDims {
			expertDims[i] = 1
		}
		e := NewMoEEngine(n, topK, expertDims, nil)
		indices, weights := e.DetectRoutingFromLogits(logits)
		_ = weights
		if indices == nil {
			return
		}
		if len(indices) != topK {
			t.Errorf("expected %d indices, got %d", topK, len(indices))
		}
	})
}

func FuzzValidateTensorElements(f *testing.F) {
	f.Add(6, []byte{2, 3})

	f.Fuzz(func(t *testing.T, elemCount int, shapeBytes []byte) {
		if elemCount < 0 || elemCount > 1000 {
			return
		}
		if len(shapeBytes) == 0 || len(shapeBytes) > 5 {
			return
		}
		shape := make([]int64, len(shapeBytes))
		for i, b := range shapeBytes {
			if b == 0 {
				return
			}
			shape[i] = int64(b)
		}
		err := validateTensorElements(elemCount, shape)
		_ = err
	})
}

func FuzzExpertActivation(f *testing.F) {
	f.Add([]byte{0, 0, 0, 0, 0, 0, 240, 63}, 0)

	f.Fuzz(func(t *testing.T, data []byte, expertIdx int) {
		if len(data) < 8 {
			return
		}
		dim := len(data) / 8
		if dim > 20 || dim < 1 {
			return
		}
		activation := make([]float64, dim)
		for i := 0; i < dim; i++ {
			bits := binary.LittleEndian.Uint64(data[i*8:])
			activation[i] = math.Float64frombits(bits)
			if math.IsNaN(activation[i]) || math.IsInf(activation[i], 0) || math.Abs(activation[i]) > 1e6 {
				return
			}
		}
		expertDims := []int{dim}
		e := NewMoEEngine(1, 1, expertDims, nil)
		result := e.ExpertActivation(activation, 0)
		if result != nil && len(result) != dim {
			t.Errorf("expected len %d, got %d", dim, len(result))
		}
	})
}
