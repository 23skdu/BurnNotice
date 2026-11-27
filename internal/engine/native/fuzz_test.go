package native

import (
	"math"
	"testing"
)

func FuzzGPUEngineCosineSimilarity(f *testing.F) {
	f.Add([]byte{1, 0, 0}, 2)

	f.Fuzz(func(t *testing.T, data []byte, numDirs int) {
		if len(data) < 3 || numDirs < 1 || numDirs > 10 {
			return
		}
		dim := 3
		if len(data) < dim*numDirs+dim {
			return
		}
		act := make([]float32, dim)
		for i := 0; i < dim; i++ {
			act[i] = float32(data[i]) / 255.0
		}
		dirs := make([]float32, dim*numDirs)
		for i := 0; i < dim*numDirs; i++ {
			dirs[i] = float32(data[dim+i]) / 255.0
		}
		ge := NewGPUEngine(dim)
		scores := ge.CosineSimilarity(act, dirs, numDirs)
		if scores == nil {
			return
		}
		if len(scores) != numDirs {
			t.Errorf("expected %d scores, got %d", numDirs, len(scores))
		}
		for _, s := range scores {
			if math.IsNaN(float64(s)) || math.IsInf(float64(s), 0) {
				t.Errorf("NaN/Inf score: %f", s)
			}
			if s < 0 || s > 1 {
				t.Errorf("score out of [0,1]: %f", s)
			}
		}
	})
}

func FuzzGPUEngineDynamicRotation(f *testing.F) {
	f.Add([]byte{1, 0}, 1.57)

	f.Fuzz(func(t *testing.T, data []byte, theta float64) {
		if len(data) < 4 || math.IsNaN(theta) || math.IsInf(theta, 0) {
			return
		}
		dim := 2
		act := make([]float32, dim)
		dir := make([]float32, dim)
		for i := 0; i < dim; i++ {
			act[i] = float32(data[i]) / 255.0
			dir[i] = float32(data[dim+i]) / 255.0
		}
		// Skip near-zero activation or direction vectors
		var actNormSq, dirNormSq float64
		for _, v := range act {
			actNormSq += float64(v) * float64(v)
		}
		for _, v := range dir {
			dirNormSq += float64(v) * float64(v)
		}
		if actNormSq < 1e-6 || dirNormSq < 1e-6 {
			return
		}
		ge := NewGPUEngine(dim)
		rotated := ge.DynamicRotation(act, dir, float32(theta))
		if rotated == nil {
			return
		}
		if len(rotated) != dim {
			t.Errorf("expected dim %d, got %d", dim, len(rotated))
		}
		// Check no NaN/Inf
		for _, v := range rotated {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Errorf("NaN/Inf in rotated: %f", v)
			}
		}
	})
}

func FuzzGPUEngineEdgeCases(f *testing.F) {
	f.Add(0, 1)
	f.Add(3, 0)

	f.Fuzz(func(t *testing.T, dim, numDirs int) {
		if dim < 0 || dim > 10 || numDirs < 0 || numDirs > 10 {
			return
		}
		ge := NewGPUEngine(dim)
		// Zero-dimension should return nil
		if dim == 0 {
			if res := ge.CosineSimilarity([]float32{}, []float32{}, numDirs); res != nil && numDirs > 0 {
				t.Error("expected nil for zero dimension")
			}
			if res := ge.DynamicRotation([]float32{}, []float32{}, 1.0); res != nil {
				t.Error("expected nil for zero dimension rotation")
			}
		}
		// Mismatched dimensions
		if dim > 1 {
			if res := ge.CosineSimilarity([]float32{1}, make([]float32, dim*numDirs), numDirs); res != nil {
				t.Error("expected nil for mismatched activation")
			}
		}
	})
}
