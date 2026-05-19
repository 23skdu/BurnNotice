package engine

import (
	"testing"
)

func TestNewMoEEngine(t *testing.T) {
	e := NewMoEEngine(4, 2, []int{128, 128, 128, 128})

	if e.NumExperts() != 4 {
		t.Errorf("NumExperts = %d, want 4", e.NumExperts())
	}
}

func TestNewMoEEngine_Panic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for mismatched expertDims")
		}
	}()
	NewMoEEngine(4, 2, []int{128, 128})
}

func TestMoEEngine_DetectRouting(t *testing.T) {
	e := NewMoEEngine(4, 2, []int{128, 128, 128, 128})

	_, weights := e.DetectRouting(0, make([]float64, 128))

	if len(weights) != 2 {
		t.Fatalf("len(weights) = %d, want 2", len(weights))
	}

	// Check weights sum to ~1 (top-K of uniform)
	var sum float64
	for _, w := range weights {
		sum += w
	}
	if sum < 0.4 || sum > 0.6 {
		t.Errorf("weights sum = %f, want ~0.5", sum)
	}
}

func TestMoEEngine_DetectRouting_Uniform(t *testing.T) {
	e := NewMoEEngine(3, 2, []int{64, 64, 64})

	_, weights := e.DetectRouting(0, make([]float64, 64))

	// With uniform routing, top-2 should have equal weights
	if weights[0] != weights[1] {
		t.Errorf("uniform routing weights should be equal: %f != %f", weights[0], weights[1])
	}
}

func TestMoEEngine_ExpertActivation(t *testing.T) {
	e := NewMoEEngine(2, 1, []int{64, 128})

	activation := make([]float64, 200)
	for i := range activation {
		activation[i] = float64(i)
	}

	exp0 := e.ExpertActivation(activation, 0)
	if len(exp0) != 64 {
		t.Errorf("expert 0 activation len = %d, want 64", len(exp0))
	}

	exp1 := e.ExpertActivation(activation, 1)
	if len(exp1) != 128 {
		t.Errorf("expert 1 activation len = %d, want 128", len(exp1))
	}
}

func TestMoEEngine_ExpertActivation_InvalidIndex(t *testing.T) {
	e := NewMoEEngine(2, 1, []int{64, 128})

	result := e.ExpertActivation(make([]float64, 200), 5)
	if result != nil {
		t.Error("should return nil for invalid expert index")
	}
}

func TestMoEEngine_ExpertActivation_ShorterDim(t *testing.T) {
	e := NewMoEEngine(1, 1, []int{256})

	activation := make([]float64, 100)
	result := e.ExpertActivation(activation, 0)

	if len(result) != 100 {
		t.Errorf("result len = %d, want 100 (clipped to activation len)", len(result))
	}
}

func TestMoEEngine_ExpertDim(t *testing.T) {
	e := NewMoEEngine(3, 2, []int{64, 128, 256})

	if e.ExpertDim(0) != 64 {
		t.Errorf("ExpertDim(0) = %d, want 64", e.ExpertDim(0))
	}
	if e.ExpertDim(1) != 128 {
		t.Errorf("ExpertDim(1) = %d, want 128", e.ExpertDim(1))
	}
	if e.ExpertDim(2) != 256 {
		t.Errorf("ExpertDim(2) = %d, want 256", e.ExpertDim(2))
	}
	if e.ExpertDim(99) != 0 {
		t.Errorf("ExpertDim(99) = %d, want 0", e.ExpertDim(99))
	}
}

func TestMoEEngine_UpdateRouting(t *testing.T) {
	e := NewMoEEngine(3, 2, []int{64, 64, 64})

	e.UpdateRouting(0, []float64{0.5, 0.3, 0.2})

	indices, weights := e.DetectRouting(0, make([]float64, 64))

	if indices[0] != 0 {
		t.Errorf("top expert = %d, want 0", indices[0])
	}
	if weights[0] != 0.5 {
		t.Errorf("top weight = %f, want 0.5", weights[0])
	}
}

func TestMoEEngine_UpdateRouting_WrongLength(t *testing.T) {
	e := NewMoEEngine(3, 2, []int{64, 64, 64})

	e.UpdateRouting(0, []float64{0.5, 0.5}) // wrong length

	// Should fall back to uniform
	_, weights := e.DetectRouting(0, make([]float64, 64))
	expected := 1.0 / 3.0
	for _, w := range weights {
		if w != expected {
			t.Errorf("weight = %f, want %f (uniform)", w, expected)
		}
	}
}
