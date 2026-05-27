package engine

import (
	"testing"
)

func TestNewMoEEngine(t *testing.T) {
	e := NewMoEEngine(4, 2, []int{128, 128, 128, 128}, nil)

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
	NewMoEEngine(4, 2, []int{128, 128}, nil)
}

func TestMoEEngine_DetectRouting(t *testing.T) {
	e := NewMoEEngine(4, 2, []int{128, 128, 128, 128}, nil)

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
	e := NewMoEEngine(3, 2, []int{64, 64, 64}, nil)

	_, weights := e.DetectRouting(0, make([]float64, 64))

	// With uniform routing, top-2 should have equal weights
	if weights[0] != weights[1] {
		t.Errorf("uniform routing weights should be equal: %f != %f", weights[0], weights[1])
	}
}

func TestMoEEngine_ExpertActivation(t *testing.T) {
	e := NewMoEEngine(2, 1, []int{64, 128}, nil)

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
	e := NewMoEEngine(2, 1, []int{64, 128}, nil)

	result := e.ExpertActivation(make([]float64, 200), 5)
	if result != nil {
		t.Error("should return nil for invalid expert index")
	}
}

func TestMoEEngine_ExpertActivation_ShorterDim(t *testing.T) {
	e := NewMoEEngine(1, 1, []int{256}, nil)

	activation := make([]float64, 100)
	result := e.ExpertActivation(activation, 0)

	if len(result) != 100 {
		t.Errorf("result len = %d, want 100 (clipped to activation len)", len(result))
	}
}

func TestMoEEngine_ExpertDim(t *testing.T) {
	e := NewMoEEngine(3, 2, []int{64, 128, 256}, nil)

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

func TestMoEEngine_DetectRouting_Dynamic(t *testing.T) {
	e := NewMoEEngine(3, 1, []int{64, 64, 64}, nil)

	activation := make([]float64, 192)
	// Make expert 0 highly active
	for i := 0; i < 64; i++ {
		activation[i] = 10.0
	}
	// Make expert 1 lightly active
	for i := 64; i < 128; i++ {
		activation[i] = 1.0
	}
	// Expert 2 inactive (0.0)

	indices, weights := e.DetectRouting(0, activation)

	if len(indices) != 1 {
		t.Fatalf("len(indices) = %d, want 1", len(indices))
	}

	if indices[0] != 0 {
		t.Errorf("top expert = %d, want 0", indices[0])
	}
	// The weights are normalized sums of squares
	// SumSq0 = 64 * 100 = 6400
	// SumSq1 = 64 * 1 = 64
	// SumSq2 = 0
	// Total = 6464
	// Weight0 = 6400 / 6464 = 0.990099
	if weights[0] < 0.98 || weights[0] > 1.0 {
		t.Errorf("top weight = %f, want ~0.99", weights[0])
	}
}

func TestMoEEngine_DetectRouting_FallbackUniform(t *testing.T) {
	e := NewMoEEngine(3, 2, []int{64, 64, 64}, nil)

	// All zeros activation
	activation := make([]float64, 192)

	_, weights := e.DetectRouting(0, activation)

	// Should fall back to uniform
	expected := 1.0 / 3.0
	for _, w := range weights {
		if w != expected {
			t.Errorf("weight = %f, want %f (uniform)", w, expected)
		}
	}
}

func TestMoEEngine_UpdateRouting(t *testing.T) {
	e := NewMoEEngine(3, 2, []int{64, 64, 64}, nil)

	// Mismatched length should return early
	e.UpdateRouting(1, []float64{0.1, 0.2})
	if _, ok := e.routingCache[1]; ok {
		t.Error("routingCache should not have been updated with mismatched weights")
	}

	// Correct length should succeed
	weights := []float64{0.2, 0.5, 0.3}
	e.UpdateRouting(1, weights)
	cached, ok := e.routingCache[1]
	if !ok {
		t.Fatal("expected cached weights for layer 1")
	}
	if len(cached) != 3 || cached[0] != 0.2 || cached[1] != 0.5 || cached[2] != 0.3 {
		t.Errorf("got cached weights %v, want %v", cached, weights)
	}
}

func TestMoEEngine_DetectRouting_ShortActivation(t *testing.T) {
	// ExpertDims is 64, 64, 64. Sum is 192.
	// We pass an activation of length 100.
	// For idx = 0: dim = 64, end = 64.
	// For idx = 1: dim = 64, start = 64, end = 128 > 100 -> end is clipped to 100.
	// For idx = 2: dim = 64, start = 100, end = 164 > 100 -> end is clipped to 100.
	e := NewMoEEngine(3, 2, []int{64, 64, 64}, nil)

	activation := make([]float64, 100)
	for i := 0; i < 100; i++ {
		activation[i] = 1.0
	}

	indices, weights := e.DetectRouting(0, activation)
	if len(indices) != 2 {
		t.Fatalf("expected 2 expert indices, got %d", len(indices))
	}
	// Expert 0 score: 64
	// Expert 1 score: 36 (100 - 64)
	// Expert 2 score: 0
	// Total: 100
	if indices[0] != 0 || indices[1] != 1 {
		t.Errorf("got active indices %v, want [0, 1]", indices)
	}
	if weights[0] != 0.64 || weights[1] != 0.36 {
		t.Errorf("got weights %v, want [0.64, 0.36]", weights)
	}
}

func TestMoEEngine_DetectRouting_TopKGreaterThanNumExperts(t *testing.T) {
	// e.topK = 5, but numExperts = 3
	e := NewMoEEngine(3, 5, []int{64, 64, 64}, nil)

	activation := make([]float64, 192)
	for i := 0; i < 64; i++ {
		activation[i] = 1.0 // Expert 0 active
	}

	indices, weights := e.DetectRouting(0, activation)
	// topK should be capped at numExperts (3)
	if len(indices) != 3 {
		t.Fatalf("expected capped topK of 3, got %d", len(indices))
	}
	if len(weights) != 3 {
		t.Fatalf("expected capped weights length of 3, got %d", len(weights))
	}
}

func TestMoEEngine_DetectRouting_Swap(t *testing.T) {
	e := NewMoEEngine(3, 2, []int{64, 64, 64}, nil)

	activation := make([]float64, 192)
	// Expert 0 score: 0
	// Make expert 1 lightly active (score 64)
	for i := 64; i < 128; i++ {
		activation[i] = 1.0
	}
	// Make expert 2 highly active (score 6400)
	for i := 128; i < 192; i++ {
		activation[i] = 10.0
	}

	indices, weights := e.DetectRouting(0, activation)
	// Expert 2 should be index 0 (highly active)
	// Expert 1 should be index 1 (lightly active)
	if indices[0] != 2 || indices[1] != 1 {
		t.Errorf("got active indices %v, want [2, 1]", indices)
	}
	if weights[0] <= weights[1] {
		t.Errorf("expected weight for index 2 to be greater than index 1, got %v", weights)
	}
}



