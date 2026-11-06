package native

import (
	"math"
	"testing"
)

func TestGPUEngine_CosineSimilarity(t *testing.T) {
	ge := NewGPUEngine(3)

	act := []float32{1.0, 0.0, 0.0}
	dirs := []float32{
		1.0, 0.0, 0.0, // perfect alignment
		0.0, 1.0, 0.0, // orthogonal
	}

	scores := ge.CosineSimilarity(act, dirs, 2)
	if len(scores) != 2 {
		t.Fatalf("expected 2 scores, got %d", len(scores))
	}

	// First score should be higher than second score
	if scores[0] <= scores[1] {
		t.Errorf("expected aligned score to be greater than orthogonal score, got %v", scores)
	}
}

func TestGPUEngine_DynamicRotation(t *testing.T) {
	ge := NewGPUEngine(2)

	act := []float32{1.0, 0.0}
	dir := []float32{0.0, 1.0} // orthogonal direction

	// Rotate by 90 degrees (pi/2 radians)
	rotated := ge.DynamicRotation(act, dir, math.Pi/2.0)
	if len(rotated) != 2 {
		t.Fatalf("expected rotated dim 2, got %d", len(rotated))
	}

	// After 90 degree rotation from [1, 0] along [0, 1], we expect the vector to be rotated close to [0, -1] or similar norm-preserving coordinates
	normSq := rotated[0]*rotated[0] + rotated[1]*rotated[1]
	if math.Abs(float64(normSq)-1.0) > 1e-5 {
		t.Errorf("expected norm-preserving rotation, got vector %v with norm squared %f", rotated, normSq)
	}
}

func TestGPUEngine_CosineSimilarity_EdgeCases(t *testing.T) {
	ge := NewGPUEngine(3)

	// Case 1: len(activation) != ge.Dimension
	if res := ge.CosineSimilarity([]float32{1.0, 0.0}, []float32{1, 0, 0, 0, 1, 0}, 2); res != nil {
		t.Errorf("expected nil for mismatched activation dimension, got %v", res)
	}

	// Case 2: len(directions) != ge.Dimension*numDirections
	if res := ge.CosineSimilarity([]float32{1.0, 0.0, 0.0}, []float32{1, 0, 0}, 2); res != nil {
		t.Errorf("expected nil for mismatched directions dimension, got %v", res)
	}

	// Case 3: numDirections <= 0
	if res := ge.CosineSimilarity([]float32{1.0, 0.0, 0.0}, []float32{}, 0); res != nil {
		t.Errorf("expected nil for non-positive numDirections, got %v", res)
	}
}

func TestGPUEngine_DynamicRotation_EdgeCases(t *testing.T) {
	ge := NewGPUEngine(2)

	// Case 1: len(activation) != ge.Dimension
	if res := ge.DynamicRotation([]float32{1.0}, []float32{0.0, 1.0}, 1.0); res != nil {
		t.Errorf("expected nil for mismatched activation dimension, got %v", res)
	}

	// Case 2: len(direction) != ge.Dimension
	if res := ge.DynamicRotation([]float32{1.0, 0.0}, []float32{0.0}, 1.0); res != nil {
		t.Errorf("expected nil for mismatched direction dimension, got %v", res)
	}
}
