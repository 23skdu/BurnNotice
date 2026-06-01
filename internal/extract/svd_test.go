package extract

import (
	"math"
	"testing"
)

func TestExtractWhitenedSVD(t *testing.T) {
	// Simple synthetic data
	positive := [][]float64{
		{1.0, 2.0, 0.5},
		{1.2, 2.1, 0.6},
		{0.9, 1.9, 0.4},
		{1.1, 2.2, 0.5},
	}
	
	negative := [][]float64{
		{-1.0, -2.0, -0.5},
		{-1.2, -2.1, -0.6},
		{-0.9, -1.9, -0.4},
		{-1.1, -2.2, -0.5},
	}

	// Test extraction with k=2, alpha=0.1
	directions, singularValues := ExtractWhitenedSVD(positive, negative, 2, 0.1)

	if len(directions) != 2 {
		t.Fatalf("Expected 2 directions, got %d", len(directions))
	}
	if len(singularValues) != 2 {
		t.Fatalf("Expected 2 singular values, got %d", len(singularValues))
	}

	// Ensure directions are normalized
	for i, dir := range directions {
		norm := 0.0
		for _, v := range dir {
			norm += v * v
		}
		norm = math.Sqrt(norm)
		if math.Abs(norm-1.0) > 1e-5 {
			t.Errorf("Direction %d is not normalized (norm = %f)", i, norm)
		}
	}

	// Singular values should be sorted descending
	if singularValues[0] < singularValues[1] {
		t.Errorf("Singular values not sorted: %f, %f", singularValues[0], singularValues[1])
	}
}
