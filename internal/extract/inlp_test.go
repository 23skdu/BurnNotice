package extract

import (
	"math"
	"testing"
)

func TestExtractINLP(t *testing.T) {
	pos := [][]float64{
		{2.0, 1.0, 0.5},
		{2.0, 2.0, 1.0},
		{3.0, 1.5, 0.8},
	}
	neg := [][]float64{
		{-1.0, 0.0, 0.0},
		{-1.0, -1.0, -0.5},
		{-2.0, -0.5, -0.3},
	}

	t.Run("Extract Two Directions", func(t *testing.T) {
		dirs := ExtractINLP(pos, neg, 2)
		if len(dirs) != 2 {
			t.Fatalf("expected 2 directions, got %d", len(dirs))
		}

		// Verify unit norm
		for i, v := range dirs {
			if math.Abs(Norm(v)-1.0) > 1e-6 {
				t.Errorf("direction %d is not normalized: %f", i, Norm(v))
			}
		}

		// Verify orthogonality
		dot := Dot(dirs[0], dirs[1])
		if math.Abs(dot) > 1e-6 {
			t.Errorf("extracted directions are not orthogonal: dot product = %f", dot)
		}
	})
}
