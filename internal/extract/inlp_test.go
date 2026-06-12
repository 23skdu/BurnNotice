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

func TestExtractINLP_ProjectsOutDirections(t *testing.T) {
	positive := [][]float64{
		{10, 1, 2},
		{12, 1, 2},
		{8, 1, 2},
	}
	negative := [][]float64{
		{0, 1, 2},
		{0, 1, 2},
		{0, 1, 2},
	}

	dirs := ExtractINLP(positive, negative, 1)
	if len(dirs) != 1 {
		t.Fatalf("expected 1 direction, got %d", len(dirs))
	}
	u := dirs[0]

	if math.Abs(math.Abs(u[0])-1.0) > 1e-5 || math.Abs(u[1]) > 1e-5 || math.Abs(u[2]) > 1e-5 {
		t.Errorf("extracted direction is not along [1, 0, 0]: %v", u)
	}

	for i := range positive {
		diff := make([]float64, 3)
		for j := 0; j < 3; j++ {
			diff[j] = positive[i][j] - negative[i][j]
		}
		dot := Dot(diff, u)
		projDiff := make([]float64, 3)
		for j := 0; j < 3; j++ {
			projDiff[j] = diff[j] - dot*u[j]
		}
		projNorm := Norm(projDiff)
		if projNorm > 1e-5 {
			t.Errorf("projected difference %d norm = %f, want ~0", i, projNorm)
		}
	}
}
