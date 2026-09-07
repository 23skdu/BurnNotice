package analysis

import (
	"math"
	"testing"
)

func TestComputeLatentSeparation(t *testing.T) {
	harmful := [][]float64{
		{1.0, 0.0, 0.0},
		{1.1, 0.1, 0.0},
	}
	benign := [][]float64{
		{0.0, 1.0, 0.0},
		{0.0, 0.9, 0.1},
	}

	t.Run("Orthogonal Clusters", func(t *testing.T) {
		sep, err := ComputeLatentSeparation(harmful, benign)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		// Nearly orthogonal centroids -> cosine sim ≈ 0 -> distance ≈ 1.0
		if math.Abs(sep-1.0) > 0.1 {
			t.Errorf("expected separation close to 1.0, got %f", sep)
		}
	})

	t.Run("Identical Clusters", func(t *testing.T) {
		sep, err := ComputeLatentSeparation(harmful, harmful)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if math.Abs(sep) > 1e-6 {
			t.Errorf("expected 0 separation for identical clusters, got %f", sep)
		}
	})

	t.Run("Opposite Clusters", func(t *testing.T) {
		opposite := [][]float64{
			{-1.0, 0.0, 0.0},
			{-1.1, -0.1, 0.0},
		}
		sep, err := ComputeLatentSeparation(harmful, opposite)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		// Opposite centroids -> cosine sim ≈ -1 -> distance ≈ 2.0
		if math.Abs(sep-2.0) > 0.1 {
			t.Errorf("expected separation close to 2.0, got %f", sep)
		}
	})
}

func TestMeasureSeparationRetention(t *testing.T) {
	preHarm := [][]float64{{1.0, 0.0}}
	preBenign := [][]float64{{0.0, 1.0}} // sepPre = 1.0

	t.Run("Preserved Separation", func(t *testing.T) {
		postHarm := [][]float64{{0.9, 0.1}}
		postBenign := [][]float64{{0.1, 0.9}} // high separation preserved
		ratio, degraded, err := MeasureSeparationRetention(preHarm, preBenign, postHarm, postBenign)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if degraded {
			t.Errorf("expected not degraded, got ratio=%f", ratio)
		}
	})

	t.Run("Collapsing Separation Drop > 25%", func(t *testing.T) {
		// Post-steering clusters collapsed into nearly identical space
		postHarm := [][]float64{{0.9, 0.9}}
		postBenign := [][]float64{{0.85, 0.85}} // distance drops drastically
		ratio, degraded, err := MeasureSeparationRetention(preHarm, preBenign, postHarm, postBenign)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !degraded {
			t.Errorf("expected degraded=true on collapse, got ratio=%f", ratio)
		}
	})
}
