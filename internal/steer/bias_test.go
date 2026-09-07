package steer

import (
	"math"
	"testing"
)

func TestComputeBiasSteering(t *testing.T) {
	bias := []float64{2.0, 3.0, 0.0}
	dir := []float64{1.0, 0.0, 0.0}

	t.Run("Full Projection Out", func(t *testing.T) {
		steered, err := ComputeBiasSteering(bias, dir, 1.0, 0.0)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if math.Abs(steered[0]) > 1e-9 {
			t.Errorf("expected 0 component along direction, got %f", steered[0])
		}
		if math.Abs(steered[1]-3.0) > 1e-9 {
			t.Errorf("expected orthogonal component preserved (3.0), got %f", steered[1])
		}
	})

	t.Run("Partial Projection", func(t *testing.T) {
		steered, err := ComputeBiasSteering(bias, dir, 0.5, 0.0)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		// 2.0 - 0.5 * 2.0 = 1.0
		if math.Abs(steered[0]-1.0) > 1e-9 {
			t.Errorf("expected 1.0 component along direction, got %f", steered[0])
		}
	})

	t.Run("Zero Strength", func(t *testing.T) {
		steered, err := ComputeBiasSteering(bias, dir, 0.0, 0.0)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if steered[0] != bias[0] || steered[1] != bias[1] {
			t.Errorf("expected identical bias for strength 0, got %v", steered)
		}
	})

	t.Run("Dimension Mismatch", func(t *testing.T) {
		_, err := ComputeBiasSteering(bias, []float64{1.0}, 1.0, 0.0)
		if err == nil {
			t.Error("expected error on dimension mismatch")
		}
	})
}

func TestIsBiasSupported(t *testing.T) {
	if !IsBiasSupported(ComponentAttentionOut) {
		t.Error("expected attention_out to support bias")
	}
	if !IsBiasSupported(ComponentMLPDown) {
		t.Error("expected mlp_down to support bias")
	}
	if IsBiasSupported(ComponentResidual) {
		t.Error("expected residual not to support bias")
	}
}
