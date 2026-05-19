package steer

import (
	"math"
	"testing"
)

func TestNewLoRASteerer(t *testing.T) {
	l := NewLoRASteerer(10, 3)

	if l.dim != 10 {
		t.Errorf("dim = %d, want 10", l.dim)
	}
	if l.rank != 3 {
		t.Errorf("rank = %d, want 3", l.rank)
	}
}

func TestNewLoRASteerer_RankClamped(t *testing.T) {
	l := NewLoRASteerer(5, 10)

	if l.rank != 5 {
		t.Errorf("rank = %d, want 5 (clamped to dim)", l.rank)
	}
}

func TestLoRASteerer_ComputeLoRAWeights(t *testing.T) {
	l := NewLoRASteerer(4, 2)

	direction := []float64{1, 2, 3, 4}
	if err := l.ComputeLoRAWeights(direction, 0.5); err != nil {
		t.Fatalf("ComputeLoRAWeights error: %v", err)
	}

	if l.A == nil || l.B == nil {
		t.Fatal("A and B should be initialized")
	}
}

func TestLoRASteerer_ComputeLoRAWeights_WrongDim(t *testing.T) {
	l := NewLoRASteerer(4, 2)

	if err := l.ComputeLoRAWeights([]float64{1, 2}, 0.5); err == nil {
		t.Fatal("expected error for wrong dimension")
	}
}

func TestLoRASteerer_ComputeLoRAWeights_ZeroDirection(t *testing.T) {
	l := NewLoRASteerer(4, 2)

	if err := l.ComputeLoRAWeights([]float64{0, 0, 0, 0}, 0.5); err == nil {
		t.Fatal("expected error for zero-norm direction")
	}
}

func TestLoRASteerer_ApplyLoRA(t *testing.T) {
	l := NewLoRASteerer(2, 1)

	direction := []float64{1, 0}
	l.ComputeLoRAWeights(direction, 1.0)

	weights := [][]float64{
		{0, 0},
		{0, 0},
	}

	updated, err := l.ApplyLoRA(weights)
	if err != nil {
		t.Fatalf("ApplyLoRA error: %v", err)
	}

	if !l.IsApplied() {
		t.Error("LoRA should be marked as applied")
	}

	// A*B should add something
	if updated[0][0] == 0 && updated[0][1] == 0 {
		t.Error("weights should be modified after ApplyLoRA")
	}
}

func TestLoRASteerer_ApplyLoRA_WrongDim(t *testing.T) {
	l := NewLoRASteerer(2, 1)
	l.ComputeLoRAWeights([]float64{1, 0}, 1.0)

	_, err := l.ApplyLoRA([][]float64{{0}})
	if err == nil {
		t.Fatal("expected error for wrong weight dimension")
	}
}

func TestLoRASteerer_ReverseLoRA(t *testing.T) {
	l := NewLoRASteerer(2, 1)

	direction := []float64{1, 0}
	l.ComputeLoRAWeights(direction, 1.0)

	weights := [][]float64{
		{1, 2},
		{3, 4},
	}

	updated, _ := l.ApplyLoRA(weights)
	reversed, err := l.ReverseLoRA(updated)
	if err != nil {
		t.Fatalf("ReverseLoRA error: %v", err)
	}

	if l.IsApplied() {
		t.Error("LoRA should be marked as not applied after reverse")
	}

	// Check reversal restores original
	for i := range weights {
		for j := range weights[i] {
			if math.Abs(reversed[i][j]-weights[i][j]) > 1e-9 {
				t.Errorf("reversed[%d][%d] = %f, want %f", i, j, reversed[i][j], weights[i][j])
			}
		}
	}
}

func TestLoRASteerer_ReverseLoRA_NotApplied(t *testing.T) {
	l := NewLoRASteerer(2, 1)

	_, err := l.ReverseLoRA([][]float64{{1, 2}, {3, 4}})
	if err == nil {
		t.Fatal("expected error when reversing without applying")
	}
}

func TestLoRASteerer_Rank(t *testing.T) {
	l := NewLoRASteerer(10, 4)

	if l.Rank() != 4 {
		t.Errorf("Rank = %d, want 4", l.Rank())
	}
}

func TestLoRASteerer_RoundTrip(t *testing.T) {
	l := NewLoRASteerer(3, 2)

	direction := []float64{0.5, 0.3, 0.2}
	l.ComputeLoRAWeights(direction, 0.5)

	weights := [][]float64{
		{1, 0, 0},
		{0, 1, 0},
		{0, 0, 1},
	}

	updated, _ := l.ApplyLoRA(weights)
	reversed, _ := l.ReverseLoRA(updated)

	for i := range weights {
		for j := range weights[i] {
			if math.Abs(reversed[i][j]-weights[i][j]) > 1e-9 {
				t.Errorf("round-trip failed at [%d][%d]: %f != %f", i, j, reversed[i][j], weights[i][j])
			}
		}
	}
}

func TestLoRASteerer_ReverseLoRA_WrongDim(t *testing.T) {
	l := NewLoRASteerer(2, 1)
	l.ComputeLoRAWeights([]float64{1, 0}, 1.0)
	_, _ = l.ApplyLoRA([][]float64{{1, 2}, {3, 4}})

	_, err := l.ReverseLoRA([][]float64{{1}})
	if err == nil {
		t.Fatal("expected error for wrong weight dimension in ReverseLoRA")
	}
}
