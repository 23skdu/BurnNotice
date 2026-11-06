package steer

import (
	"fmt"
	"gonum.org/v1/gonum/floats"
	"math"
)

// LoRASteerer implements LoRA-based reversible steering.
type LoRASteerer struct {
	dim     int
	rank    int
	A       [][]float64 // d x r
	B       [][]float64 // r x d
	applied bool
}

// NewLoRASteerer creates a LoRA steerer with the given dimension and rank.
func NewLoRASteerer(dim, rank int) *LoRASteerer {
	if rank > dim {
		rank = dim
	}
	return &LoRASteerer{
		dim:  dim,
		rank: rank,
		A:    make([][]float64, dim),
		B:    make([][]float64, rank),
	}
}

// ComputeLoRAWeights factorizes a steering direction into low-rank matrices A and B.
func (l *LoRASteerer) ComputeLoRAWeights(direction []float64, strength float64) error {
	if len(direction) != l.dim {
		return fmt.Errorf("direction dim %d != steerer dim %d", len(direction), l.dim)
	}

	// Simple rank-1 decomposition: A = direction * sqrt(strength), B = identity-like
	norm := vecNorm(direction)
	if norm < 1e-10 {
		return fmt.Errorf("zero-norm direction")
	}

	// A: d x r, B: r x d
	for i := 0; i < l.dim; i++ {
		l.A[i] = make([]float64, l.rank)
		l.A[i][0] = direction[i] * math.Sqrt(strength)
	}

	for i := 0; i < l.rank; i++ {
		l.B[i] = make([]float64, l.dim)
		if i == 0 {
			for j := 0; j < l.dim; j++ {
				l.B[i][j] = direction[j] / norm
			}
		}
	}

	return nil
}

// ApplyLoRA applies the LoRA update to a weight matrix: W' = W + A*B.
func (l *LoRASteerer) ApplyLoRA(weights [][]float64) ([][]float64, error) {
	if len(weights) != l.dim {
		return nil, fmt.Errorf("weights rows %d != dim %d", len(weights), l.dim)
	}

	result := make([][]float64, l.dim)
	for i := 0; i < l.dim; i++ {
		result[i] = make([]float64, len(weights[i]))
		copy(result[i], weights[i])
		for k := 0; k < l.rank; k++ {
			floats.AddScaled(result[i], l.A[i][k], l.B[k])
		}
	}

	l.applied = true
	return result, nil
}

// ReverseLoRA removes the LoRA update: W = W' - A*B.
func (l *LoRASteerer) ReverseLoRA(weights [][]float64) ([][]float64, error) {
	if !l.applied {
		return nil, fmt.Errorf("LoRA not applied, cannot reverse")
	}
	if len(weights) != l.dim {
		return nil, fmt.Errorf("weights rows %d != dim %d", len(weights), l.dim)
	}

	result := make([][]float64, l.dim)
	for i := 0; i < l.dim; i++ {
		result[i] = make([]float64, len(weights[i]))
		copy(result[i], weights[i])
		for k := 0; k < l.rank; k++ {
			floats.AddScaled(result[i], -l.A[i][k], l.B[k])
		}
	}

	l.applied = false
	return result, nil
}

// IsApplied returns whether LoRA is currently applied.
func (l *LoRASteerer) IsApplied() bool {
	return l.applied
}

// Rank returns the LoRA rank.
func (l *LoRASteerer) Rank() int {
	return l.rank
}
