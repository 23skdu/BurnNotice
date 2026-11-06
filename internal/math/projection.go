package math

import (
	"fmt"
	"math"
)

// RidgeRegularizedProjection performs projection onto the subspace spanned by V_k
// with ridge regularization. Given W (d-dimensional vector) and V_k (d×k basis):
//
//	W_proj = V_k · (V_k^T · V_k + alpha · I)^{-1} · V_k^T · W
//
// As alpha → 0, this reduces to standard orthogonal projection.
// As alpha → ∞, the projection shrinks to zero.
func RidgeRegularizedProjection(W []float64, V_k [][]float64, alpha float64) ([]float64, error) {
	d := len(W)
	k := len(V_k)
	if k == 0 {
		return nil, fmt.Errorf("empty basis")
	}
	for _, v := range V_k {
		if len(v) != d {
			return nil, fmt.Errorf("basis vector dimension %d != input dimension %d", len(v), d)
		}
	}

	// Compute V_k^T · V_k (k×k Gram matrix)
	gram := make([][]float64, k)
	for i := 0; i < k; i++ {
		gram[i] = make([]float64, k)
		for j := 0; j < k; j++ {
			var sum float64
			for l := 0; l < d; l++ {
				sum += V_k[i][l] * V_k[j][l]
			}
			gram[i][j] = sum
		}
		gram[i][i] += alpha // ridge regularization on diagonal
	}

	// Cholesky decomposition: gram = L · L^T
	L, err := choleskyDecomp(gram)
	if err != nil {
		return nil, fmt.Errorf("cholesky failed (alpha may be too small): %w", err)
	}

	// Compute V_k^T · W (k-dimensional)
	vTw := make([]float64, k)
	for i := 0; i < k; i++ {
		for l := 0; l < d; l++ {
			vTw[i] += V_k[i][l] * W[l]
		}
	}

	// Solve L · y = V_k^T · W (forward substitution)
	y := make([]float64, k)
	for i := 0; i < k; i++ {
		var sum float64
		for j := 0; j < i; j++ {
			sum += L[i][j] * y[j]
		}
		if L[i][i] == 0 {
			return nil, fmt.Errorf("singular gram matrix")
		}
		y[i] = (vTw[i] - sum) / L[i][i]
	}

	// Solve L^T · x = y (back substitution)
	x := make([]float64, k)
	for i := k - 1; i >= 0; i-- {
		var sum float64
		for j := i + 1; j < k; j++ {
			sum += L[j][i] * x[j]
		}
		if L[i][i] == 0 {
			return nil, fmt.Errorf("singular gram matrix")
		}
		x[i] = (y[i] - sum) / L[i][i]
	}

	// Compute W_proj = V_k · x
	result := make([]float64, d)
	for l := 0; l < d; l++ {
		for i := 0; i < k; i++ {
			result[l] += V_k[i][l] * x[i]
		}
	}

	return result, nil
}

// AlphaScheduler generates dynamic regularization strengths.
// Starts at alpha_high and decays toward alpha_low over totalPasses.
type AlphaScheduler struct {
	AlphaHigh   float64
	AlphaLow    float64
	TotalPasses int
}

// NewAlphaScheduler creates a scheduler with default values.
func NewAlphaScheduler(totalPasses int) *AlphaScheduler {
	if totalPasses <= 0 {
		totalPasses = 1
	}
	return &AlphaScheduler{
		AlphaHigh:   1.0,
		AlphaLow:    0.0,
		TotalPasses: totalPasses,
	}
}

// Alpha returns the regularization strength for the given pass index.
// Uses exponential decay: alpha(pass) = alpha_low + (alpha_high - alpha_low) * exp(-3 * pass / totalPasses)
func (s *AlphaScheduler) Alpha(pass int) float64 {
	if pass >= s.TotalPasses {
		return s.AlphaLow
	}
	t := float64(pass) / float64(s.TotalPasses-1)
	decay := math.Exp(-3 * t)
	return s.AlphaLow + (s.AlphaHigh-s.AlphaLow)*decay
}

func choleskyDecomp(A [][]float64) ([][]float64, error) {
	n := len(A)
	L := make([][]float64, n)
	for i := 0; i < n; i++ {
		L[i] = make([]float64, n)
	}

	for i := 0; i < n; i++ {
		for j := 0; j <= i; j++ {
			var sum float64
			for k := 0; k < j; k++ {
				sum += L[i][k] * L[j][k]
			}
			if i == j {
				val := A[i][i] - sum
				if val <= 0 {
					return nil, fmt.Errorf("matrix not positive definite")
				}
				L[i][j] = math.Sqrt(val)
			} else {
				if L[j][j] == 0 {
					return nil, fmt.Errorf("singular diagonal")
				}
				L[i][j] = (A[i][j] - sum) / L[j][j]
			}
		}
	}

	return L, nil
}
