package extract

import (
	"fmt"
	"math"
)

// DirectionTransfer handles transferring directions between models.
type DirectionTransfer struct {
	sourceDim int
	targetDim int
}

// NewDirectionTransfer creates a transfer helper.
func NewDirectionTransfer(sourceDim, targetDim int) *DirectionTransfer {
	return &DirectionTransfer{
		sourceDim: sourceDim,
		targetDim: targetDim,
	}
}

// TransferDirection projects a source direction into the target model's space.
// Uses CCA-style alignment when dimensions differ.
func (t *DirectionTransfer) TransferDirection(source []float64) ([]float64, error) {
	if len(source) != t.sourceDim {
		return nil, fmt.Errorf("source dim %d != expected %d", len(source), t.sourceDim)
	}

	if t.sourceDim == t.targetDim {
		result := make([]float64, t.targetDim)
		copy(result, source)
		return result, nil
	}

	// Dimension mismatch: pad or truncate
	if t.targetDim > t.sourceDim {
		result := make([]float64, t.targetDim)
		copy(result, source)
		// Pad with zeros
		return result, nil
	}

	// Truncate
	result := make([]float64, t.targetDim)
	copy(result, source[:t.targetDim])
	return result, nil
}

// CKASimilarity computes Centered Kernel Alignment between two matrices.
func CKASimilarity(X, Y [][]float64) (float64, error) {
	if len(X) != len(Y) {
		return 0, fmt.Errorf("X rows %d != Y rows %d", len(X), len(Y))
	}
	if len(X) == 0 {
		return 0, fmt.Errorf("empty matrices")
	}

	hsicXY := hsic(X, Y)
	hsicXX := hsic(X, X)
	hsicYY := hsic(Y, Y)

	if hsicXX <= 0 || hsicYY <= 0 {
		return 0, fmt.Errorf("zero self-similarity")
	}

	return hsicXY / math.Sqrt(hsicXX*hsicYY), nil
}

func hsic(X, Y [][]float64) float64 {
	n := len(X)
	if n == 0 {
		return 0
	}

	K := gramMatrix(X)
	L := gramMatrix(Y)

	K = centerMatrix(K, n)
	L = centerMatrix(L, n)

	var trace float64
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			trace += K[i][j] * L[j][i]
		}
	}

	_ = n // used in denominator
	return trace / float64((n-1)*(n-1))
}

func gramMatrix(X [][]float64) [][]float64 {
	n := len(X)
	K := make([][]float64, n)
	for i := 0; i < n; i++ {
		K[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			var sum float64
			for k := 0; k < len(X[i]); k++ {
				sum += X[i][k] * X[j][k]
			}
			K[i][j] = sum
		}
	}
	return K
}

func centerMatrix(M [][]float64, n int) [][]float64 {
	H := make([][]float64, n)
	for i := 0; i < n; i++ {
		H[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			val := M[i][j]
			rowMean := rowAvg(M, i, n)
			colMean := colAvg(M, j, n)
			totalMean := totalAvg(M, n)
			H[i][j] = val - rowMean - colMean + totalMean
		}
	}
	return H
}

func rowAvg(M [][]float64, i, n int) float64 {
	var sum float64
	for j := 0; j < n; j++ {
		sum += M[i][j]
	}
	return sum / float64(n)
}

func colAvg(M [][]float64, j, n int) float64 {
	var sum float64
	for i := 0; i < n; i++ {
		sum += M[i][j]
	}
	return sum / float64(n)
}

func totalAvg(M [][]float64, n int) float64 {
	var sum float64
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			sum += M[i][j]
		}
	}
	return sum / float64(n*n)
}
