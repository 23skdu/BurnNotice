package extract

import (
	"math"

	burnmath "burnnotice/internal/math"
)

// ExtractDIM computes the Difference-in-Means direction between positive and negative activation sets.
// Returns a normalized direction vector.
func ExtractDIM(positive, negative [][]float64) []float64 {
	if len(positive) == 0 || len(negative) == 0 {
		return nil
	}

	burnmath.WinsorizeMatrixInPlace(positive, 0.005, 0.995)
	burnmath.WinsorizeMatrixInPlace(negative, 0.005, 0.995)

	posMean := columnMean(positive)
	negMean := columnMean(negative)

	dim := len(posMean)
	direction := make([]float64, dim)
	for i := range posMean {
		direction[i] = posMean[i] - negMean[i]
	}

	return Normalize(direction)
}

func columnMean(matrix [][]float64) []float64 {
	if len(matrix) == 0 {
		return nil
	}
	cols := len(matrix[0])
	mean := make([]float64, cols)
	for _, row := range matrix {
		for j := range row {
			mean[j] += row[j]
		}
	}
	n := float64(len(matrix))
	for j := range mean {
		mean[j] /= n
	}
	return mean
}

// Normalize returns a unit vector in the same direction as v.
func Normalize(v []float64) []float64 {
	n := Norm(v)
	if n < 1e-10 {
		return v
	}
	result := make([]float64, len(v))
	for i := range v {
		result[i] = v[i] / n
	}
	return result
}

// Norm computes the L2 norm of a vector.
func Norm(v []float64) float64 {
	var sum float64
	for _, x := range v {
		sum += x * x
	}
	return math.Sqrt(sum)
}

// Dot computes the dot product of two vectors.
func Dot(a, b []float64) float64 {
	var sum float64
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// Sigmoid computes the logistic sigmoid function.
func Sigmoid(x float64) float64 {
	if x > 20 {
		return 1.0
	}
	if x < -20 {
		return 0.0
	}
	return 1.0 / (1.0 + math.Exp(-x))
}
