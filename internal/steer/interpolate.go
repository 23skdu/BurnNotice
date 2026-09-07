package steer

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/floats"
)

// FloatDirectionInterpolation computes a smooth Gaussian-weighted blend of discrete basis directions.
// t is the continuous interpolation coordinate in [0, len(directions)-1].
// sigma is the Gaussian kernel width controlling the softness of the blend (default 1.0 if <= 0).
func FloatDirectionInterpolation(directions [][]float64, t float64, sigma float64) ([]float64, error) {
	if len(directions) == 0 {
		return nil, fmt.Errorf("no directions provided for interpolation")
	}

	dim := len(directions[0])
	if dim == 0 {
		return nil, fmt.Errorf("empty direction dimension")
	}

	for i, d := range directions {
		if len(d) != dim {
			return nil, fmt.Errorf("direction %d dimension %d mismatches expected %d", i, len(d), dim)
		}
	}

	if len(directions) == 1 {
		result := make([]float64, dim)
		copy(result, directions[0])
		return result, nil
	}

	if sigma <= 0 {
		sigma = 1.0
	}

	n := len(directions)
	// Clamp t to [0, n-1]
	if t < 0 {
		t = 0
	} else if t > float64(n-1) {
		t = float64(n - 1)
	}

	weights := make([]float64, n)
	sumW := 0.0
	twoSigmaSq := 2.0 * sigma * sigma

	for i := 0; i < n; i++ {
		diff := float64(i) - t
		w := math.Exp(-(diff * diff) / twoSigmaSq)
		weights[i] = w
		sumW += w
	}

	if sumW < 1e-12 {
		// Fallback to nearest index
		idx := int(math.Round(t))
		if idx < 0 {
			idx = 0
		} else if idx >= n {
			idx = n - 1
		}
		result := make([]float64, dim)
		copy(result, directions[idx])
		return result, nil
	}

	// Blend vectors with normalized weights
	result := make([]float64, dim)
	for i := 0; i < n; i++ {
		normW := weights[i] / sumW
		floats.AddScaled(result, normW, directions[i])
	}

	// Normalize blended direction to unit norm
	norm := floats.Norm(result, 2)
	if norm > 1e-9 {
		floats.Scale(1.0/norm, result)
	}

	return result, nil
}

// InterpolateDirectionFromCommand resolves the effective steering vector when DirectionMix is set.
// If DirectionMix <= 0, returns primary vector. Otherwise blends primary with auxiliary basis directions.
func InterpolateDirectionFromCommand(primary []float64, basis [][]float64, mix float64) ([]float64, error) {
	if mix <= 0 || len(basis) == 0 {
		result := make([]float64, len(primary))
		copy(result, primary)
		return result, nil
	}

	// Combine primary as index 0 followed by basis directions
	allDirs := make([][]float64, 0, len(basis)+1)
	allDirs = append(allDirs, primary)
	allDirs = append(allDirs, basis...)

	// Mix coordinate ranges from 0 to len(allDirs)-1
	t := mix
	if t > float64(len(allDirs)-1) {
		t = float64(len(allDirs) - 1)
	}

	return FloatDirectionInterpolation(allDirs, t, 0.75)
}
