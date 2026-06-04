package math

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// WhiteningMatrix returns the ZCA (Mahalanobis) whitening matrix W such that
//
//	W = cov^(-1/2) = V · diag(1/√λ) · Vᵀ
//
// where cov = V · diag(λ) · Vᵀ is the eigendecomposition. The matrix satisfies
// W · cov · W = I, so X_white = (X - mean) @ W yields a dataset with identity
// sample covariance.
//
// eigFloor is the minimum eigenvalue retained. Eigenvalues below eigFloor are
// clamped to eigFloor before taking the reciprocal square root, which prevents
// noise amplification and division-by-zero in rank-deficient covariances.
// A sensible default is 1e-5 times the largest eigenvalue, or an absolute
// floor of ~1e-10.
//
// Returns nil, false if cov is nil, ragged, has zero dimension, is not
// numerically symmetric, or if the eigendecomposition fails.
func WhiteningMatrix(cov [][]float64, eigFloor float64) ([][]float64, bool) {
	d := len(cov)
	if d == 0 {
		return nil, false
	}
	for i := 0; i < d; i++ {
		if len(cov[i]) != d {
			return nil, false
		}
	}
	for i := 0; i < d; i++ {
		for j := i + 1; j < d; j++ {
			if !isFinite(cov[i][j]) || !isFinite(cov[j][i]) {
				return nil, false
			}
			if cov[i][j] != cov[j][i] {
				return nil, false
			}
		}
	}

	if eigFloor < 0 {
		eigFloor = 0
	}

	data := make([]float64, d*d)
	for i := 0; i < d; i++ {
		for j := 0; j < d; j++ {
			data[i*d+j] = cov[i][j]
		}
	}
	sym := mat.NewSymDense(d, data)

	var eigen mat.EigenSym
	if !eigen.Factorize(sym, true) {
		return nil, false
	}
	values := eigen.Values(nil)
	vecs := mat.NewDense(d, d, nil)
	eigen.VectorsTo(vecs)

	floor := eigFloor
	if floor <= 0 {
		var maxLambda float64
		for _, v := range values {
			if v > maxLambda {
				maxLambda = v
			}
		}
		if maxLambda <= 0 {
			return nil, false
		}
		floor = maxLambda * 1e-10
	}

	W := makeMatrix(d, d)
	for i := 0; i < d; i++ {
		lambda := values[i]
		if lambda < floor {
			lambda = floor
		}
		invSqrt := 1.0 / sqrt(lambda)
		for r := 0; r < d; r++ {
			for c := 0; c < d; c++ {
				W[r][c] += vecs.At(r, i) * invSqrt * vecs.At(c, i)
			}
		}
	}
	for i := 0; i < d; i++ {
		for j := 0; j < d; j++ {
			if !isFinite(W[i][j]) {
				return nil, false
			}
		}
	}
	return W, true
}

// UnwhiteningMatrix returns the inverse of the whitening matrix, i.e. the
// "re-coloring" matrix cov^{1/2} = V · diag(√λ) · Vᵀ. This is used to map
// directions extracted in the whitened space back to the original space:
// V_orig = cov^{1/2} · V_white.
//
// The same eigFloor rule is applied for consistency with WhiteningMatrix.
func UnwhiteningMatrix(cov [][]float64, eigFloor float64) ([][]float64, bool) {
	d := len(cov)
	if d == 0 {
		return nil, false
	}
	for i := 0; i < d; i++ {
		if len(cov[i]) != d {
			return nil, false
		}
	}
	for i := 0; i < d; i++ {
		for j := i + 1; j < d; j++ {
			if cov[i][j] != cov[j][i] {
				return nil, false
			}
		}
	}

	if eigFloor < 0 {
		eigFloor = 0
	}

	data := make([]float64, d*d)
	for i := 0; i < d; i++ {
		for j := 0; j < d; j++ {
			data[i*d+j] = cov[i][j]
		}
	}
	sym := mat.NewSymDense(d, data)

	var eigen mat.EigenSym
	if !eigen.Factorize(sym, true) {
		return nil, false
	}
	values := eigen.Values(nil)
	vecs := mat.NewDense(d, d, nil)
	eigen.VectorsTo(vecs)

	floor := eigFloor
	if floor <= 0 {
		var maxLambda float64
		for _, v := range values {
			if v > maxLambda {
				maxLambda = v
			}
		}
		if maxLambda <= 0 {
			return nil, false
		}
		floor = maxLambda * 1e-10
	}

	W := makeMatrix(d, d)
	for i := 0; i < d; i++ {
		lambda := values[i]
		if lambda < floor {
			lambda = floor
		}
		sqrtL := sqrt(lambda)
		for r := 0; r < d; r++ {
			for c := 0; c < d; c++ {
				W[r][c] += vecs.At(r, i) * sqrtL * vecs.At(c, i)
			}
		}
	}
	for i := 0; i < d; i++ {
		for j := 0; j < d; j++ {
			if !isFinite(W[i][j]) {
				return nil, false
			}
		}
	}
	return W, true
}

// ApplyWhitening computes X_white = X @ W where X is row-major (n×d) and W is
// d×d. Returns nil on dimension mismatch. The input is not modified.
func ApplyWhitening(X [][]float64, W [][]float64) [][]float64 {
	if len(X) == 0 {
		return nil
	}
	d := len(X[0])
	if len(W) != d || len(W[0]) != d {
		return nil
	}
	out := make([][]float64, len(X))
	for i, row := range X {
		if len(row) != d {
			return nil
		}
		res := make([]float64, d)
		for j := 0; j < d; j++ {
			var s float64
			for k := 0; k < d; k++ {
				s += row[k] * W[k][j]
			}
			res[j] = s
		}
		out[i] = res
	}
	return out
}

// ZCAWhitening is a convenience wrapper that centers the data, estimates a
// (possibly shrunk) covariance, derives the ZCA whitening matrix, and applies
// it in one call.
//
// shrinkage < 0 selects the Ledoit-Wolf optimal intensity; >=0 uses the given
// value (clipped to [0, 1]). eigFloor <= 0 is interpreted as a relative
// floor of 1e-10 times the largest eigenvalue.
//
// Returns (X_white, W, mean, ok). The whitened data has (approximately)
// identity sample covariance. The whitening matrix W and the mean can be
// retained to map new data into the same coordinate frame.
func ZCAWhitening(data [][]float64, shrinkage, eigFloor float64) (Xwhite [][]float64, W [][]float64, mean []float64, ok bool) {
	if len(data) < 2 {
		return nil, nil, nil, false
	}
	d := len(data[0])
	if d == 0 {
		return nil, nil, nil, false
	}
	for _, row := range data {
		if len(row) != d {
			return nil, nil, nil, false
		}
	}

	mean = columnMean(data)
	centered := centerRows(data, mean)

	cov := ShrinkageCovariance(data, shrinkage)
	if cov == nil {
		return nil, nil, nil, false
	}

	W, ok = WhiteningMatrix(cov, eigFloor)
	if !ok {
		return nil, nil, nil, false
	}
	Xwhite = ApplyWhitening(centered, W)
	if Xwhite == nil {
		return nil, nil, nil, false
	}
	return Xwhite, W, mean, true
}

func sqrt(x float64) float64 {
	if x <= 0 {
		return 0
	}
	return math.Sqrt(x)
}
