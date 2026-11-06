package extract

import (
	burnmath "burnnotice/internal/math"
	"gonum.org/v1/gonum/mat"
)

// ExtractWhitenedSVD extracts the top-k directions from the whitened SVD of
// the difference matrix between positive and negative activations. The
// pipeline is:
//
//  1. D = positive − mean(negative)            (n × d difference matrix)
//  2. cov_combined = ShrinkageCovariance(pos ∪ neg, shrinkage)
//  3. W = cov^(-1/2)   (ZCA whitening matrix)
//  4. U = cov^(1/2)    (un-whitening matrix)
//  5. D_white = D · W
//  6. SVD: D_white = Σ_white · U_white · V_whiteᵀ
//  7. V_orig = U · V_white                    (per nextsteps.md 7.3)
//
// Each returned direction is L2-normalized. Singular values are returned in
// descending SVD order.
//
// shrinkage < 0 selects the Ledoit-Wolf optimal intensity (recommended when
// n < d for rank safety). eigFloor <= 0 is interpreted as 1e-10 times the
// largest eigenvalue of cov_combined.
//
// Returns (nil, nil, false) on invalid input or when whitening cannot be
// performed (e.g. zero-variance data with eigFloor=0).
func ExtractWhitenedSVD(positive, negative [][]float64, k int, shrinkage, eigFloor float64) (directions [][]float64, singularValues []float64, ok bool) {
	if len(positive) == 0 || len(negative) == 0 {
		return nil, nil, false
	}
	d := len(positive[0])
	if d == 0 || len(negative[0]) != d {
		return nil, nil, false
	}
	for _, row := range positive {
		if len(row) != d {
			return nil, nil, false
		}
	}
	for _, row := range negative {
		if len(row) != d {
			return nil, nil, false
		}
	}

	posClean := copyMatrix(positive)
	negClean := copyMatrix(negative)
	burnmath.WinsorizeMatrixInPlace(posClean, 0.005, 0.995)
	burnmath.WinsorizeMatrixInPlace(negClean, 0.005, 0.995)

	negMean := columnMean(negClean)
	rows := len(posClean)
	maxK := d
	if rows < maxK {
		maxK = rows
	}
	if k > maxK {
		k = maxK
	}
	if k <= 0 {
		return nil, nil, false
	}

	diff := make([]float64, rows*d)
	for i := 0; i < rows; i++ {
		for j := 0; j < d; j++ {
			v := posClean[i][j] - negMean[j]
			if v > 1e10 {
				v = 1e10
			}
			if v < -1e10 {
				v = -1e10
			}
			diff[i*d+j] = v
		}
	}

	combined := make([][]float64, 0, len(posClean)+len(negClean))
	combined = append(combined, posClean...)
	combined = append(combined, negClean...)
	cov := burnmath.ShrinkageCovariance(combined, shrinkage)
	if cov == nil {
		return nil, nil, false
	}

	W, wok := burnmath.WhiteningMatrix(cov, eigFloor)
	U, uok := burnmath.UnwhiteningMatrix(cov, eigFloor)
	if !wok || !uok {
		return nil, nil, false
	}

	dWhite := make([]float64, rows*d)
	for i := 0; i < rows; i++ {
		for j := 0; j < d; j++ {
			var s float64
			for kk := 0; kk < d; kk++ {
				s += diff[i*d+kk] * W[kk][j]
			}
			dWhite[i*d+j] = s
		}
	}

	diffMat := mat.NewDense(rows, d, dWhite)
	var svd mat.SVD
	if !svd.Factorize(diffMat, mat.SVDThin) {
		return nil, nil, false
	}
	allValues := svd.Values(nil)

	minDim := rows
	if d < rows {
		minDim = d
	}
	vWhiteDense := mat.NewDense(d, minDim, nil)
	svd.VTo(vWhiteDense)

	singularValues = make([]float64, k)
	for i := 0; i < k; i++ {
		singularValues[i] = allValues[i]
	}

	directions = make([][]float64, k)
	for c := 0; c < k; c++ {
		vWhite := make([]float64, d)
		for r := 0; r < d; r++ {
			vWhite[r] = vWhiteDense.At(r, c)
		}
		vOrig := make([]float64, d)
		for r := 0; r < d; r++ {
			var s float64
			for kk := 0; kk < d; kk++ {
				s += U[r][kk] * vWhite[kk]
			}
			vOrig[r] = s
		}
		directions[c] = Normalize(vOrig)
	}
	return directions, singularValues, true
}

func copyMatrix(m [][]float64) [][]float64 {
	out := make([][]float64, len(m))
	for i, row := range m {
		c := make([]float64, len(row))
		copy(c, row)
		out[i] = c
	}
	return out
}
