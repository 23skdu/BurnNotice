package extract

import (
	burnmath "burnnotice/internal/math"
	"math"

	"gonum.org/v1/gonum/mat"
)

// ExtractWhitenedSVD computes the top-k principal directions from whitened activation differences.
// alpha is the shrinkage intensity for covariance estimation (0.0 to 1.0).
func ExtractWhitenedSVD(positive, negative [][]float64, k int, alpha float64) (directions [][]float64, singularValues []float64) {
	if len(positive) == 0 || len(negative) == 0 {
		return nil, nil
	}

	// 1. Winsorize (from standard PCA extraction)
	burnmath.WinsorizeMatrixInPlace(positive, 0.005, 0.995)
	burnmath.WinsorizeMatrixInPlace(negative, 0.005, 0.995)

	rows := len(positive)
	cols := len(positive[0])
	maxK := cols
	if rows < maxK {
		maxK = rows
	}
	if k > maxK {
		k = maxK
	}
	if k <= 0 {
		return nil, nil
	}

	negMean := columnMean(negative)

	// Combine all activations to estimate covariance (using both positive and negative to get better estimate)
	allActivations := make([][]float64, len(positive)+len(negative))
	copy(allActivations, positive)
	copy(allActivations[len(positive):], negative)

	// 2. Estimate Covariance
	cov := burnmath.EstimateCovariance(allActivations, alpha)
	if cov == nil {
		return nil, nil
	}

	// 3. Eigen Decomposition of Covariance for ZCA Whitening
	var eig mat.EigenSym
	if !eig.Factorize(cov, true) {
		return nil, nil
	}

	eigenVals := eig.Values(nil)
	var U mat.Dense
	eig.VectorsTo(&U)

	// Compute C^{-1/2} (whitening transform) and C^{1/2} (unwhitening transform)
	// We add a tiny epsilon to eigenvalues to prevent division by zero or NaN
	invSqrtEig := mat.NewDiagDense(cols, nil)
	sqrtEig := mat.NewDiagDense(cols, nil)
	epsilon := 1e-8
	for i := 0; i < cols; i++ {
		val := eigenVals[i]
		if val < epsilon {
			val = epsilon
		}
		invSqrtEig.SetDiag(i, 1.0/math.Sqrt(val))
		sqrtEig.SetDiag(i, math.Sqrt(val))
	}

	// C^{-1/2} = U * diag(lambda^{-1/2}) * U^T
	var invSqrtCov mat.Dense
	var U_T mat.Dense
	U_T.CloneFrom(U.T())
	
	var temp mat.Dense
	temp.Mul(&U, invSqrtEig)
	invSqrtCov.Mul(&temp, &U_T)

	// C^{1/2} = U * diag(lambda^{1/2}) * U^T
	var sqrtCov mat.Dense
	temp.Mul(&U, sqrtEig)
	sqrtCov.Mul(&temp, &U_T)

	// 4. Compute differences
	diff := mat.NewDense(rows, cols, nil)
	for i := range positive {
		for j := range positive[i] {
			diff.Set(i, j, positive[i][j]-negMean[j])
		}
	}

	// 5. Whiten differences: diff_white = diff * C^{-1/2}
	var diffWhite mat.Dense
	diffWhite.Mul(diff, &invSqrtCov)

	// 6. SVD on whitened differences
	var svd mat.SVD
	if !svd.Factorize(&diffWhite, mat.SVDThin) {
		return nil, nil
	}

	minDim := rows
	if cols < rows {
		minDim = cols
	}
	vtWhite := mat.NewDense(cols, minDim, nil)
	svd.VTo(vtWhite)

	allValues := svd.Values(nil)
	singularValues = make([]float64, k)
	for i := range singularValues {
		singularValues[i] = allValues[i]
	}

	// 7. Transform directions back: V_orig = C^{1/2} * V_white
	// Since vtWhite has directions as columns, we multiply C^{1/2} * vtWhite
	var vOrig mat.Dense
	vOrig.Mul(&sqrtCov, vtWhite)

	directions = make([][]float64, k)
	for i := 0; i < k; i++ {
		colVec := make([]float64, cols)
		for rowIdx := 0; rowIdx < cols; rowIdx++ {
			colVec[rowIdx] = vOrig.At(rowIdx, i)
		}
		directions[i] = Normalize(colVec)
	}

	return directions, singularValues
}
