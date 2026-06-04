package math

// Covariance returns the unbiased sample covariance matrix of the centered
// activation data, using the 1/(n-1) Bessel correction.
//
// data is a row-major matrix where each row is a sample and each column is a
// feature dimension. The returned matrix is d×d (symmetric).
//
// Returns nil if data is empty, ragged, or fewer than 2 rows.
func Covariance(data [][]float64) [][]float64 {
	n := len(data)
	if n < 2 {
		return nil
	}
	d := len(data[0])
	if d == 0 {
		return nil
	}
	for _, row := range data {
		if len(row) != d {
			return nil
		}
	}

	mean := columnMean(data)
	centered := centerRows(data, mean)

	cov := makeMatrix(d, d)
	denom := 1.0 / float64(n-1)
	for i := 0; i < d; i++ {
		for k := 0; k < n; k++ {
			ci := centered[k][i]
			for j := i; j < d; j++ {
				cov[i][j] += ci * centered[k][j]
			}
		}
		for j := i; j < d; j++ {
			cov[i][j] *= denom
			cov[j][i] = cov[i][j]
		}
	}
	return cov
}

// ShrinkageCovariance returns a regularized covariance matrix using Ledoit-Wolf
// shrinkage toward the scalar-target T = μ·I, where μ = trace(S)/d and S is
// the 1/n (MLE) sample covariance.
//
// The returned matrix is
//
//	Σ_shrunk = (1 - ρ) · S + ρ · μ · I
//
// where ρ ∈ [0, 1] is the shrinkage intensity. When ρ > 0 the diagonal of S is
// pulled toward the mean variance, which guarantees the result is full-rank
// even when n < d (small-sample rank safety).
//
// If shrinkage < 0, the Ledoit-Wolf (2004) optimal intensity is estimated from
// the data. The estimate is clipped to [0, 1]. For n < 2 or d == 0 the
// function returns nil.
//
// Reference: Ledoit & Wolf (2004) "A well-conditioned estimator for
// large-dimensional covariance matrices", Journal of Multivariate Analysis.
func ShrinkageCovariance(data [][]float64, shrinkage float64) [][]float64 {
	n := len(data)
	if n < 2 {
		return nil
	}
	d := len(data[0])
	if d == 0 {
		return nil
	}
	for _, row := range data {
		if len(row) != d {
			return nil
		}
	}

	mean := columnMean(data)
	centered := centerRows(data, mean)

	sMLE := makeMatrix(d, d)
	invN := 1.0 / float64(n)
	for i := 0; i < d; i++ {
		for k := 0; k < n; k++ {
			ci := centered[k][i]
			for j := i; j < d; j++ {
				sMLE[i][j] += ci * centered[k][j]
			}
		}
		for j := i; j < d; j++ {
			sMLE[i][j] *= invN
			sMLE[j][i] = sMLE[i][j]
		}
	}

	rho := shrinkage
	if rho < 0 {
		rho = ledoitWolfIntensity(centered, sMLE, d, n)
	}
	if rho < 0 {
		rho = 0
	}
	if rho > 1 {
		rho = 1
	}

	var traceS float64
	for i := 0; i < d; i++ {
		traceS += sMLE[i][i]
	}
	mu := traceS / float64(d)

	if !isFinite(mu) || mu == 0 {
		return sMLE
	}

	out := makeMatrix(d, d)
	for i := 0; i < d; i++ {
		for j := 0; j < d; j++ {
			v := (1.0-rho)*sMLE[i][j] + rho*mu*kroneckerDelta(i, j)
			if !isFinite(v) {
				v = mu * kroneckerDelta(i, j)
			}
			out[i][j] = v
		}
	}
	return out
}

// ledoitWolfIntensity estimates the optimal Ledoit-Wolf shrinkage intensity
// for the μ·I target using the closed-form estimator from Ledoit & Wolf
// (2004). The numerator is the sum of entrywise variances of the MLE
// covariance; the denominator is the squared Frobenius distance from S to μ·I
// scaled by n².
func ledoitWolfIntensity(centered [][]float64, sMLE [][]float64, d, n int) float64 {
	var traceS float64
	for i := 0; i < d; i++ {
		traceS += sMLE[i][i]
	}
	mu := traceS / float64(d)

	var sumS2 float64
	for i := 0; i < d; i++ {
		for j := 0; j < d; j++ {
			sumS2 += sMLE[i][j] * sMLE[i][j]
		}
	}

	var num float64
	for i := 0; i < d; i++ {
		for j := 0; j < d; j++ {
			var ss float64
			sij := sMLE[i][j]
			for k := 0; k < n; k++ {
				diff := centered[k][i]*centered[k][j] - sij
				ss += diff * diff
			}
			num += ss
		}
	}
	num /= float64(n * n)

	var denomInner float64
	for i := 0; i < d; i++ {
		for j := 0; j < d; j++ {
			target := mu * kroneckerDelta(i, j)
			diff := sMLE[i][j] - target
			denomInner += diff * diff
		}
	}
	denom := float64(n*n) * denomInner
	if denom < 1e-300 {
		return 1.0
	}
	return num / denom
}

// AverageVariance returns trace(S)/d, the scalar Ledoit-Wolf shrinkage target
// multiplier for a d×d covariance matrix S. Returns 0 for a nil/empty matrix.
func AverageVariance(cov [][]float64) float64 {
	d := len(cov)
	if d == 0 {
		return 0
	}
	var trace float64
	for i := 0; i < d; i++ {
		trace += cov[i][i]
	}
	return trace / float64(d)
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

func centerRows(data [][]float64, mean []float64) [][]float64 {
	out := make([][]float64, len(data))
	for i, row := range data {
		centered := make([]float64, len(row))
		for j, v := range row {
			diff := v - mean[j]
			if isNaN(diff) || isInf(diff) {
				diff = 0
			}
			centered[j] = diff
		}
		out[i] = centered
	}
	return out
}

func makeMatrix(rows, cols int) [][]float64 {
	m := make([][]float64, rows)
	for i := range m {
		m[i] = make([]float64, cols)
	}
	return m
}

func kroneckerDelta(i, j int) float64 {
	if i == j {
		return 1
	}
	return 0
}

func isFinite(v float64) bool {
	return !isNaN(v) && !isInf(v)
}
