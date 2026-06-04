package math

import (
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

// EstimateCovariance calculates the empirical covariance matrix of X and applies
// a regularized shrinkage estimator for small-sample rank safety.
// X is assumed to be an n x d slice of slices (n samples, d features).
// alpha is the shrinkage intensity (0.0 means empirical, 1.0 means pure diagonal).
func EstimateCovariance(X [][]float64, alpha float64) *mat.SymDense {
	n := len(X)
	if n == 0 {
		return nil
	}
	d := len(X[0])
	if d == 0 {
		return nil
	}

	// Convert [][]float64 to mat.Dense
	xMat := mat.NewDense(n, d, nil)
	for i := 0; i < n; i++ {
		for j := 0; j < d; j++ {
			xMat.Set(i, j, X[i][j])
		}
	}

	// Calculate empirical covariance
	empCov := mat.NewSymDense(d, nil)
	stat.CovarianceMatrix(empCov, xMat, nil)

	// Calculate shrinkage target (Trace(S) / d * I)
	trace := 0.0
	for i := 0; i < d; i++ {
		trace += empCov.At(i, i)
	}
	avgVar := trace / float64(d)

	// If alpha < 0, compute optimal Ledoit-Wolf shrinkage
	if alpha < 0.0 {
		diffNormSq := 0.0
		for i := 0; i < d; i++ {
			for j := 0; j < d; j++ {
				val := empCov.At(i, j)
				if i == j {
					val -= avgVar
				}
				diffNormSq += val * val
			}
		}

		if diffNormSq == 0.0 {
			alpha = 0.0
		} else {
			// Center X to compute piSum
			means := make([]float64, d)
			for i := 0; i < n; i++ {
				for j := 0; j < d; j++ {
					means[j] += X[i][j]
				}
			}
			for j := 0; j < d; j++ {
				means[j] /= float64(n)
			}

			sNormSq := 0.0
			for j := 0; j < d; j++ {
				for k := 0; k < d; k++ {
					val := empCov.At(j, k)
					sNormSq += val * val
				}
			}

			piSum := 0.0
			for i := 0; i < n; i++ {
				xi := make([]float64, d)
				normXiSq := 0.0
				for j := 0; j < d; j++ {
					val := X[i][j] - means[j]
					xi[j] = val
					normXiSq += val * val
				}

				xiTSxi := 0.0
				for j := 0; j < d; j++ {
					sum := 0.0
					for k := 0; k < d; k++ {
						sum += empCov.At(j, k) * xi[k]
					}
					xiTSxi += xi[j] * sum
				}

				piSum += normXiSq*normXiSq - 2.0*xiTSxi + sNormSq
			}

			alpha = (piSum / float64(n*n)) / diffNormSq
			if alpha > 1.0 {
				alpha = 1.0
			} else if alpha < 0.0 {
				alpha = 0.0
			}
		}
	}

	// Apply shrinkage: S_shrink = (1 - alpha) * S + alpha * target
	shrunkCov := mat.NewSymDense(d, nil)
	for i := 0; i < d; i++ {
		for j := i; j < d; j++ {
			val := (1.0 - alpha) * empCov.At(i, j)
			if i == j {
				val += alpha * avgVar
			}
			shrunkCov.SetSym(i, j, val)
		}
	}

	return shrunkCov
}
