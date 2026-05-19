package policydetect

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

// WhiteningDetector detects policy violations using whitening transformation.
type WhiteningDetector struct {
	mean         []float64
	whiteningMat *mat.Dense
	trained      bool
	epsilon      float64
}

// NewWhiteningDetector creates a new detector with numerical stability epsilon.
func NewWhiteningDetector(epsilon float64) *WhiteningDetector {
	if epsilon <= 0 {
		epsilon = 1e-8
	}
	return &WhiteningDetector{epsilon: epsilon}
}

// Train computes the whitening matrix from compliant and violating samples.
func (d *WhiteningDetector) Train(compliant, violating [][]float64) error {
	if len(compliant) < 2 {
		return fmt.Errorf("need at least 2 compliant samples, got %d", len(compliant))
	}

	dim := len(compliant[0])
	for i, c := range compliant {
		if len(c) != dim {
			return fmt.Errorf("compliant sample %d has dim %d, expected %d", i, len(c), dim)
		}
	}

	// Compute mean of compliant samples
	d.mean = make([]float64, dim)
	for _, c := range compliant {
		for j := 0; j < dim; j++ {
			d.mean[j] += c[j]
		}
	}
	for j := 0; j < dim; j++ {
		d.mean[j] /= float64(len(compliant))
	}

	// Center compliant samples
	centered := mat.NewDense(len(compliant), dim, nil)
	for i, c := range compliant {
		for j := 0; j < dim; j++ {
			centered.Set(i, j, c[j]-d.mean[j])
		}
	}

	// Compute covariance matrix
	cov := mat.NewSymDense(dim, nil)
	stat.CovarianceMatrix(cov, centered, nil)

	// Eigen-decomposition
	var eigen mat.EigenSym
	if !eigen.Factorize(cov, true) {
		return fmt.Errorf("eigen-decomposition failed")
	}

	eigenvalues := eigen.Values(nil)
	eigenvectors := mat.NewDense(dim, dim, nil)
	eigen.VectorsTo(eigenvectors)

	// Build whitening matrix: V * diag(1/sqrt(λ + ε)) * V^T
	n := dim
	invSqrtData := make([]float64, n*n)
	for i := 0; i < n; i++ {
		val := eigenvalues[i]
		if val < 0 {
			val = 0
		}
		invSqrtData[i*n+i] = 1.0 / math.Sqrt(val+d.epsilon)
	}
	invSqrt := mat.NewDense(n, n, invSqrtData)

	var tmp mat.Dense
	tmp.Mul(eigenvectors, invSqrt)
	d.whiteningMat = mat.NewDense(n, n, nil)
	d.whiteningMat.Mul(&tmp, eigenvectors.T())

	d.trained = true
	return nil
}

// Detect computes a compliance score for a single activation.
// Higher score = more compliant.
func (d *WhiteningDetector) Detect(activation []float64) (float64, error) {
	if !d.trained {
		return 0, fmt.Errorf("detector not trained")
	}
	if len(activation) != len(d.mean) {
		return 0, fmt.Errorf("activation dim %d != trained dim %d", len(activation), len(d.mean))
	}

	// Center
	centered := make([]float64, len(activation))
	for i := range activation {
		centered[i] = activation[i] - d.mean[i]
	}

	// Whiten
	centeredVec := mat.NewVecDense(len(centered), centered)
	var whitened mat.VecDense
	whitened.MulVec(d.whiteningMat, centeredVec)

	// Compute norm (lower norm = more compliant)
	norm := mat.Norm(&whitened, 2)

	// Convert to compliance score: exp(-norm^2 / 2)
	score := math.Exp(-norm * norm / 2)
	return score, nil
}

// IsTrained returns whether the detector has been trained.
func (d *WhiteningDetector) IsTrained() bool {
	return d.trained
}

// Dimension returns the dimensionality of the trained space.
func (d *WhiteningDetector) Dimension() int {
	return len(d.mean)
}
