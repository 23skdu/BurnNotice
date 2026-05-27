package policydetect

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
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

	// SVD factorization
	var svd mat.SVD
	ok := svd.Factorize(centered, mat.SVDThin)
	if !ok {
		return fmt.Errorf("SVD factorization failed")
	}

	s := svd.Values(nil)
	var v mat.Dense
	svd.VTo(&v)

	_, k := v.Dims() // v is d x k
	N := float64(len(compliant))

	d.whiteningMat = mat.NewDense(k, dim, nil)
	for i := 0; i < k; i++ {
		lambda := (s[i] * s[i]) / (N - 1)
		scale := 1.0 / math.Sqrt(lambda+d.epsilon)
		for j := 0; j < dim; j++ {
			d.whiteningMat.Set(i, j, v.At(j, i)*scale)
		}
	}

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
	k, _ := d.whiteningMat.Dims()
	whitened := mat.NewVecDense(k, nil)
	whitened.MulVec(d.whiteningMat, centeredVec)

	// Compute norm (lower norm = more compliant)
	norm := mat.Norm(whitened, 2)

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
