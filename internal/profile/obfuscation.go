package profile

import (
	"fmt"
	"math"
)

// MLPProbe implements a non-linear Multi-Layer Perceptron probe capturing complex concept manifolds.
type MLPProbe struct {
	Weights1 [][]float64 // InputDim -> HiddenDim
	Biases1  []float64   // HiddenDim
	Weights2 []float64   // HiddenDim -> 1
	Bias2    float64
}

// NewMLPProbe creates an MLP probe with the given dimensions.
func NewMLPProbe(inputDim, hiddenDim int) *MLPProbe {
	w1 := make([][]float64, hiddenDim)
	for i := range w1 {
		w1[i] = make([]float64, inputDim)
		// Initialize weights with standard values
		for j := range w1[i] {
			w1[i][j] = 0.01 * math.Sin(float64(i*j))
		}
	}
	b1 := make([]float64, hiddenDim)
	w2 := make([]float64, hiddenDim)
	for i := range w2 {
		w2[i] = 0.05 * math.Cos(float64(i))
	}
	return &MLPProbe{
		Weights1: w1,
		Biases1:  b1,
		Weights2: w2,
		Bias2:    0.0,
	}
}

// Predict evaluates the activation vector through the MLP probe.
func (p *MLPProbe) Predict(activation []float64) float64 {
	hiddenDim := len(p.Weights1)
	if hiddenDim == 0 || len(activation) == 0 {
		return 0.5
	}

	// First layer: Hidden = ReLU(W1 * X + B1)
	hidden := make([]float64, hiddenDim)
	for i := 0; i < hiddenDim; i++ {
		var sum float64
		for j := 0; j < len(activation); j++ {
			if j < len(p.Weights1[i]) {
				sum += activation[j] * p.Weights1[i][j]
			}
		}
		sum += p.Biases1[i]
		if sum < 0 {
			sum = 0 // ReLU activation
		}
		hidden[i] = sum
	}

	// Second layer: Output = Sigmoid(W2 * Hidden + B2)
	var output float64
	for i := 0; i < hiddenDim; i++ {
		output += hidden[i] * p.Weights2[i]
	}
	output += p.Bias2

	// Sigmoid scaling to [0, 1]
	return 1.0 / (1.0 + math.Exp(-output))
}

// AdversarialPerturber optimizes activation perturbations to simulate evasion attacks (Bailey et al.)
type AdversarialPerturber struct {
	Epsilon float64
}

// Perturb applies gradient-directed adversarial perturbation to an activation vector.
func (ap *AdversarialPerturber) Perturb(activation []float64, probe *MLPProbe, targetScore float64) []float64 {
	perturbed := make([]float64, len(activation))
	copy(perturbed, activation)

	// Approximate gradient of probe prediction w.r.t input using numerical finite differences
	h := 1e-5
	grad := make([]float64, len(activation))
	baseScore := probe.Predict(activation)

	for i := 0; i < len(activation); i++ {
		temp := perturbed[i]
		perturbed[i] += h
		highScore := probe.Predict(perturbed)
		perturbed[i] = temp

		grad[i] = (highScore - baseScore) / h
	}

	// Apply gradient-sign perturbation (similar to FGSM) to push score toward or away from target
	sign := 1.0
	if baseScore > targetScore {
		sign = -1.0
	}

	for i := 0; i < len(activation); i++ {
		perturbed[i] = activation[i] + sign*ap.Epsilon*math.Copysign(1.0, grad[i])
	}

	return perturbed
}

// OODAnomalyDetector flags suspicious activation states that suggest intentional evasion or obfuscation attacks.
type OODAnomalyDetector struct {
	MeanVector []float64
	StdVector  []float64
	Threshold  float64
}

// NewOODAnomalyDetector constructs a detector based on a baseline activation distribution.
func NewOODAnomalyDetector(baseline [][]float64, threshold float64) (*OODAnomalyDetector, error) {
	if len(baseline) == 0 {
		return nil, fmt.Errorf("empty baseline activations")
	}
	dim := len(baseline[0])
	mean := make([]float64, dim)
	variance := make([]float64, dim)

	for _, vec := range baseline {
		for i := 0; i < dim; i++ {
			mean[i] += vec[i]
		}
	}
	for i := 0; i < dim; i++ {
		mean[i] /= float64(len(baseline))
	}

	for _, vec := range baseline {
		for i := 0; i < dim; i++ {
			diff := vec[i] - mean[i]
			variance[i] += diff * diff
		}
	}

	std := make([]float64, dim)
	for i := 0; i < dim; i++ {
		std[i] = math.Sqrt(variance[i] / float64(len(baseline)))
		if std[i] < 1e-6 {
			std[i] = 1e-6 // Avoid division by zero
		}
	}

	if threshold <= 0 {
		threshold = 3.0 // Default to 3 standard deviations
	}

	return &OODAnomalyDetector{
		MeanVector: mean,
		StdVector:  std,
		Threshold:  threshold,
	}, nil
}

// IsAnomaly returns true if the activation vector deviates significantly (OOD Mahalanobis-style distance threshold).
func (d *OODAnomalyDetector) IsAnomaly(activation []float64) bool {
	if len(activation) != len(d.MeanVector) {
		return true
	}

	var sumDistSq float64
	for i := 0; i < len(activation); i++ {
		diff := activation[i] - d.MeanVector[i]
		normalizedDiff := diff / d.StdVector[i]
		sumDistSq += normalizedDiff * normalizedDiff
	}

	avgStdDevs := math.Sqrt(sumDistSq / float64(len(activation)))
	return avgStdDevs > d.Threshold
}
