package analysis

import (
	"fmt"
	"math"
)

// ComputeCentroid computes the mean vector across a collection of activation samples.
func ComputeCentroid(activations [][]float64) ([]float64, error) {
	if len(activations) == 0 {
		return nil, fmt.Errorf("empty activations slice")
	}
	dim := len(activations[0])
	if dim == 0 {
		return nil, fmt.Errorf("zero-dimension activation vectors")
	}

	centroid := make([]float64, dim)
	for _, act := range activations {
		if len(act) != dim {
			return nil, fmt.Errorf("activation dimension mismatch: %d != %d", len(act), dim)
		}
		for i := 0; i < dim; i++ {
			centroid[i] += act[i]
		}
	}

	n := float64(len(activations))
	for i := 0; i < dim; i++ {
		centroid[i] /= n
	}

	return centroid, nil
}

// ComputeLatentSeparation measures the geometric cosine distance between harmful and benign cluster centroids:
//   Separation = 1.0 - CosineSimilarity(c_harm, c_benign)
// Higher values indicate distinct, non-overlapping representation manifolds.
func ComputeLatentSeparation(harmfulActs, benignActs [][]float64) (float64, error) {
	cHarm, err := ComputeCentroid(harmfulActs)
	if err != nil {
		return 0.0, fmt.Errorf("compute harmful centroid: %w", err)
	}
	cBenign, err := ComputeCentroid(benignActs)
	if err != nil {
		return 0.0, fmt.Errorf("compute benign centroid: %w", err)
	}

	var dot, harmNormSq, benignNormSq float64
	for i := 0; i < len(cHarm); i++ {
		dot += cHarm[i] * cBenign[i]
		harmNormSq += cHarm[i] * cHarm[i]
		benignNormSq += cBenign[i] * cBenign[i]
	}

	hNorm := math.Sqrt(harmNormSq)
	bNorm := math.Sqrt(benignNormSq)
	if hNorm < 1e-10 || bNorm < 1e-10 {
		return 0.0, fmt.Errorf("zero-norm centroid encountered")
	}

	sim := dot / (hNorm * bNorm)
	if sim > 1.0 {
		sim = 1.0
	} else if sim < -1.0 {
		sim = -1.0
	}

	// Cosine distance ranges [0, 2]
	return 1.0 - sim, nil
}

// MeasureSeparationRetention assesses whether activation steering preserves manifold separability.
// Flags isDegraded if latent separation drops by more than 25% post-intervention.
func MeasureSeparationRetention(preHarm, preBenign, postHarm, postBenign [][]float64) (retentionRatio float64, isDegraded bool, err error) {
	sepPre, err := ComputeLatentSeparation(preHarm, preBenign)
	if err != nil {
		return 0.0, false, fmt.Errorf("pre-steering separation failed: %w", err)
	}
	sepPost, err := ComputeLatentSeparation(postHarm, postBenign)
	if err != nil {
		return 0.0, false, fmt.Errorf("post-steering separation failed: %w", err)
	}

	if sepPre < 1e-6 {
		return 1.0, false, nil // Negligible pre-existing separation
	}

	retentionRatio = sepPost / sepPre
	// Flag if drop > 25% (retention < 75%)
	isDegraded = (sepPre - sepPost) / sepPre > 0.25

	return retentionRatio, isDegraded, nil
}
