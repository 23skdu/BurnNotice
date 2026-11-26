//go:build !cgo

package native

import "math"

// GPUEngine handles direct execution of high-performance activation operations.
// This pure-Go fallback mirrors the C++ kernel semantics (kernel.cpp) for
// platforms built without cgo (e.g. CGO_ENABLED=0 cross-compilation).
type GPUEngine struct {
	Dimension int
}

// NewGPUEngine creates a new GPUEngine.
func NewGPUEngine(dim int) *GPUEngine {
	return &GPUEngine{Dimension: dim}
}

// CosineSimilarity parallelizes activation metric evaluations, mirroring
// gpu_cosine_similarity: sigmoid-scaled cosine similarity per concept direction.
func (ge *GPUEngine) CosineSimilarity(activation []float32, directions []float32, numDirections int) []float32 {
	if ge.Dimension <= 0 || ge.Dimension > math.MaxInt32 || numDirections <= 0 || numDirections > math.MaxInt32 ||
		len(activation) != ge.Dimension || len(directions) != ge.Dimension*numDirections {
		return nil
	}

	output := make([]float32, numDirections)

	var actNormSq float32
	for _, v := range activation {
		actNormSq += v * v
	}
	actNorm := float32(math.Sqrt(float64(actNormSq)))
	if actNorm < 1e-10 {
		actNorm = 1e-10
	}

	for d := 0; d < numDirections; d++ {
		var dot, dirNormSq float32
		offset := d * ge.Dimension
		for i := 0; i < ge.Dimension; i++ {
			dv := directions[offset+i]
			dot += activation[i] * dv
			dirNormSq += dv * dv
		}
		dirNorm := float32(math.Sqrt(float64(dirNormSq)))
		if dirNorm < 1e-10 {
			dirNorm = 1e-10
		}
		similarity := dot / (actNorm * dirNorm)
		output[d] = float32(1.0 / (1.0 + math.Exp(float64(-similarity*10.0))))
	}

	return output
}

// DynamicRotation rotates activation coordinates along concept manifolds,
// mirroring gpu_dynamic_rotation: norm-preserving rotation in the
// (direction, orthogonal) plane.
func (ge *GPUEngine) DynamicRotation(activation []float32, direction []float32, theta float32) []float32 {
	if ge.Dimension <= 0 || ge.Dimension > math.MaxInt32 ||
		len(activation) != ge.Dimension || len(direction) != ge.Dimension {
		return nil
	}

	output := make([]float32, ge.Dimension)
	copy(output, activation)

	var dirNormSq float32
	for _, v := range direction {
		dirNormSq += v * v
	}
	dirNorm := float32(math.Sqrt(float64(dirNormSq)))
	if dirNorm < 1e-10 {
		dirNorm = 1e-10
	}

	var proj float32
	for i := 0; i < ge.Dimension; i++ {
		proj += activation[i] * direction[i] / dirNorm
	}

	ortho := make([]float32, ge.Dimension)
	var orthoNormSq float32
	for i := 0; i < ge.Dimension; i++ {
		ortho[i] = activation[i] - proj*(direction[i]/dirNorm)
		orthoNormSq += ortho[i] * ortho[i]
	}
	orthoNorm := float32(math.Sqrt(float64(orthoNormSq)))
	if orthoNorm < 1e-10 {
		orthoNorm = 1e-10
	}

	cosT := float32(math.Cos(float64(theta)))
	sinT := float32(math.Sin(float64(theta)))
	for i := 0; i < ge.Dimension; i++ {
		normDir := direction[i] / dirNorm
		normOrtho := ortho[i] / orthoNorm
		output[i] = cosT*activation[i] + sinT*(orthoNorm*normDir-proj*normOrtho)
	}

	return output
}
