package native

/*
#cgo CXXFLAGS: -std=c++11
#cgo LDFLAGS: -lm
#include "kernel.h"
*/
import "C"
import "unsafe"

// GPUEngine handles direct execution of high-performance activation operations on GPU VRAM.
type GPUEngine struct {
	Dimension int
}

// NewGPUEngine creates a new GPUEngine.
func NewGPUEngine(dim int) *GPUEngine {
	return &GPUEngine{Dimension: dim}
}

// CosineSimilarity parallelizes activation metric evaluations directly within GPU kernels.
func (ge *GPUEngine) CosineSimilarity(activation []float32, directions []float32, numDirections int) []float32 {
	if len(activation) != ge.Dimension || len(directions) != ge.Dimension*numDirections || numDirections <= 0 {
		return nil
	}

	output := make([]float32, numDirections)

	dim32 := toInt32(ge.Dimension)
	numDir32 := toInt32(numDirections)

	C.gpu_cosine_similarity(
		(*C.float)(unsafe.Pointer(&activation[0])),
		(*C.float)(unsafe.Pointer(&directions[0])),
		C.int(dim32),
		C.int(numDir32),
		(*C.float)(unsafe.Pointer(&output[0])),
	)

	return output
}

// DynamicRotation rotates activation coordinates along Concept manifolds inside inline GPU shader contexts.
func (ge *GPUEngine) DynamicRotation(activation []float32, direction []float32, theta float32) []float32 {
	if len(activation) != ge.Dimension || len(direction) != ge.Dimension {
		return nil
	}

	output := make([]float32, ge.Dimension)
	copy(output, activation)

	dim32 := toInt32(ge.Dimension)

	C.gpu_dynamic_rotation(
		(*C.float)(unsafe.Pointer(&output[0])),
		(*C.float)(unsafe.Pointer(&direction[0])),
		C.int(dim32),
		C.float(theta),
		(*C.float)(unsafe.Pointer(&output[0])),
	)

	return output
}
