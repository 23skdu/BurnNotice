#ifndef NATIVE_KERNEL_H
#define NATIVE_KERNEL_H

#ifdef __cplusplus
extern "C" {
#endif

// Computes parallel cosine similarity scores on GPU VRAM for the given activation and concept directions.
void gpu_cosine_similarity(const float* activation, const float* directions, int dim, int num_directions, float* output_scores);

// Executes norm-preserving 2D dynamic coordinate rotations directly inside GPU space (Metal/CUDA shader hook).
void gpu_dynamic_rotation(float* activation, const float* direction, int dim, float theta, float* output_activation);

#ifdef __cplusplus
}
#endif

#endif // NATIVE_KERNEL_H
