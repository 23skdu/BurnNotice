#include <metal_stdlib>

using namespace metal;

// Compute parallel cosine similarity scores on GPU VRAM
kernel void cosine_similarity_kernel(
    device const float* activation [[buffer(0)]],
    device const float* directions [[buffer(1)]],
    device float* output_scores [[buffer(2)]],
    constant int& dim [[buffer(3)]],
    constant int& num_directions [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]
) {
    int dir_idx = tgid.x; // Each threadgroup handles one direction
    if (dir_idx >= num_directions) return;

    device const float* current_dir = directions + (dir_idx * dim);

    float dot_sum = 0.0f;
    float norm_a = 0.0f;
    float norm_d = 0.0f;

    // Strided loop across the dimension
    for (int i = lid.x; i < dim; i += threads_per_threadgroup) {
        float a = activation[i];
        float d = current_dir[i];
        dot_sum += a * d;
        norm_a += a * a;
        norm_d += d * d;
    }

    // SIMD group reduction
    float simd_dot = simd_sum(dot_sum);
    float simd_na = simd_sum(norm_a);
    float simd_nd = simd_sum(norm_d);

    // Write result from thread 0
    if (lid.x == 0) {
        float denominator = sqrt(simd_na) * sqrt(simd_nd);
        output_scores[dir_idx] = (denominator == 0.0f) ? 0.0f : (simd_dot / denominator);
    }
}

// Execute norm-preserving 2D dynamic coordinate rotations directly inside GPU space
kernel void dynamic_rotation_kernel(
    device float* activation [[buffer(0)]],
    device const float* direction [[buffer(1)]],
    device float* output_activation [[buffer(2)]],
    constant int& dim [[buffer(3)]],
    constant float& theta [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= (uint)dim) return;

    float cos_theta = cos(theta);
    float sin_theta = sin(theta);

    float a = activation[gid];
    float d = direction[gid];

    output_activation[gid] = a * cos_theta + d * sin_theta;
}
