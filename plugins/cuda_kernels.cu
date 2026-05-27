#include <cuda_runtime.h>
#include <math.h>

#define WARP_SIZE 32

// Warp-level reduction for sums
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) 
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Compute parallel cosine similarity scores on GPU VRAM
__global__ void cosine_similarity_kernel(const float* activation, const float* directions, int dim, int num_directions, float* output_scores) {
    int dir_idx = blockIdx.x; // Each block handles one direction
    if (dir_idx >= num_directions) return;

    const float* current_dir = directions + (dir_idx * dim);
    
    float dot_sum = 0.0f;
    float norm_a = 0.0f;
    float norm_d = 0.0f;

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float a = activation[i];
        float d = current_dir[i];
        dot_sum += a * d;
        norm_a += a * a;
        norm_d += d * d;
    }

    // Warp reduction
    dot_sum = warpReduceSum(dot_sum);
    norm_a = warpReduceSum(norm_a);
    norm_d = warpReduceSum(norm_d);

    // Block reduction
    __shared__ float shared_dot[32];
    __shared__ float shared_na[32];
    __shared__ float shared_nd[32];

    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    if (lane == 0) {
        shared_dot[wid] = dot_sum;
        shared_na[wid] = norm_a;
        shared_nd[wid] = norm_d;
    }
    __syncthreads();

    if (wid == 0) {
        dot_sum = (lane < blockDim.x / WARP_SIZE) ? shared_dot[lane] : 0.0f;
        norm_a = (lane < blockDim.x / WARP_SIZE) ? shared_na[lane] : 0.0f;
        norm_d = (lane < blockDim.x / WARP_SIZE) ? shared_nd[lane] : 0.0f;
        
        dot_sum = warpReduceSum(dot_sum);
        norm_a = warpReduceSum(norm_a);
        norm_d = warpReduceSum(norm_d);
    }

    if (threadIdx.x == 0) {
        float denominator = sqrtf(norm_a) * sqrtf(norm_d);
        output_scores[dir_idx] = (denominator == 0.0f) ? 0.0f : (dot_sum / denominator);
    }
}

// Execute norm-preserving 2D dynamic coordinate rotations directly inside GPU space
__global__ void dynamic_rotation_kernel(float* activation, const float* direction, int dim, float theta, float* output_activation) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;

    float cos_theta = cosf(theta);
    float sin_theta = sinf(theta);

    float a = activation[i];
    float d = direction[i];

    // Simple rotation along the direction vector
    output_activation[i] = a * cos_theta + d * sin_theta;
}

extern "C" {
    void gpu_cosine_similarity(const float* activation, const float* directions, int dim, int num_directions, float* output_scores) {
        int threadsPerBlock = 256;
        int blocks = num_directions;
        cosine_similarity_kernel<<<blocks, threadsPerBlock>>>(activation, directions, dim, num_directions, output_scores);
    }

    void gpu_dynamic_rotation(float* activation, const float* direction, int dim, float theta, float* output_activation) {
        int threadsPerBlock = 256;
        int blocks = (dim + threadsPerBlock - 1) / threadsPerBlock;
        dynamic_rotation_kernel<<<blocks, threadsPerBlock>>>(activation, direction, dim, theta, output_activation);
    }
}
