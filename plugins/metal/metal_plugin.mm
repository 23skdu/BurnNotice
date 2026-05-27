#include "kernel.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <vector>

static id<MTLDevice> device = nil;
static id<MTLCommandQueue> commandQueue = nil;
static id<MTLComputePipelineState> cosinePipeline = nil;
static id<MTLComputePipelineState> rotationPipeline = nil;

// Embed MSL Source for runtime compilation to avoid bundle/file-system dependencies
static NSString* const kShaderSource = @"\n"
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"kernel void cosine_similarity_kernel(\n"
"    device const float* activation [[buffer(0)]],\n"
"    device const float* directions [[buffer(1)]],\n"
"    device float* output_scores [[buffer(2)]],\n"
"    constant int& dim [[buffer(3)]],\n"
"    constant int& num_directions [[buffer(4)]],\n"
"    uint2 lid [[thread_position_in_threadgroup]],\n"
"    uint2 tgid [[threadgroup_position_in_grid]],\n"
"    uint threads_per_threadgroup [[threads_per_threadgroup]]\n"
") {\n"
"    int dir_idx = tgid.x;\n"
"    if (dir_idx >= num_directions) return;\n"
"    device const float* current_dir = directions + (dir_idx * dim);\n"
"    float dot_sum = 0.0f;\n"
"    float norm_a = 0.0f;\n"
"    float norm_d = 0.0f;\n"
"    for (int i = lid.x; i < dim; i += threads_per_threadgroup) {\n"
"        float a = activation[i];\n"
"        float d = current_dir[i];\n"
"        dot_sum += a * d;\n"
"        norm_a += a * a;\n"
"        norm_d += d * d;\n"
"    }\n"
"    float simd_dot = simd_sum(dot_sum);\n"
"    float simd_na = simd_sum(norm_a);\n"
"    float simd_nd = simd_sum(norm_d);\n"
"    if (lid.x == 0) {\n"
"        float denominator = sqrt(simd_na) * sqrt(simd_nd);\n"
"        output_scores[dir_idx] = (denominator == 0.0f) ? 0.0f : (simd_dot / denominator);\n"
"    }\n"
"}\n"
"\n"
"kernel void dynamic_rotation_kernel(\n"
"    device float* activation [[buffer(0)]],\n"
"    device const float* direction [[buffer(1)]],\n"
"    device float* output_activation [[buffer(2)]],\n"
"    constant int& dim [[buffer(3)]],\n"
"    constant float& theta [[buffer(4)]],\n"
"    uint gid [[thread_position_in_grid]]\n"
") {\n"
"    if (gid >= (uint)dim) return;\n"
"    float cos_theta = cos(theta);\n"
"    float sin_theta = sin(theta);\n"
"    float a = activation[gid];\n"
"    float d = direction[gid];\n"
"    output_activation[gid] = a * cos_theta + d * sin_theta;\n"
"}\n";


void initialize_metal() {
    if (device != nil) return; // Already initialized

    device = MTLCreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "Failed to find a suitable Metal device." << std::endl;
        return;
    }

    commandQueue = [device newCommandQueue];

    // Follow ARC memory best practices with local error
    __block NSError *localError = nil;
    
    // Runtime compilation
    id<MTLLibrary> library = [device newLibraryWithSource:kShaderSource options:nil error:&localError];
    if (localError) {
        std::cerr << "Failed to compile Metal shader: " << [[localError localizedDescription] UTF8String] << std::endl;
        return;
    }

    id<MTLFunction> cosineFunc = [library newFunctionWithName:@"cosine_similarity_kernel"];
    id<MTLFunction> rotationFunc = [library newFunctionWithName:@"dynamic_rotation_kernel"];

    localError = nil;
    cosinePipeline = [device newComputePipelineStateWithFunction:cosineFunc error:&localError];
    if (localError) {
        std::cerr << "Failed to create cosine pipeline: " << [[localError localizedDescription] UTF8String] << std::endl;
        return;
    }

    localError = nil;
    rotationPipeline = [device newComputePipelineStateWithFunction:rotationFunc error:&localError];
    if (localError) {
        std::cerr << "Failed to create rotation pipeline: " << [[localError localizedDescription] UTF8String] << std::endl;
        return;
    }
}

extern "C" {

void gpu_cosine_similarity(const float* activation, const float* directions, int dim, int num_directions, float* output_scores) {
    @autoreleasepool {
        initialize_metal();
        if (!cosinePipeline) return;

        size_t act_size = dim * sizeof(float);
        size_t dir_size = num_directions * dim * sizeof(float);
        size_t out_size = num_directions * sizeof(float);

        id<MTLBuffer> actBuffer = [device newBufferWithBytes:activation length:act_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> dirBuffer = [device newBufferWithBytes:directions length:dir_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> outBuffer = [device newBufferWithLength:out_size options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:cosinePipeline];
        [encoder setBuffer:actBuffer offset:0 atIndex:0];
        [encoder setBuffer:dirBuffer offset:0 atIndex:1];
        [encoder setBuffer:outBuffer offset:0 atIndex:2];
        [encoder setBytes:&dim length:sizeof(int) atIndex:3];
        [encoder setBytes:&num_directions length:sizeof(int) atIndex:4];

        MTLSize gridSize = MTLSizeMake(1, num_directions, 1);
        NSUInteger threadGroupSize = cosinePipeline.maxTotalThreadsPerThreadgroup;
        if (threadGroupSize > (NSUInteger)dim) {
            threadGroupSize = dim;
        }
        MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Copy back results
        memcpy(output_scores, [outBuffer contents], out_size);
    }
}

void gpu_dynamic_rotation(float* activation, const float* direction, int dim, float theta, float* output_activation) {
    @autoreleasepool {
        initialize_metal();
        if (!rotationPipeline) return;

        size_t size = dim * sizeof(float);

        id<MTLBuffer> actBuffer = [device newBufferWithBytes:activation length:size options:MTLResourceStorageModeShared];
        id<MTLBuffer> dirBuffer = [device newBufferWithBytes:direction length:size options:MTLResourceStorageModeShared];
        id<MTLBuffer> outBuffer = [device newBufferWithLength:size options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:rotationPipeline];
        [encoder setBuffer:actBuffer offset:0 atIndex:0];
        [encoder setBuffer:dirBuffer offset:0 atIndex:1];
        [encoder setBuffer:outBuffer offset:0 atIndex:2];
        [encoder setBytes:&dim length:sizeof(int) atIndex:3];
        [encoder setBytes:&theta length:sizeof(float) atIndex:4];

        NSUInteger w = rotationPipeline.maxTotalThreadsPerThreadgroup;
        if (w > (NSUInteger)dim) {
            w = dim;
        }
        MTLSize threadgroupSize = MTLSizeMake(w, 1, 1);
        MTLSize gridSize = MTLSizeMake((dim + w - 1) / w, 1, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Copy back to output activation buffer
        memcpy(output_activation, [outBuffer contents], size);
    }
}

} // extern "C"
