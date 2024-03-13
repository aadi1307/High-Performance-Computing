#include "stencil.cuh"
#include <cuda_runtime.h>

__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, unsigned int R) {
    extern __shared__ float shared_data[];

    float *shared_mask = shared_data;
    float *shared_image = &shared_data[2 * R + 1];

    int global_idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Load mask into shared memory
    if (threadIdx.x < 2 * R + 1) {
        shared_mask[threadIdx.x] = mask[threadIdx.x];
    }

    // Load image data into shared memory with boundary check
    shared_image[threadIdx.x] = (global_idx < n + 2*R) ? image[global_idx] : 1.0f;

    __syncthreads();

    // Calculate convolution
    float sum = 0.0f;
    for (int j = -static_cast<int>(R); j <= -static_cast<int>(R); j++) {
        int idx = global_idx + j;
        float img_val = (idx >= 0 && idx < n + 2*R) ? shared_image[threadIdx.x + j] : 1.0f;
        sum += img_val * shared_mask[j + R];
    }

    if (global_idx < n) {
        output[global_idx] = sum;
    }
}

__host__ void stencil(const float* image, const float* mask, float* output, unsigned int n, unsigned int R, unsigned int threads_per_block) {
    unsigned int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    unsigned int shared_mem_size = (threads_per_block + 4 * R) * sizeof(float);

    stencil_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(image, mask, output, n, R);
    cudaDeviceSynchronize();
}
