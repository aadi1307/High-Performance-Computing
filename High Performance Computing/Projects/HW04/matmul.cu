#include <cuda.h>
#include "matmul.cuh"

__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n * n) {
        int row = idx / n;
        int col = idx % n;
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[idx] = sum;
    }
}

void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block) {
    dim3 blocks((n * n + threads_per_block - 1) / threads_per_block);
    matmul_kernel<<<blocks, threads_per_block>>>(A, B, C, n);
    cudaDeviceSynchronize(); 
}
