#include "matmul.cuh"
#include <cuda_runtime.h>

// Define the tile width (can be set to any optimal value, for example, 32)
#define TILE_WIDTH 32

template<typename T>
__global__ void matmul_kernel(const T *A, const T *B, T *C, unsigned int n) {
    __shared__ T As[TILE_WIDTH][TILE_WIDTH];
    __shared__ T Bs[TILE_WIDTH][TILE_WIDTH];

    unsigned int bx = blockIdx.x, by = blockIdx.y;
    unsigned int tx = threadIdx.x, ty = threadIdx.y;

    // Identify the row and column of the result matrix to work on
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    T value = 0;

    // Loop over tiles
    for (int m = 0; m < (n-1)/TILE_WIDTH + 1; ++m) {
        if (row < n && m*TILE_WIDTH+tx < n)
            As[ty][tx] = A[row*n + m*TILE_WIDTH+tx];
        else
            As[ty][tx] = 0;

        if (col < n && m*TILE_WIDTH+ty < n)
            Bs[ty][tx] = B[(m*TILE_WIDTH+ty)*n + col];
        else
            Bs[ty][tx] = 0;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            value += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }

    if (row < n && col < n)
        C[row*n + col] = value;
}

// Define the host functions for each matrix type
void matmul_1(const int *A, const int *B, int *C, unsigned int n, unsigned int block_dim) {
    dim3 blockDim(block_dim, block_dim);
    dim3 gridDim((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);
    matmul_kernel<int><<<gridDim, blockDim>>>(A, B, C, n);
    cudaDeviceSynchronize();
}

void matmul_2(const float *A, const float *B, float *C, unsigned int n, unsigned int block_dim) {
    dim3 blockDim(block_dim, block_dim);
    dim3 gridDim((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);
    matmul_kernel<float><<<gridDim, blockDim>>>(A, B, C, n);
    cudaDeviceSynchronize();
}

void matmul_3(const double *A, const double *B, double *C, unsigned int n, unsigned int block_dim) {
    dim3 blockDim(block_dim, block_dim);
    dim3 gridDim((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);
    matmul_kernel<double><<<gridDim, blockDim>>>(A, B, C, n);
    cudaDeviceSynchronize();
}
