#include <iostream>
#include <cuda.h>
#include <curand_kernel.h>
#include "matmul.cuh"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: ./task1 n threads_per_block" << std::endl;
        return 1;
    }

    int n = std::atoi(argv[1]);
    int threads_per_block = std::atoi(argv[2]);

    float* A = new float[n * n];
    float* B = new float[n * n];
    float* C = new float[n * n];

    for (int i = 0; i < n * n; i++) {
        A[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        B[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, n * n * sizeof(float));
    cudaMalloc(&d_B, n * n * sizeof(float));
    cudaMalloc(&d_C, n * n * sizeof(float));

    cudaMemcpy(d_A, A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * n * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul(d_A, d_B, d_C, n, threads_per_block);
    cudaEventRecord(stop);

    cudaMemcpy(C, d_C, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << C[n * n - 1] << std::endl;
    std::cout << milliseconds << std::endl;

    delete[] A;
    delete[] B;
    delete[] C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
