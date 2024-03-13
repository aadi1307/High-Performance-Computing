#include <vector>
#include <random>
#include <iomanip>
#include "mmul.h"
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " n n_tests" << std::endl;
        return 1;
    }

    int n = std::atoi(argv[1]);
    int n_tests = std::atoi(argv[2]);
    size_t total_elements = n * n;
    size_t bytes = total_elements * sizeof(float);

    // Use the default random engine and a uniform distribution to generate random floats
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-1.0, 1.0);

    // Create three vectors to hold the matrix data
    std::vector<float> h_A(total_elements);
    std::vector<float> h_B(total_elements);
    std::vector<float> h_C(total_elements);

    // Fill the vectors with random data
    for (size_t i = 0; i < total_elements; ++i) {
        h_A[i] = distribution(generator);
        h_B[i] = distribution(generator);
        h_C[i] = distribution(generator); // If C is the output matrix, this initialization is not necessary
    }

    // Pointers for device memory
    float *d_A, *d_B, *d_C;

    // Allocate device memory
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Transfer data from host to device memory
    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C.data(), bytes, cudaMemcpyHostToDevice); // If C is the output matrix, this memcpy is not necessary

    cublasHandle_t handle;
    cublasCreate(&handle);

    // Timing using CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Conduct all tests
    cudaEventRecord(start);
    for (int i = 0; i < n_tests; ++i) {
        mmul(handle, d_A, d_B, d_C, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time in milliseconds
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << std::fixed << std::setprecision(6) << (milliseconds / n_tests) << std::endl;

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
