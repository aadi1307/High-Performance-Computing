// task3.cu

#include <iostream>
#include <random>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>
#include "vscale.cuh"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <n>" << std::endl;
        return EXIT_FAILURE;
    }

    unsigned int n = std::atoi(argv[1]);

    float* a_host = new float[n];
    float* b_host = new float[n];
    float* a_device;
    float* b_device;

    std::mt19937 gen;
    std::uniform_real_distribution<float> dist_a(-10.0f, 10.0f);
    std::uniform_real_distribution<float> dist_b(0.0f, 1.0f);

    for (unsigned int i = 0; i < n; ++i) {
        a_host[i] = dist_a(gen);
        b_host[i] = dist_b(gen);
    }

    cudaMalloc(&a_device, n * sizeof(float));
    cudaMalloc(&b_device, n * sizeof(float));

    cudaMemcpy(a_device, a_host, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_device, b_host, n * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int threadsPerBlock = 512;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    cudaEventRecord(start);
    vscale<<<blocksPerGrid, threadsPerBlock>>>(a_device, b_device, n);
    cudaEventRecord(stop);

    cudaMemcpy(b_host, b_device, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << milliseconds << std::endl;
    std::cout << b_host[0] << std::endl;
    std::cout << b_host[n-1] << std::endl;

    delete[] a_host;
    delete[] b_host;
    cudaFree(a_device);
    cudaFree(b_device);

    return 0;
}
