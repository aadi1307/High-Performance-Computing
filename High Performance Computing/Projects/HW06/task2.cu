#include "scan.cuh"
#include <vector>
#include <cuda_runtime.h>
#include <iostream>
#include <random>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " n threads_per_block" << std::endl;
        return 1;
    }

    int n = std::atoi(argv[1]);
    int threads_per_block = std::atoi(argv[2]);

    // Allocate managed memory for input and output
    float *input, *output;
    cudaMallocManaged(&input, n * sizeof(float));
    cudaMallocManaged(&output, n * sizeof(float));

    // Fill the input array with random numbers
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    for (int i = 0; i < n; i++) {
        input[i] = dis(gen);
    }

    // Record the start event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, nullptr);
    // Execute the scan function
    scan(input, output, n, threads_per_block);
    cudaDeviceSynchronize(); // Wait for compute device to finish
    cudaEventRecord(stop, nullptr);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Print the last element of output array, which is the total sum for an inclusive scan
    std::cout << output[n-1] << std::endl;
    std::cout << milliseconds << std::endl;

    // Free memory and destroy events
    cudaFree(input);
    cudaFree(output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
