#include "reduce.cuh"
#include <iostream>
#include <cuda_runtime.h>
#include <random>
#include <cuda.h>

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./task2 N threads_per_block\n";
        return 1;
    }

    unsigned int N = atoi(argv[1]);
    unsigned int threads_per_block = atoi(argv[2]);

    float *host_data = new float[N];
    float *device_data, *device_output;
    cudaMalloc((void**)&device_data, N * sizeof(float));
    cudaMalloc((void**)&device_output, (N + threads_per_block * 2 - 1) / (threads_per_block * 2) * sizeof(float));

    // Using the <random> library to fill the array with random numbers in the range [-1, 1]
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (unsigned int i = 0; i < N; ++i) {
        host_data[i] = static_cast<float>(dis(gen));
    }

    cudaMemcpy(device_data, host_data, N * sizeof(float), cudaMemcpyHostToDevice);

    // Call the reduction function and time it using cudaEvent_t
    float *device_input = device_data, *device_out = device_output;
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent, 0);
    reduce(&device_input, &device_out, N, threads_per_block);
    cudaEventRecord(stopEvent, 0);

    cudaEventSynchronize(stopEvent);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    //std::cout << elapsedTime << std::endl;

    // Print the result
    cudaMemcpy(host_data, device_input, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << host_data[0] << std::endl;
    std::cout << elapsedTime << std::endl;

    // Cleanup
    delete[] host_data;
    cudaFree(device_data);
    cudaFree(device_output);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    return 0;
}
