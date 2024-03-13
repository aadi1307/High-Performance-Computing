#define CUB_STDERR // Activate CUB's error logging to the standard error stream
#include <stdio.h>
#include <iostream>
#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>
#include <random>
#include <chrono>
#include <cstdlib> // Necessary for utilizing the std::atoi function

// Create a shortcut to the CUB namespace to avoid constant prefixing
using namespace cub;

// Instantiate an allocator for device-side memory management
CachingDeviceAllocator g_allocator(true);

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage instructions: " << argv[0] << " N" << std::endl;
        return 1;
    }

    int N = std::atoi(argv[1]); // Convert command-line argument to integer for array size
    if (N <= 0) {
        std::cerr << "Error message: The number of elements should be positive." << std::endl;
        return 1;
    }

    // Setting up the mechanism for generating random numbers
    std::random_device rd;  
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    // Establish a host-side array and populate it with random floats
    std::vector<float> h_in(N);
    for(auto& num : h_in) {
        num = dis(gen);
    }

    // Reserve memory on the device for input data
    float* d_in = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in, sizeof(float) * N));
    CubDebugExit(cudaMemcpy(d_in, h_in.data(), sizeof(float) * N, cudaMemcpyHostToDevice));

    // Allocate memory on the device for storing the reduction result
    float* d_sum = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_sum, sizeof(float)));

    // Prepare and allocate temporary storage on the device
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, N));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // Initialize CUDA events used for timing the execution
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    // Mark the beginning of the timed section
    cudaEventRecord(start_event, NULL);

    // Execute the reduction operation on the device, skipping CUB's usual error-checking for purer timing, used chatgpt for how the sum is calculated and storage stored
    DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, N);

    // Mark the end of the timed section and synchronize to ensure completion
    cudaEventRecord(stop_event, NULL);
    cudaEventSynchronize(stop_event);

    // Calculate and retrieve the elapsed time in milliseconds
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);

    // Fetch and display the outcome of the reduction
    float gpu_sum = 0;
    CubDebugExit(cudaMemcpy(&gpu_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << gpu_sum << std::endl;
    std::cout << milliseconds << std::endl;

    // Free up the allocated resources and conclude the program
    if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
    if (d_sum) CubDebugExit(g_allocator.DeviceFree(d_sum));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    return 0;
}
