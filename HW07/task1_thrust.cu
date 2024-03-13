#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <iostream>
#include <random>
#include <cstdlib> // Needed for the std::atoi function

int main(int argc, char* argv[]) {
    // Verify the correct number of command-line arguments
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " N" << std::endl;
        return 1;
    }

    // Convert the command line argument to an integer
    int N = std::atoi(argv[1]);
    // Check for a valid positive integer for the array size
    if (N <= 0) {
        std::cerr << "Error: The number of elements should be a positive integer." << std::endl;
        return 1;
    }

    // Prepare for random number generation
    std::random_device rd;  // Obtain a random number from hardware
    std::mt19937 gen(rd()); // Seed the generator
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f); // Uniform distribution between -1.0 and 1.0

    // Initialize host memory and populate with random numbers
    thrust::host_vector<float> h_vec(N);
    // Fill the vector with random data
    thrust::generate(h_vec.begin(), h_vec.end(), [&gen, &dis] { return dis(gen); });

    // Move the data from host memory to device memory
    thrust::device_vector<float> d_vec = h_vec;

    // Set up CUDA events for timing the operation
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Mark the start point
    cudaEventRecord(start);

    // Execute reduction on the GPU, summing up the array
    float result = thrust::reduce(d_vec.begin(), d_vec.end(), 0.0f, thrust::plus<float>());

    // Mark the end point
    cudaEventRecord(stop);
    // Ensure all operations are completed before proceeding
    cudaEventSynchronize(stop);

    // Determine the elapsed time for the reduction operation
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Release the event resources
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Display the total from the reduction
    std::cout << result << std::endl;

    // Print the operation time in milliseconds
    std::cout << milliseconds << std::endl;

    return 0;
}
