#include "count.cuh"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/copy.h>
#include <iostream>
#include <cstdlib>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " N" << std::endl;
        return 1;
    }

    int N = std::atoi(argv[1]);
    thrust::host_vector<int> h_in(N);

    // Generate random numbers in the range [0, 500]
    thrust::default_random_engine rng;
    thrust::uniform_int_distribution<int> dist(0, 500);
    for (int& val : h_in) {
        val = dist(rng);
    }

    // Copy data from host to device
    thrust::device_vector<int> d_in = h_in;

    // Prepare device vectors for output
    thrust::device_vector<int> values;
    thrust::device_vector<int> counts;

    // For measuring time taken by the 'count' function
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // Call the 'count' function
    count(d_in, values, counts);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Print the results
    std::cout << values.back() << std::endl; // the last element of 'values'
    std::cout << counts.back() << std::endl; // the last element of 'counts'
    std::cout << milliseconds << std::endl;  // time taken in milliseconds

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
