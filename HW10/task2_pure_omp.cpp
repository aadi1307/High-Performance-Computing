#include "reduce.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

// Helper function to generate random float values in a given range
void fill_with_random_floats(std::vector<float>& arr, float low, float high) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> dist(low, high);

    for (float& elem : arr) {
        elem = dist(mt);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <array_size> <num_threads>\n";
        return 1;
    }

    const size_t n = std::strtoul(argv[1], nullptr, 10);
    const int num_threads = std::atoi(argv[2]);
    std::vector<float> arr(n);

    // Fill the array with random numbers
    fill_with_random_floats(arr, -10.0, 10.0);

    // Set the number of threads for OpenMP
    omp_set_num_threads(num_threads);

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // Perform the reduction
    float sum = reduce(arr.data(), 0, n);

    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();

    // Calculate the time taken
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // Output the result
    std::cout << sum << std::endl;
    std::cout << duration << std::endl;

    return 0;
}
