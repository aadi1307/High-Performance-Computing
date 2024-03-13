#include "reduce.h"
#include <iostream>
#include <random>
#include <chrono>
#include <cstdlib>  // For std::strtoul

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <n> <t>" << std::endl;
        return 1;
    }

    char* end;
    size_t n = std::strtoul(argv[1], &end, 10);
    int t = std::atoi(argv[2]);

    // Initialize the OpenMP environment
    omp_set_num_threads(t);

    // Create and fill the array with random numbers in the range [-1.0, 1.0]
    std::vector<float> arr(n);
    std::default_random_engine engine;
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < n; ++i) {
        arr[i] = dist(engine);
    }

    // Start timing the reduction process
    auto start_time = std::chrono::high_resolution_clock::now();

    // Call the reduce function
    float global_res = reduce(arr.data(), 0, n / 2) + reduce(arr.data() + n / 2, n / 2, n);

    // End timing the reduction process
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end_time - start_time;

    // Print the global result from one process
    std::cout << global_res << std::endl;

    // Print the time taken for the entire reduction process
    std::cout << elapsed.count() << std::endl;

    return 0;
}
