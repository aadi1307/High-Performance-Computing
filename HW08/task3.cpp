#include "msort.h"
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <random>

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <n> <t> <ts>" << std::endl;
        return 1;
    }

    const std::size_t n = static_cast<std::size_t>(std::atoi(argv[1]));
    const int t = std::atoi(argv[2]);
    const std::size_t threshold = static_cast<std::size_t>(std::atoi(argv[3]));

    omp_set_num_threads(t);

    // Create and fill the array with random numbers
    std::vector<int> arr(n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(-1000, 1000);

    for (int &value : arr) {
        value = dis(gen);
    }

    auto start = std::chrono::high_resolution_clock::now();
    
    // Call the msort function
    msort(arr.data(), n, threshold);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << arr.front() << std::endl;
    std::cout << arr.back() << std::endl;
    std::cout << elapsed.count() << std::endl;

    return 0;
}
