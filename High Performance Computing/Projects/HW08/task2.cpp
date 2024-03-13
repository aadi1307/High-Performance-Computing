// task2.cpp
#include "convolution.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cstdlib>
#include <omp.h>

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <n> <t>" << std::endl;
        return 1;
    }

    const std::size_t n = static_cast<std::size_t>(std::atoi(argv[1]));
    const int t = std::atoi(argv[2]);

    omp_set_num_threads(t);

    std::vector<float> image(n * n);
    // Example mask values for a 3x3 convolution mask
    std::vector<float> mask = { 0.0f, 1.0f, 0.0f,
                                1.0f, -4.0f, 1.0f,
                                0.0f, 1.0f, 0.0f };
    std::vector<float> output(n * n);

    // Fill image with random float numbers
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (float &value : image) {
        value = dis(gen);
    }

    auto start = std::chrono::high_resolution_clock::now();
    
    convolve(image.data(), output.data(), n, mask.data(), 3);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << output.front() << std::endl;
    std::cout << output.back() << std::endl;
    std::cout << elapsed.count() << std::endl;

    return 0;
}
