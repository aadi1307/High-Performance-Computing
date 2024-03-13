// task2.cpp

#include "convolution.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <cstdlib>

int main(int argc, char** argv) {
    // Check if the program has the correct number of arguments
    if (argc != 3) {
        std::cerr << "Usage: ./task2 n m" << std::endl;
        return 1;
    }

    // Parse the command-line arguments
    int n = std::stoi(argv[1]);
    int m = std::stoi(argv[2]);

    // 1) Initialize and fill the image matrix with random values between -10 and 10
    std::vector<float> image(n * n);
    for (float& val : image) {
        val = (rand() / (float)RAND_MAX) * 20.0f - 10.0f;
    }

    // 2) Initialize and fill the mask matrix with random values between -1 and 1
    std::vector<float> mask(m * m);
    for (float& val : mask) {
        val = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }

    // 3) Measure the time taken to convolve the image with the mask
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<float> convolved = convolve(image, n, mask, m);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end - start;

    // 4) Display results
    std::cout << elapsed_time.count() * 1000.0 << std::endl;  // Print the time in milliseconds
    std::cout << convolved.front() << std::endl;              // Print the first element of the convolved image
    std::cout << convolved.back() << std::endl;               // Print the last element of the convolved image

    return 0;
}
