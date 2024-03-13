#include "matmul.h"
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <random>

void fill_matrix(float* M, std::size_t n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (std::size_t i = 0; i < n * n; ++i) {
        M[i] = dis(gen);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <n> <t>" << std::endl;
        return 1;
    }

    const std::size_t n = static_cast<std::size_t>(std::atoi(argv[1]));
    const int t = std::atoi(argv[2]);

    omp_set_num_threads(t);

    float* A = new float[n * n];
    float* B = new float[n * n];
    float* C = new float[n * n]();

    fill_matrix(A, n);
    fill_matrix(B, n);

    auto start = std::chrono::high_resolution_clock::now();
    
    mmul(A, B, C, n);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << C[0] << std::endl;
    std::cout << C[n * n - 1] << std::endl;
    std::cout << elapsed.count() << std::endl;

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
