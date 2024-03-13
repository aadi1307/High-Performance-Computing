#include <iostream>
#include <vector>
#include <chrono>
#include "matmul.h"

int main() {
    const int n = 1024;
    std::vector<double> A(n * n, 1.0); // fill with ones
    std::vector<double> B(n * n, 2.0); // fill with twos
    std::vector<double> C(n * n, 0.0); // fill with zeros

    // Print the number of rows
    std::cout << n << std::endl;

    auto timeIt = [&](auto func) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        std::cout << elapsed.count() << std::endl;
        std::cout << C.back() << std::endl;
        std::fill(C.begin(), C.end(), 0.0);  // Reset C
    };

    // mmul1
    timeIt([&]() { mmul1(A.data(), B.data(), C.data(), n); });

    // mmul2
    timeIt([&]() { mmul2(A.data(), B.data(), C.data(), n); });

    // mmul3
    timeIt([&]() { mmul3(A.data(), B.data(), C.data(), n); });

    // mmul4
    timeIt([&]() { mmul4(A, B, C, n); });

    return 0;
}
