#include "scan.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <ctime>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "integer c." << std::endl;
        return 1;
    }

    int c = std::atoi(argv[1]);

    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // array of c between -1.0 and 1.0.
    std::vector<float> arr(c);
    for (int i = 0; i < c; ++i) {
        arr[i] = (2.0 * rand() / RAND_MAX) - 1.0;  
    }

    //./ Scanning array.
    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<float> result = scan(arr);
    auto end_time = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << duration << std::endl;

    if (!result.empty()) {
        std::cout << result.front() << std::endl;
    }

    if (!result.empty()) {
        std::cout << result.back() << std::endl;
    }


    return 0;
}
