#include <iostream>
#include <chrono>
#include "optimize.h"
#include <cstdlib>  // For std::atoi

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <n>" << std::endl;
        return 1;
    }
    
    size_t n = static_cast<size_t>(std::atoi(argv[1]));
    vec v(n);
    v.data = new data_t[n];
    
    // Fill the vector with data
    for (size_t i = 0; i < n; i++) {
        v.data[i] = static_cast<data_t>(i); // Simple example to avoid overflow
    }
    
    data_t dest;
    
    // Define an array of function pointers to the optimize functions
    void (*optimize_functions[])(vec*, data_t*) = {optimize1, optimize2, optimize3, optimize4, optimize5};
    
    for (auto& optimize : optimize_functions) {
        auto start_time = std::chrono::high_resolution_clock::now();
        optimize(&v, &dest);
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end_time - start_time;
        
        std::cout << dest << std::endl;
        std::cout << elapsed.count() << std::endl;
    }
    
    delete[] v.data;
    return 0;
}
