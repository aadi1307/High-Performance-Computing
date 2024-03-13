#include <iostream>
#include <omp.h>

// Function to calculate factorial
unsigned long long factorial(int n) {
    unsigned long long fact = 1;
    for (int i = 2; i <= n; i++) {
        fact *= i;
    }
    return fact;
}

int main() {
    // Set the number of threads
    omp_set_num_threads(4);

    // Variable to hold the actual number of threads used
    int actual_threads = 0;

    #pragma omp parallel
    {
        // The following block is executed by a single thread
        #pragma omp single
        {
            actual_threads = omp_get_num_threads();
            // Print the number of threads while still in the single block to avoid race conditions
            std::cout << "Number of threads: " << actual_threads << std::endl;
        }

        // Each thread introduces itself
        int thread_id = omp_get_thread_num();

        // Synchronize threads here to avoid mixing output
        #pragma omp critical
        {
            std::cout << "I am thread No. " << thread_id << std::endl;
        }

        // Each thread computes factorial for numbers from 1 to 8
        #pragma omp for schedule(static) nowait
        for (int i = 1; i <= 8; ++i) {
            unsigned long long fact = factorial(i);
            // Critical section to avoid mixing output
            #pragma omp critical
            {
                std::cout << i << "!=" << fact << std::endl;
            }
        }
    }

    return 0;
}
