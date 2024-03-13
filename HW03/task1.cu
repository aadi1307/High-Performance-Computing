#include <cstdio>
#include <cuda_runtime.h>

// CUDA kernel to compute and print factorials
__global__ void computeFactorials() {
    int threadID = threadIdx.x + 1;  // Each thread handles a unique integer
    int factorial = 1;

    // Calculate factorial
    for (int i = 2; i <= threadID; i++) {
        factorial *= i;
    }

    // Print the result
    printf("%d! = %d\n", threadID, factorial);
}

int main() {
    const int numThreads = 8;   // No of threads
    const int numBlocks = 1;    // No of GPU blocks

    // CUDA kernel compute and print factorials
    computeFactorials<<<numBlocks, numThreads>>>();

    // Synchronize all kernel prints
    cudaDeviceSynchronize();

    return 0;
}
