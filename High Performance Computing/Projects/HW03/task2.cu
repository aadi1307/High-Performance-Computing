#include <iostream>
#include <cstdlib>  // For std::rand and std::srand
#include <ctime>    // For std::time

// Kernel to perform the ax + y operation
__global__ void compute(int a, int* dA) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int x = threadIdx.x;
    int y = blockIdx.x;
    dA[idx] = a * x + y;
}

int main() {
    const int arraySize = 16;
    const int blockSize = 8;

    // Allocate host memory
    int hA[arraySize];

    // Allocate device memory
    int* dA;
    cudaMalloc((void**)&dA, arraySize * sizeof(int));

    // Seed the random number generator and generate a random value for a
    std::srand(std::time(0));
    int a = std::rand() % 100;

    // Launch the compute kernel
    compute<<<2, blockSize>>>(a, dA);

    // Copy the result back to host
    cudaMemcpy(hA, dA, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result
    for(int i = 0; i < arraySize; i++) {
        std::cout << hA[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(dA);

    return 0;
}
