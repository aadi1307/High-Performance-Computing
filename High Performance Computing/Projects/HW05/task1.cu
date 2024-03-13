#include <iostream>
#include <chrono>
#include "matmul.cuh"
#include <cuda_runtime.h>

template<typename T>
void fillMatrix(T *mat, unsigned int n) {
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            mat[i*n + j] = (T)(i + j);
        }
    }
}

template<typename T>
void test_matmul(void (*matmul_func)(const T*, const T*, T*, unsigned int, unsigned int),
                 unsigned int n, unsigned int block_dim) {
    T *A, *B, *C;

    // Allocate and fill matrices
    cudaMallocManaged(&A, n*n*sizeof(T));
    cudaMallocManaged(&B, n*n*sizeof(T));
    cudaMallocManaged(&C, n*n*sizeof(T));
    
    fillMatrix(A, n);
    fillMatrix(B, n);

    // CUDA event timing mechanism
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent, 0);

    // Call the matrix multiplication function
    matmul_func(A, B, C, n, block_dim);

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);

    std::cout << C[0] << std::endl;
    std::cout << C[n*n-1] << std::endl;
    //std::cout << elapsedTime << " ms" << std::endl;
    std::cout << elapsedTime << std::endl;

    // Free memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    // Cleanup events
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size> <block_dim>" << std::endl;
        exit(1);
    }

    unsigned int n = atoi(argv[1]);
    unsigned int block_dim = atoi(argv[2]);

    test_matmul<int>(matmul_1, n, block_dim);
    test_matmul<float>(matmul_2, n, block_dim);
    test_matmul<double>(matmul_3, n, block_dim);

    return 0;
}
