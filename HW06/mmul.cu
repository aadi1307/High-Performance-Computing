// mmul.cu
#include "mmul.h"

void mmul(cublasHandle_t handle, const float* A, const float* B, float* C, int n) {
    const float alpha = 1.0f;
    const float beta = 1.0f;

    // Perform matrix-matrix multiplication: C = alpha*A*B + beta*C
    // Note: cuBLAS assumes matrices are in column-major order
    // Leading dimensions are typically the number of rows in the respective matrices
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                A, n, B, n, &beta, C, n);

    // Synchronize the device to make sure that all computations are finished before stopping the timer
    cudaDeviceSynchronize();
}
