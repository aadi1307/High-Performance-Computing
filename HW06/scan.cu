#include "scan.cuh"
#include <cuda_runtime.h>
#include <algorithm> // for std::min

// Kernel function for the Hillis-Steele inclusive scan algorithm
__global__ void hillis_steele(const float* g_idata, float* g_odata, int n, int p) {
    int global_tid = threadIdx.x + blockDim.x * blockIdx.x;

    // Load input into shared memory.
    // This is padded to avoid bank conflicts on shared memory reads.
    extern __shared__ float temp[];

    if (global_tid < n) {
        temp[threadIdx.x] = g_idata[global_tid];
    } else {
        temp[threadIdx.x] = 0; // pad values outside the array with 0
    }

    __syncthreads();

    int stride = 1 << (p - 1); // equivalent to pow(2, p-1)

    if (threadIdx.x >= stride) {
        temp[threadIdx.x] += temp[threadIdx.x - stride];
    }

    __syncthreads();

    if (global_tid < n) {
        g_odata[global_tid] = temp[threadIdx.x];
    }
}

__host__ void scan(const float* input, float* output, unsigned int n, unsigned int threads_per_block) {
    // Calculate the number of blocks to launch
    unsigned int num_blocks = (n + threads_per_block - 1) / threads_per_block;

    // Allocate temporary device memory to hold intermediate results
    float* d_temp;
    cudaMalloc(&d_temp, n * sizeof(float));
    cudaMemcpy(d_temp, input, n * sizeof(float), cudaMemcpyDeviceToDevice);

    // Shared memory size (it's used inside the kernel)
    unsigned int shared_mem_size = sizeof(float) * threads_per_block;

    // Loop over the data in steps, performing the inclusive scan operation
    // The Hillis-Steele algorithm requires log(n) steps.
    for (int p = 1; p <= int(ceilf(log2f(float(n)))); p++) {
        hillis_steele<<<num_blocks, threads_per_block, shared_mem_size>>>(d_temp, output, n, p);

        // Sync device before next iteration (needed before cudaMemcpy for next line)
        cudaDeviceSynchronize();

        // Swap input and output arrays: the previous output becomes the input for the next step
        std::swap(d_temp, output);
    }

    // Free temporary memory
    cudaFree(d_temp);
}
