#include "reduce.cuh"
#include <cuda_runtime.h>

// Kernel function: First add during global load optimization
__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Load two data elements per thread
    float sum = (i < n) ? g_idata[i] : 0;
    if (i + blockDim.x < n) sum += g_idata[i + blockDim.x];

    sdata[tid] = sum;
    __syncthreads();

    // Do reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result to output data
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__host__ void reduce(float **input, float **output, unsigned int N, unsigned int threads_per_block) {
    float *in = *input;
    float *out = *output;

    unsigned int numBlocks;
    while (N > 1) {
        numBlocks = (N + threads_per_block * 2 - 1) / (threads_per_block * 2);
        reduce_kernel<<<numBlocks, threads_per_block, threads_per_block * sizeof(float)>>>(in, out, N);

        // Now, 'output' becomes 'input' for the next round
        N = numBlocks;
        in = out;
        if (numBlocks > 1) cudaMalloc((void**)&out, numBlocks * sizeof(float));
    }

    // Copy the final result to the start of the initial input array
    cudaMemcpy(*input, in, sizeof(float), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
}
