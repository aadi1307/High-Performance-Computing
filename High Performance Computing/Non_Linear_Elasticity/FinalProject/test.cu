#include <iostream>

class MyClass {
public:
    __device__ void memberKernel(int *data) {

        int idx = threadIdx.x;
        data[idx] = idx;  // An example operation
    }
};

__global__ static void globalWrapper(int *data) {
        printf("globalWrapper:\n");

        MyClass obj;
        obj.memberKernel(data);
    }
 
// Usage
int main() {
    const int arraySize = 256;
    int *d_data;
 
    // Allocate memory on the GPU
    cudaMalloc((void**)&d_data, arraySize * sizeof(int));
 
    // Launch the kernel
    globalWrapper<<<1, arraySize>>>(d_data);
 
    // Free memory
    cudaFree(d_data);
 
    return 0;
}
