#include "stencil.cuh"
#include <iostream>
#include <random>

__host__ void populateWithRandomValues(float* data, int len) {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_real_distribution<float> uniformDist(-1.0f, 1.0f);

    for (int idx = 0; idx < len; idx++) {
        data[idx] = uniformDist(gen);
    }
}

int main(int argCount, char** args) {
    if (argCount != 4) {
        std::cerr << "Command format: " << args[0] << " <n> <R> <threads_per_block>" << std::endl;
        exit(EXIT_FAILURE);
    }

    unsigned int n = atoi(args[1]); // Original data size
    int rad = atoi(args[2]);
    unsigned int threadsPerBlock = atoi(args[3]);

    // Extend the image data by 2*R
    unsigned int totalData = n + 2 * rad;

    float *hostImage, *hostOutput, *hostMask;
    float *devImage, *devOutput, *devMask;

    hostImage = new float[totalData];
    hostOutput = new float[n];  // Output remains of size n
    hostMask = new float[2 * rad + 1];

    populateWithRandomValues(hostImage + rad, n);  // Starting from + rad to leave space for padding
    populateWithRandomValues(hostMask, 2 * rad + 1);

    // Initializing the boundaries to some value. Adjust as necessary.
    for(int i = 0; i < rad; i++) {
        hostImage[i] = 1.0f; // Beginning padding
        hostImage[n + rad + i] = 1.0f; // Ending padding
    }

    cudaMalloc(&devImage, totalData * sizeof(float));
    cudaMalloc(&devOutput, n * sizeof(float));  // Output remains of size n
    cudaMalloc(&devMask, (2 * rad + 1) * sizeof(float));

    cudaMemcpy(devImage, hostImage, totalData * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devMask, hostMask, (2 * rad + 1) * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
    cudaEventRecord(begin);

    stencil(devImage, devMask, devOutput, n, rad, threadsPerBlock);  // Pass n, not totalData

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float duration = 0;
    cudaEventElapsedTime(&duration, begin, end);

    cudaMemcpy(hostOutput, devOutput, n * sizeof(float), cudaMemcpyDeviceToHost);  // Fetch only n elements

    std::cout << hostOutput[n - 1] << std::endl;
    std::cout << duration << std::endl;

    delete[] hostImage;
    delete[] hostOutput;
    delete[] hostMask;
    cudaFree(devImage);
    cudaFree(devOutput);
    cudaFree(devMask);

    return 0;
}
