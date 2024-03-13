// convolution.cpp
#include "convolution.h"
#include <vector>
#include <cstddef>

void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m) {
    int offset = (m - 1) / 2;

    #pragma omp parallel for collapse(2)
    for (std::size_t x = 0; x < n; ++x) {
        for (std::size_t y = 0; y < n; ++y) {
            float result = 0.0f;
            for (std::size_t i = 0; i < m; ++i) {
                for (std::size_t j = 0; j < m; ++j) {
                    int xi = static_cast<int>(x) + i - offset;
                    int yj = static_cast<int>(y) + j - offset;

                    float value = 0.0f;
                    // Correct the boundary conditions by clamping the indices to the valid range
                    if (xi >= 0 && xi < static_cast<int>(n) && yj >= 0 && yj < static_cast<int>(n)) {
                        value = image[xi * n + yj];
                    }

                    result += mask[i * m + j] * value;
                }
            }
            output[x * n + y] = result;
        }
    }
}
