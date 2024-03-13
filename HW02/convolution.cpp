// convolution.cpp

#include "convolution.h"

// Implementation of the convolution function
std::vector<float> convolve(const std::vector<float>& image, int n, const std::vector<float>& mask, int m) {
    std::vector<float> result(n * n, 0.0f);  // Initialize result vector with zeros
    int offset = (m - 1) / 2;  // Calculate the offset from the mask dimension

    // Iterate through the image pixels
    for (int x = 0; x < n; ++x) {
        for (int y = 0; y < n; ++y) {
            
            // Apply the mask for each pixel
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < m; ++j) {
                    int xi = x + i - offset;
                    int yj = y + j - offset;
                    float value;

                    // Boundary handling: check if current coordinates are outside the image boundaries
                    if (xi < 0 || xi >= n || yj < 0 || yj >= n) {
                        if (xi == -1 || xi == n || yj == -1 || yj == n)
                            value = 1.0f;
                        else
                            value = 0.0f;
                    } else {
                        value = image[xi * n + yj];
                    }

                    // Update the result value using the mask value
                    result[x * n + y] += mask[i * m + j] * value;
                }
            }
        }
    }

    return result;
}
