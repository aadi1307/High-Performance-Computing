// convolution.h

#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <vector>

// Function prototype for the convolution operation
std::vector<float> convolve(const std::vector<float>& image, int n, const std::vector<float>& mask, int m);

#endif // CONVOLUTION_H
