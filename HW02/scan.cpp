#include "scan.h"

std::vector<float> scan(const std::vector<float>& input) {
    std::vector<float> output(input.size());

    if (!input.empty()) {
        output[0] = input[0];
        for (size_t i = 1; i < input.size(); ++i) {
            output[i] = output[i - 1] + input[i];
        }
    }

    return output;
}
