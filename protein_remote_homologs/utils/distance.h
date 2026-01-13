#pragma once
#include <vector>
#include <cmath>

inline float euclidean_distance(const std::vector<float>& a, const std::vector<float>& b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); i++)
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    return sqrt(sum);
}
