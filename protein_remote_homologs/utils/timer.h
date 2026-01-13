#pragma once
#include <chrono>

class Timer {
    std::chrono::high_resolution_clock::time_point start;
public:
    void tic() { start = std::chrono::high_resolution_clock::now(); }
    double toc() {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        return diff.count();
    }
};
