#pragma once
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>

// Δομή για αποτέλεσμα nearest neighbor
struct NearestResult {
    int index;          // index του γείτονα στο dataset
    float distance;     // απόσταση από το query
};

// Brute Force: Βρες τον 1 πλησιέστερο γείτονα
inline NearestResult brute_force_nearest(
    const std::vector<float>& query,
    const std::vector<std::vector<float>>& data)
{
    float min_dist = std::numeric_limits<float>::max();
    int nearest_idx = -1;
    
    for (size_t i = 0; i < data.size(); ++i) {
        float dist = 0.0f;
        for (size_t d = 0; d < query.size(); ++d) {
            float diff = query[d] - data[i][d];
            dist += diff * diff;
        }
        dist = std::sqrt(dist);
        
        if (dist < min_dist) {
            min_dist = dist;
            nearest_idx = i;
        }
    }
    
    return {nearest_idx, min_dist};
}

// Brute Force: Βρες τους N πλησιέστερους γείτονες
inline std::vector<NearestResult> brute_force_knn(
    const std::vector<float>& query,
    const std::vector<std::vector<float>>& data,
    int N)
{
    std::vector<NearestResult> results;
    
    // Υπολόγισε όλες τις αποστάσεις
    for (size_t i = 0; i < data.size(); ++i) {
        float dist = 0.0f;
        for (size_t d = 0; d < query.size(); ++d) {
            float diff = query[d] - data[i][d];
            dist += diff * diff;
        }
        dist = std::sqrt(dist);
        results.push_back({(int)i, dist});
    }
    
    // Ταξινόμηση
    std::sort(results.begin(), results.end(),
        [](const NearestResult& a, const NearestResult& b) {
            return a.distance < b.distance;
        });
    
    // Επέστρεψε τους N πρώτους
    if (results.size() > (size_t)N) {
        results.resize(N);
    }
    
    return results;
}

// Brute Force: Range search (R-near neighbors)
inline std::vector<NearestResult> brute_force_range(
    const std::vector<float>& query,
    const std::vector<std::vector<float>>& data,
    double R)
{
    std::vector<NearestResult> results;
    
    for (size_t i = 0; i < data.size(); ++i) {
        float dist = 0.0f;
        for (size_t d = 0; d < query.size(); ++d) {
            float diff = query[d] - data[i][d];
            dist += diff * diff;
        }
        dist = std::sqrt(dist);
        
        if (dist <= R) {
            results.push_back({(int)i, dist});
        }
    }
    
    // Ταξινόμηση
    std::sort(results.begin(), results.end(),
        [](const NearestResult& a, const NearestResult& b) {
            return a.distance < b.distance;
        });
    
    return results;
}