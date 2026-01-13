#pragma once
#include <vector>
#include <cmath>
#include <limits>
#include <random>
#include <algorithm>

// Ευκλείδεια απόσταση (inline για να μην έχουμε duplicates)
inline float euclidean_distance(const std::vector<float>& a, const std::vector<float>& b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// Εύρεση πλησιέστερου centroid
inline int nearest_centroid(const std::vector<float>& point, 
                            const std::vector<std::vector<float>>& centroids) {
    int best = 0;
    float best_dist = std::numeric_limits<float>::max();
    for (int i = 0; i < (int)centroids.size(); ++i) {
        float dist = euclidean_distance(point, centroids[i]);
        if (dist < best_dist) {
            best_dist = dist;
            best = i;
        }
    }
    return best;
}

// K-means clustering
inline std::vector<std::vector<float>> kmeans(
    const std::vector<std::vector<float>>& data, 
    int k, 
    int seed, 
    std::vector<int>& assignments) 
{
    int n = data.size();
    int dim = data[0].size();
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(0, n - 1);
    
    // Αρχικοποίηση τυχαίων κέντρων
    std::vector<std::vector<float>> centroids(k);
    for (int i = 0; i < k; ++i)
        centroids[i] = data[dist(gen)];
    
    bool changed = true;
    int max_iters = 20;
    assignments.resize(n);
    
    for (int iter = 0; iter < max_iters && changed; ++iter) {
        changed = false;
        
        // Ανάθεση σημείων στο κοντινότερο κέντρο
        for (int i = 0; i < n; ++i) {
            int cluster_id = nearest_centroid(data[i], centroids);
            if (cluster_id != assignments[i]) {
                assignments[i] = cluster_id;
                changed = true;
            }
        }
        
        // Ενημέρωση κέντρων
        std::vector<std::vector<float>> new_centroids(k, std::vector<float>(dim, 0.0f));
        std::vector<int> counts(k, 0);
        
        for (int i = 0; i < n; ++i) {
            int c = assignments[i];
            for (int d = 0; d < dim; ++d)
                new_centroids[c][d] += data[i][d];
            counts[c]++;
        }
        
        for (int c = 0; c < k; ++c)
            if (counts[c] > 0)
                for (int d = 0; d < dim; ++d)
                    new_centroids[c][d] /= counts[c];
        
        centroids = new_centroids;
    }
    
    return centroids;
}