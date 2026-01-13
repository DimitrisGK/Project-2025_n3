#pragma once
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

// Ευκλείδεια απόσταση
inline float euclidean_dist(const std::vector<float>& a, const std::vector<float>& b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// Υπολογισμός Silhouette Score για clustering
// data: το dataset
// assignments: το cluster ID κάθε σημείου
// k: αριθμός clusters
inline double compute_silhouette_score(
    const std::vector<std::vector<float>>& data,
    const std::vector<int>& assignments,
    int k)
{
    int n = data.size();
    if (n == 0 || k <= 1) return 0.0;
    
    // Δημιουργία clusters
    std::vector<std::vector<int>> clusters(k);
    for (int i = 0; i < n; ++i) {
        if (assignments[i] >= 0 && assignments[i] < k) {
            clusters[assignments[i]].push_back(i);
        }
    }
    
    double total_silhouette = 0.0;
    int valid_points = 0;
    
    // Για κάθε σημείο
    for (int i = 0; i < n; ++i) {
        int cluster_i = assignments[i];
        if (cluster_i < 0 || cluster_i >= k) continue;
        if (clusters[cluster_i].size() <= 1) continue;  // Άγνοια μεμονωμένων σημείων
        
        // a(i): μέση απόσταση από σημεία του ίδιου cluster
        float a_i = 0.0f;
        int count_a = 0;
        for (int j : clusters[cluster_i]) {
            if (i != j) {
                a_i += euclidean_dist(data[i], data[j]);
                count_a++;
            }
        }
        if (count_a > 0) a_i /= count_a;
        
        // b(i): ελάχιστη μέση απόσταση από σημεία άλλου cluster
        float b_i = std::numeric_limits<float>::max();
        for (int c = 0; c < k; ++c) {
            if (c == cluster_i || clusters[c].empty()) continue;
            
            float avg_dist = 0.0f;
            for (int j : clusters[c]) {
                avg_dist += euclidean_dist(data[i], data[j]);
            }
            avg_dist /= clusters[c].size();
            b_i = std::min(b_i, avg_dist);
        }
        
        // s(i) = (b(i) - a(i)) / max(a(i), b(i))
        float s_i = 0.0f;
        if (b_i != std::numeric_limits<float>::max()) {
            float max_val = std::max(a_i, b_i);
            if (max_val > 0.0f) {
                s_i = (b_i - a_i) / max_val;
            }
        }
        
        total_silhouette += s_i;
        valid_points++;
    }
    
    return valid_points > 0 ? total_silhouette / valid_points : 0.0;
}

// Εύρεση βέλτιστου k με Silhouette Score
// Δοκιμάζει k από k_min έως k_max και επιστρέφει το καλύτερο
inline int find_optimal_k_silhouette(
    const std::vector<std::vector<float>>& data,
    int k_min, int k_max, int seed,
    std::vector<int>& best_assignments,
    std::vector<std::vector<float>>& best_centroids)
{
    // Πρέπει να δηλώσεις την kmeans εδώ ή να την κάνεις include
    // Για απλότητα, επιστρέφω μόνο το interface
    // Θα το συνδέσεις με την υπάρχουσα kmeans στα ivfflat/ivfpq
    
    double best_score = -1.0;
    int best_k = k_min;
    
    // Σημείωση: Αυτή η συνάρτηση χρειάζεται πρόσβαση στην kmeans
    // Θα την καλέσεις από τα ivfflat/ivfpq αρχεία
    
    return best_k;
}