#pragma once
#include <vector>
#include <set>
#include <algorithm>
#include "brute_force.h"

// Υπολογισμός Approximation Factor (AF)
// AF = distApprox / distTrue
inline double approximation_factor(double dist_approx, double dist_true) {
    if (dist_true == 0.0) return 1.0;
    return dist_approx / dist_true;
}

// Υπολογισμός Average AF για πολλά queries
inline double average_approximation_factor(
    const std::vector<double>& approx_distances,
    const std::vector<double>& true_distances)
{
    double sum_af = 0.0;
    int count = 0;
    
    for (size_t i = 0; i < approx_distances.size(); ++i) {
        if (true_distances[i] > 0.0) {
            sum_af += approximation_factor(approx_distances[i], true_distances[i]);
            count++;
        }
    }
    
    return count > 0 ? sum_af / count : 1.0;
}

// Υπολογισμός Recall@N
// Recall@N = (# σωστών γειτόνων στα Ν προσεγγιστικά) / N
inline double recall_at_n(
    const std::vector<NearestResult>& approx_neighbors,
    const std::vector<NearestResult>& true_neighbors)
{
    if (true_neighbors.empty()) return 1.0;
    
    // Δημιουργία set με τα indices των αληθινών γειτόνων
    std::set<int> true_indices;
    for (const auto& neighbor : true_neighbors) {
        true_indices.insert(neighbor.index);
    }
    
    // Μέτρηση πόσοι από τους προσεγγιστικούς είναι στους αληθινούς
    int matches = 0;
    for (const auto& neighbor : approx_neighbors) {
        if (true_indices.count(neighbor.index) > 0) {
            matches++;
        }
    }
    
    return (double)matches / (double)true_neighbors.size();
}

// Υπολογισμός Average Recall@N για πολλά queries
inline double average_recall(
    const std::vector<std::vector<NearestResult>>& all_approx,
    const std::vector<std::vector<NearestResult>>& all_true)
{
    double sum_recall = 0.0;
    int count = 0;
    
    for (size_t i = 0; i < all_approx.size(); ++i) {
        sum_recall += recall_at_n(all_approx[i], all_true[i]);
        count++;
    }
    
    return count > 0 ? sum_recall / count : 0.0;
}

// Υπολογισμός QPS (Queries Per Second)
inline double queries_per_second(int num_queries, double total_time_seconds) {
    if (total_time_seconds == 0.0) return 0.0;
    return (double)num_queries / total_time_seconds;
}
