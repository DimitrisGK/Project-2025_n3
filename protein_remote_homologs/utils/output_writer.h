#pragma once
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>
#include "brute_force.h"

// Δομή για αποτελέσματα ενός query
struct QueryResults {
    int query_id;
    std::vector<NearestResult> approx_neighbors;
    std::vector<NearestResult> true_neighbors;
    std::vector<int> range_neighbors;  // R-near neighbors (indices)
};

class OutputWriter {
private:
    std::ofstream file;
    std::string method_name;
    
public:
    OutputWriter(const std::string& filename, const std::string& method) 
        : method_name(method) {
        file.open(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open output file: " + filename);
        }
        file << std::fixed << std::setprecision(6);
    }
    
    ~OutputWriter() {
        if (file.is_open()) {
            file.close();
        }
    }
    
    // Γράψε τα αποτελέσματα ενός query
    void write_query_results(const QueryResults& results) {
        file << method_name << "\n";
        file << "Query: " << results.query_id << "\n";
        
        // Nearest neighbors
        int N = results.approx_neighbors.size();
        for (int i = 0; i < N; ++i) {
            file << "Nearest neighbor-" << (i + 1) << ": " 
                 << results.approx_neighbors[i].index << "\n";
            file << "distanceApproximate: " 
                 << results.approx_neighbors[i].distance << "\n";
            
            if (i < (int)results.true_neighbors.size()) {
                file << "distanceTrue: " 
                     << results.true_neighbors[i].distance << "\n";
            } else {
                file << "distanceTrue: N/A\n";
            }
        }
        
        // R-near neighbors
        if (!results.range_neighbors.empty()) {
            file << "R-near neighbors:";
            for (int idx : results.range_neighbors) {
                file << " " << idx;
            }
            file << "\n";
        }
        
        file << "\n";  // Κενή γραμμή μεταξύ queries
    }
    
    // Γράψε τις συνολικές μετρικές
    void write_summary_metrics(
        double avg_af,
        double recall_at_n,
        double qps,
        double t_approx_avg,
        double t_true_avg)
    {
        file << "=== SUMMARY METRICS ===\n";
        file << "Average AF: " << avg_af << "\n";
        file << "Recall@N: " << recall_at_n << "\n";
        file << "QPS: " << qps << "\n";
        file << "tApproximateAverage: " << t_approx_avg << " seconds\n";
        file << "tTrueAverage: " << t_true_avg << " seconds\n";
    }
    
    // Γράψε header (method name)
    void write_header() {
        file << "============================================\n";
        file << "METHOD: " << method_name << "\n";
        file << "============================================\n\n";
    }
};
