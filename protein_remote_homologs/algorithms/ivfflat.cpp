#include "ivfflat.h"
#include "../utils/brute_force.h"
#include "../utils/metrics.h"
#include "../utils/output_writer.h"
#include "../utils/timer.h"
#include "../utils/clustering.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <random>
#include <algorithm>

using namespace std;

// IVFFlat search για 1 query
vector<NearestResult> ivfflat_search_query(
    const vector<float>& query,
    const vector<vector<float>>& data,
    const vector<vector<float>>& centroids,     // Κέντρα clusters
    const vector<vector<int>>& inverted_index,  // cluster_id → [indices]
    int N, double R, int nprobe)
{
    // Βρήσκει nprobe κοντινότερα clusters
    vector<pair<float, int>> cluster_dists;
    for (int c = 0; c < (int)centroids.size(); ++c) {
        float dist = euclidean_distance(query, centroids[c]);
        cluster_dists.push_back({dist, c});     // (distance, cluster_id)
    }
    sort(cluster_dists.begin(), cluster_dists.end());   // Ταξινόμηση
    
    // Συλλογή υποψηφίων
    vector<pair<int, float>> candidates;
    // Ψάχνει μόνο στα nprobe πλησιέστερα clusters
    for (int i = 0; i < min(nprobe, (int)cluster_dists.size()); ++i) {
        int cid = cluster_dists[i].second;  // Cluster ID
        // Ψάχνει όλα τα σημεία στο cluster
        for (int idx : inverted_index[cid]) {
            float dist = euclidean_distance(query, data[idx]);
            if (dist <= R)
                candidates.push_back({idx, dist});
        }
    }
    
    // Ταξινόμηση
    sort(candidates.begin(), candidates.end(),
         [](auto& a, auto& b) { return a.second < b.second; });
    
    // Μετατροπή και επιστροφή των Ν
    vector<NearestResult> results;
    for (int i = 0; i < min(N, (int)candidates.size()); ++i) {
        results.push_back({candidates[i].first, candidates[i].second});
    }
    
    return results;
}

void run_ivfflat(const vector<vector<float>>& data,
                 const vector<vector<float>>& queries,
                 int N, double R,
                 int kclusters, int nprobe, int seed,
                 const string& output_file, bool do_range)
{
    cout << "[IVFFlat] Εκκίνηση με kclusters=" << kclusters 
         << ", nprobe=" << nprobe << endl;
    
    // K-means cluster
    vector<int> assignments;    // cluster_id για κάθε σημείο
    auto centroids = kmeans(data, kclusters, seed, assignments);
    
    // Δημιουργία inverted index
    vector<vector<int>> inverted_index(kclusters);
    for (int i = 0; i < (int)data.size(); ++i)      // Για κάθε σημείο βάλτο στο cluster του
        inverted_index[assignments[i]].push_back(i);
    
    cout << "[IVFFlat] Clustering ολοκληρώθηκε. Έναρξη αναζήτησης..." << endl;
    
    // Output writer
    OutputWriter writer(output_file, "IVFFlat");
    writer.write_header();
    
    // Μετρικές
    vector<double> approx_dists, true_dists;
    vector<vector<NearestResult>> all_approx, all_true;
    
    Timer timer;
    double total_approx_time = 0.0;
    double total_true_time = 0.0;
    
    // Επεξεργασία κάθε query
    for (int qid = 0; qid < (int)queries.size(); ++qid) {
        const auto& q = queries[qid];
        
        // Προσεγγιστική αναζήτηση 
        timer.tic();
        auto approx_results = ivfflat_search_query(
            q, data, centroids, inverted_index, N, R, nprobe);
        total_approx_time += timer.toc();
        
        // Αληθινή αναζήτηση
        timer.tic();
        auto true_results = brute_force_knn(q, data, N);
        total_true_time += timer.toc();
        
        // Range search
        vector<int> range_neighbors;
        if (do_range) {
            auto range_results = brute_force_range(q, data, R);
            for (const auto& res : range_results) {
                range_neighbors.push_back(res.index);
            }
        }
        
        // Αποθήκευση για μετρικές
        if (!approx_results.empty() && !true_results.empty()) {
            approx_dists.push_back(approx_results[0].distance);
            true_dists.push_back(true_results[0].distance);
        }
        all_approx.push_back(approx_results);
        all_true.push_back(true_results);
        
        // Aποτελέσματα
        QueryResults qr;
        qr.query_id = qid;
        qr.approx_neighbors = approx_results;
        qr.true_neighbors = true_results;
        qr.range_neighbors = range_neighbors;
        writer.write_query_results(qr);
    }
    
    // Υπολογισμός συνολικών μετρικών
    double avg_af = average_approximation_factor(approx_dists, true_dists);
    double recall = average_recall(all_approx, all_true);
    double qps = queries_per_second(queries.size(), total_approx_time);
    double t_approx_avg = total_approx_time / queries.size();
    double t_true_avg = total_true_time / queries.size();
    
    // Γράψε μετρικές στο file 
    writer.write_summary_metrics(avg_af, recall, qps, t_approx_avg, t_true_avg);
    
    cout << "[IVFFlat] Ολοκληρώθηκε. Αποτελέσματα στο: " << output_file << endl;
    cout << "      Average AF: " << avg_af << endl;
    cout << "      Recall@N: " << recall << endl;
    cout << "      QPS: " << qps << endl;
}