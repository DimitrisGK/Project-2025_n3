#include "ivfpq.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <random>
#include <algorithm>
#include "../utils/brute_force.h"
#include "../utils/metrics.h"
#include "../utils/output_writer.h"
#include "../utils/timer.h"
#include "../utils/clustering.h"

using namespace std;

// IVFPQ Αναζήτηση
void run_ivfpq(const vector<vector<float>>& data,
               const vector<vector<float>>& queries,
               int N, double R,
               int kclusters, int nprobe,
               int M, int nbits, int seed,
               const string& output_file, bool do_range)
{
    cout << "[IVFPQ] Εκκίνηση με kclusters=" << kclusters 
         << ", nprobe=" << nprobe 
         << ", M=" << M << ", nbits=" << nbits << endl;

    // Kmeans για clusters
    vector<int> assignments;
    auto centroids = kmeans(data, kclusters, seed, assignments);

    // Δημιουργία inverted index
    vector<vector<int>> inverted_index(kclusters);
    for (int i = 0; i < (int)data.size(); ++i)
        inverted_index[assignments[i]].push_back(i);

    int dim = data[0].size();
    int subdim = dim / M;      // Διαστάσεις subvector

    // Product Quantization
    // codebooks[cluster][subvector] = codebook για αυτό το subspace
    vector<vector<vector<vector<float>>>> codebooks(kclusters, vector<vector<vector<float>>>(M));
    // codes[cluster][point][subvector] = code για το subvecto
    vector<vector<vector<int>>> codes(kclusters);

    mt19937 gen(seed);

    // Για κάθε cluster
    for (int c = 0; c < kclusters; ++c) {
        int cluster_size = inverted_index[c].size();
        if (cluster_size == 0) continue;

        codes[c].resize(cluster_size, vector<int>(M));
        // Για κάθε subvector dimension
        for (int m = 0; m < M; ++m) {
            vector<vector<float>> subvecs;  //Συλλογή Subvectors
            for (int idx = 0; idx < cluster_size; ++idx) {
                int pid = inverted_index[c][idx];
                // Πάρε το m-οστό subvector (διαστάσεις [m*subdim : (m+1)*subdim])
                vector<float> sub(data[pid].begin() + m*subdim, data[pid].begin() + (m+1)*subdim);
                subvecs.push_back(sub);
            }

            // K-means στο subspace 
            int ksub = 1 << nbits;  // 2^nbits codes
            vector<int> dummy;
            codebooks[c][m] = kmeans(subvecs, ksub, seed, dummy);

            // Κωδικοποίηση
            for (int idx = 0; idx < cluster_size; ++idx) {
                int best = 0;
                float best_dist = numeric_limits<float>::max();
                // Βρες το πιο κοντινό code
                vector<float> sub(data[inverted_index[c][idx]].begin() + m*subdim, data[inverted_index[c][idx]].begin() + (m+1)*subdim);
                for (int kidx = 0; kidx < ksub; ++kidx) {
                    float dist = euclidean_distance(sub, codebooks[c][m][kidx]);
                    if (dist < best_dist) {
                        best_dist = dist;
                        best = kidx;
                    }
                }
                codes[c][idx][m] = best;    // Αποθήκευση code
            }
        }
    }

    cout << "[IVFPQ] Κωδικοποίηση ολοκληρώθηκε. Έναρξη αναζήτησης..." << endl;

    // Output writer, προετοιμασία εξόδου
    OutputWriter writer(output_file, "IVFPQ");
    writer.write_header();
    
    // Μετρικές μεταβλητές
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
        
        // Βρες nprobe κοντινότερα clusters
        vector<pair<int,float>> cluster_dists;
        for (int c = 0; c < kclusters; ++c)
            cluster_dists.push_back({c, euclidean_distance(q, centroids[c])});
        sort(cluster_dists.begin(), cluster_dists.end(), 
             [](auto& a, auto& b){ return a.second < b.second; });

        // Υποψήφιοι μέσω PQ
        vector<pair<int,float>> candidates;
        for (int i = 0; i < min(nprobe, (int)cluster_dists.size()); ++i) {
            int cid = cluster_dists[i].first;
            int cluster_size = inverted_index[cid].size();
            // Για κάθε σημείο στο cluster
            for (int idx = 0; idx < cluster_size; ++idx) {
                float dist = 0.0;
                // Υπολογισμός απόστασης απο codes
                for (int m = 0; m < M; ++m) {
                    int code_idx = codes[cid][idx][m];  // code του subvector
                    vector<float> q_sub(q.begin() + m*subdim, q.begin() + (m+1)*subdim);    // Subvector του query
                    dist += euclidean_distance(q_sub, codebooks[cid][m][code_idx]);     // Απόσταση από το codebook centroid
                }
                dist = sqrt(dist);
                if (dist <= R)
                    candidates.push_back({inverted_index[cid][idx], dist});
            }
        }

        sort(candidates.begin(), candidates.end(),
             [](auto& a, auto& b){ return a.second < b.second; });
        
        // Μετατροπή
        vector<NearestResult> approx_results;
        for (int i = 0; i < min(N, (int)candidates.size()); ++i) {
            approx_results.push_back({candidates[i].first, candidates[i].second});
        }
        
        total_approx_time += timer.toc();
        
        // Αληθινή αναζήτηση (brute force)
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
        
        // Αποτελέσματα
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
    
    cout << "[IVFPQ] Ολοκληρώθηκε. Αποτελέσματα στο: " << output_file << endl;
    cout << "      Average AF: " << avg_af << endl;
    cout << "      Recall@N: " << recall << endl;
    cout << "      QPS: " << qps << endl;
}