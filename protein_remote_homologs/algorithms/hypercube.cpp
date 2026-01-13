#include "hypercube.h"
#include "../utils/brute_force.h"
#include "../utils/metrics.h"
#include "../utils/output_writer.h"
#include "../utils/timer.h"
#include <iostream>
#include <vector>
#include <random>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <numeric>

using namespace std;

// Binary hash
string binary_hash(const vector<int>& bits) {
    string s;
    for (int b : bits)      // Για κάθε bit (0 ή 1) μετατροπή σε string
        s += (b > 0 ? '1' : '0');
    return s;
}

// Hamming distance
int hamming_distance(const string& a, const string& b) {
    int dist = 0;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) dist++;   // Μέτρα πόσα bits διαφέρουν
    return dist;
}

// Hypercube search για 1 query
vector<NearestResult> hypercube_search_query(
    const vector<float>& query,
    const vector<vector<float>>& data,
    const unordered_map<string, vector<int>>& cube, // υπερκύβος
    const vector<vector<float>>& projections,       // kproj τυχαία διανύσματα
    const vector<float>& offsets,
    int N, double R, int kproj, int M, int probes, int dim)
{
    // Προβολή του query
    vector<int> bits(kproj);
    for (int i = 0; i < kproj; ++i) {
        float dot = inner_product(query.begin(), query.end(), projections[i].begin(), 0.0f); // Προβολή σε τυχαίο διάνυσμα
        bits[i] = (dot + offsets[i] >= 0 ? 1 : 0);  // Bit = 1 αν >= 0, αλλιώς 0
    }
    string query_key = binary_hash(bits);
    
    vector<pair<int, float>> candidates;
    int checked_points = 0;
    
    // Έλεγχος μόνο κορυφών με Hamming distance <= probes
    for (const auto& [key, indices] : cube) {
        if (hamming_distance(key, query_key) <= probes) {
            // Έλεγχος σημείων κορυφών
            for (int idx : indices) {
                if (checked_points++ >= M) break;   // Όριο M σημείων
                
                // Υπολογισμός L2 απόστασης
                float dist = 0.0;
                for (int d = 0; d < dim; ++d)
                    dist += pow(query[d] - data[idx][d], 2);
                dist = sqrt(dist);
                
                // Κρατάει αν distance ≤ R
                if (dist <= R)
                    candidates.push_back({idx, dist});
            }
        }
        if (checked_points >= M) break; // Στματάει αν φτάσαμε το όριο
    }
    
    // Ταξινόμηση
    sort(candidates.begin(), candidates.end(),
         [](auto& a, auto& b) { return a.second < b.second; });
    
    // Μετατροπή και επιστροφή Ν
    vector<NearestResult> results;
    for (int i = 0; i < min(N, (int)candidates.size()); ++i) {
        results.push_back({candidates[i].first, candidates[i].second});
    }
    
    return results;
}

void run_hypercube(const vector<vector<float>>& data,
                   const vector<vector<float>>& queries,
                   int N, double R,
                   int kproj, double w, int M, int probes, int seed,
                   const string& output_file, bool do_range)
{
    cout << "[Hypercube] Εκκίνηση με kproj=" << kproj 
         << ", w=" << w << ", M=" << M << ", probes=" << probes << endl;
    
    mt19937 gen(seed);
    normal_distribution<float> gaussian(0.0, 1.0);
    uniform_real_distribution<float> uniform(0.0, w);
    
    int dim = data[0].size();
    
    // Δημιουργία kproj τυχαίων προβολών
    vector<vector<float>> projections(kproj, vector<float>(dim));
    for (int i = 0; i < kproj; ++i)
        for (int d = 0; d < dim; ++d)
            projections[i][d] = gaussian(gen);  // N(0,1)
    
    vector<float> offsets(kproj);
    for (int i = 0; i < kproj; ++i)
        offsets[i] = uniform(gen);  // U(0,w)
    
    // Δημιουργία κύβου
    unordered_map<string, vector<int>> cube;        // key → [indices]
    for (int idx = 0; idx < (int)data.size(); ++idx) {
        vector<int> bits(kproj);
        // Προβολή κάθε σημείου
        for (int i = 0; i < kproj; ++i) {
            float dot = inner_product(data[idx].begin(), data[idx].end(), projections[i].begin(), 0.0f);
            bits[i] = (dot + offsets[i] >= 0 ? 1 : 0);
        }
        string key = binary_hash(bits);
        cube[key].push_back(idx);       // Βάζει το σημείο στην κορυφή
    }
    
    cout << "[Hypercube] Κατασκευή ολοκληρώθηκε. Έναρξη αναζήτησης..." << endl;
    
    // Output writer
    OutputWriter writer(output_file, "Hypercube");
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
        auto approx_results = hypercube_search_query(q, data, cube, projections, offsets, N, R, kproj, M, probes, dim);
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
    
    cout << "[Hypercube] Ολοκληρώθηκε. Αποτελέσματα στο: " << output_file << endl;
    cout << "      Average AF: " << avg_af << endl;
    cout << "      Recall@N: " << recall << endl;
    cout << "      QPS: " << qps << endl;
}