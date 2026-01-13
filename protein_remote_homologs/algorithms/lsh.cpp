#include "lsh.h"
#include "../utils/brute_force.h"
#include "../utils/metrics.h"
#include "../utils/output_writer.h"
#include "../utils/timer.h"
#include <iostream>
#include <vector>
#include <random>
#include <unordered_map>
#include <cmath>
#include <limits>
#include <numeric>

using namespace std;

// Δομή για LSH πίνακα
struct LSHTable {
    vector<vector<float>> a_vectors;                // k τυχαία διανύσματα (projections)
    vector<float> b_values;                         // k τυχαίες μετατοπίσεις
    unordered_map<string, vector<int>> buckets;     // key → [indices σημείων]
};

// Hash key από g(q)
string hash_key(const vector<int>& g) {
    string key;
    for (int bit : g)                   // Για κάθε hash value μετατροπή σε string
        key += to_string(bit) + "_";    // Μετατρέπει το vector<int> σε string key για το hash table
    return key;
}

// Brute force για 1 query (helper)
vector<NearestResult> lsh_search_query(
    const vector<float>& query,
    const vector<vector<float>>& data,
    const vector<LSHTable>& tables,
    int N, double R, double w, int k, int L, int dim)
{
    unordered_map<int, bool> checked;
    vector<pair<int, float>> candidates;
    
    // Ψάξε σε όλα τα L tables
    for (int l = 0; l < L; ++l) {
        vector<int> g(k);          // k hash values

        //Υπολογισμός k hash functions
        for (int i = 0; i < k; ++i) {
            float dot = inner_product(query.begin(), query.end(),
                                      tables[l].a_vectors[i].begin(), 0.0f);
            g[i] = floor((dot + tables[l].b_values[i]) / w);
        }
        
        // Βρίσκει το bucket
        string key = hash_key(g);   // Μετατροπή σε string
        auto it = tables[l].buckets.find(key);

        // Αν υπάρχει bucket, πάρε τους υποψηφίους
        if (it != tables[l].buckets.end()) {
            for (int idx : it->second) {    // Για κάθε σημείο στο bucket
                if (!checked[idx]) {        // Αν δεν το έχουμε ξαναελέγξει
                    checked[idx] = true;
                    // Υπολογισμός L2 απόστασης
                    float dist = 0.0;
                    for (int d = 0; d < dim; ++d)
                        dist += pow(query[d] - data[idx][d], 2);
                    dist = sqrt(dist);
                    
                    if (dist <= R)      // Κραταει μόνο αν distance ≤ R
                        candidates.push_back({idx, dist});
                }
            }
        }
    }
    
    // Ταξινόμηση κατά απόσταση
    sort(candidates.begin(), candidates.end(),
         [](auto& a, auto& b) { return a.second < b.second; });
    
    // Μετατροπή σε NearestResult, κρατάει μόνο N πλησιέστερους
    vector<NearestResult> results;
    for (int i = 0; i < min(N, (int)candidates.size()); ++i) {
        results.push_back({candidates[i].first, candidates[i].second});
    }
    
    return results;
}

void run_lsh(const vector<vector<float>>& data,
             const vector<vector<float>>& queries,
             int N, double R,
             int k, int L, double w, int seed,
             const string& output_file, bool do_range)
{
    cout << "[LSH] Εκκίνηση με k=" << k << ", L=" << L << ", w=" << w << endl;
    
    mt19937 gen(seed);
    normal_distribution<float> gaussian(0.0, 1.0);
    uniform_real_distribution<float> uniform(0.0, w);
    
    int dim = data[0].size();   // Διαστάσεις
    vector<LSHTable> tables(L); // L Hash Tables
    
    // Δημιουργία L tables
    for (int l = 0; l < L; ++l) {
        LSHTable table;
        for (int i = 0; i < k; ++i) {
            vector<float> a(dim);
            for (int d = 0; d < dim; ++d)
                a[d] = gaussian(gen);               // Τυχαίο διάνυσμα gaussian
            table.a_vectors.push_back(a);           // Προσθήκη διανύσματος
            table.b_values.push_back(uniform(gen)); // Τυχαία μετατόπιση
        }
        tables[l] = move(table);
    }
    
    // Καταχώρηση δεδομένων
    for (int idx = 0; idx < (int)data.size(); ++idx) {
        for (int l = 0; l < L; ++l) {       // Για κάθε σημείο καταχωρεί σε όλα τα L tables
            vector<int> g(k);

            // Υπολογισμός k hash values
            for (int i = 0; i < k; ++i) {
                float dot = inner_product(data[idx].begin(), data[idx].end(),
                                          tables[l].a_vectors[i].begin(), 0.0f);
                g[i] = floor((dot + tables[l].b_values[i]) / w);
            }
            string key = hash_key(g);                   // Δημιουργία key
            tables[l].buckets[key].push_back(idx);      // Βάλε το σημείο στο bucket
        }
    }
    
    cout << "[LSH] Κατασκευή ολοκληρώθηκε. Έναρξη αναζήτησης..." << endl;
    
    // Output writer, προετοιμασία εξόδου
    OutputWriter writer(output_file, "LSH");
    writer.write_header();      // Γράψε header στο file
    
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
        timer.tic();    // Ξεκίνα χρονόμετρο
        auto approx_results = lsh_search_query(q, data, tables, N, R, w, k, L, dim);
        total_approx_time += timer.toc();   // Σταματάει και προσθέτει χρόνο
        
        // Αληθινή αναζήτηση (brute force)
        timer.tic();
        auto true_results = brute_force_knn(q, data, N);  // Για κάθε query κάνει και brute force
        total_true_time += timer.toc();     
        
        // Range search (optional)
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
    
    cout << "[LSH] Ολοκληρώθηκε. Αποτελέσματα στο: " << output_file << endl;
    cout << "      Average AF: " << avg_af << endl;
    cout << "      Recall@N: " << recall << endl;
    cout << "      QPS: " << qps << endl;
}
