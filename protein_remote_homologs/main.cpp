#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include "algorithms/lsh.h"
#include "algorithms/hypercube.h"
#include "algorithms/ivfflat.h"
#include "algorithms/ivfpq.h"

using namespace std;

// Διάβασμα binary dataset (SIFT format)
vector<vector<float>> load_sift_binary(const string& filename, int& n, int& dim) {
    ifstream file(filename, ios::binary);
    if (!file) {
        cerr << "Error: Cannot open file " << filename << endl;
        exit(1);
    }
    
    // Read header
    file.read(reinterpret_cast<char*>(&n), sizeof(int));
    file.read(reinterpret_cast<char*>(&dim), sizeof(int));
    
    cout << "Loading " << n << " vectors of dimension " << dim << endl;
    
    // Read data
    vector<vector<float>> data(n, vector<float>(dim));
    for (int i = 0; i < n; ++i) {
        file.read(reinterpret_cast<char*>(data[i].data()), dim * sizeof(float));
    }
    
    file.close();
    return data;
}

// Print usage
void print_usage(const char* prog) {
    cout << "Usage: " << prog << " -method <METHOD> [OPTIONS]\n\n";
    cout << "Methods:\n";
    cout << "  lsh       - Locality-Sensitive Hashing\n";
    cout << "  hypercube - Hypercube Projection\n";
    cout << "  ivfflat   - IVF with Flat vectors\n";
    cout << "  ivfpq     - IVF with Product Quantization\n\n";
    cout << "Common Options:\n";
    cout << "  -d <file>     Dataset file\n";
    cout << "  -q <file>     Query file\n";
    cout << "  -o <file>     Output file\n";
    cout << "  -type <type>  Dataset type (sift/mnist)\n";
    cout << "  -N <int>      Number of neighbors (default: 1)\n";
    cout << "  -R <float>    Range search radius (default: 1000)\n";
    cout << "  -range <bool> Enable range search (true/false)\n\n";
    cout << "LSH Options:\n";
    cout << "  -k <int>      Hash functions per table (default: 4)\n";
    cout << "  -L <int>      Number of hash tables (default: 5)\n";
    cout << "  -w <float>    Bucket width (default: 4.0)\n\n";
    cout << "Hypercube Options:\n";
    cout << "  -k <int>      Projection dimensions (default: 14)\n";
    cout << "  -M <int>      Max points to check (default: 10)\n";
    cout << "  -probes <int> Max probes (default: 2)\n\n";
    cout << "IVF Options:\n";
    cout << "  -kclusters <int> Number of clusters (default: 100)\n";
    cout << "  -nprobe <int>    Clusters to probe (default: 10)\n\n";
    cout << "IVF-PQ Options:\n";
    cout << "  -M <int>      Number of subquantizers (default: 8)\n";
    cout << "  -nbits <int>  Bits per subquantizer (default: 8)\n";
}

int main(int argc, char* argv[]) {
    // Default parameters
    string method = "";
    string dataset_file = "";
    string query_file = "";
    string output_file = "";
    string dataset_type = "sift";
    int N = 1;
    double R = 1000.0;
    bool do_range = false;
    int seed = 1;
    
    // LSH parameters
    int k_lsh = 4;
    int L = 5;
    double w = 4.0;
    
    // Hypercube parameters
    int kproj = 14;
    int M_hyper = 10;
    int probes = 2;
    
    // IVF parameters
    int kclusters = 100;
    int nprobe = 10;
    
    // IVF-PQ parameters
    int M_pq = 8;
    int nbits = 8;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        
        if (arg == "-method" && i + 1 < argc) {
            method = argv[++i];
        } else if (arg == "-d" && i + 1 < argc) {
            dataset_file = argv[++i];
        } else if (arg == "-q" && i + 1 < argc) {
            query_file = argv[++i];
        } else if (arg == "-o" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "-type" && i + 1 < argc) {
            dataset_type = argv[++i];
        } else if (arg == "-N" && i + 1 < argc) {
            N = atoi(argv[++i]);
        } else if (arg == "-R" && i + 1 < argc) {
            R = atof(argv[++i]);
        } else if (arg == "-range" && i + 1 < argc) {
            string val = argv[++i];
            do_range = (val == "true" || val == "1");
        } else if (arg == "-k" && i + 1 < argc) {
            k_lsh = atoi(argv[++i]);
            kproj = k_lsh;  // Also for hypercube
        } else if (arg == "-L" && i + 1 < argc) {
            L = atoi(argv[++i]);
        } else if (arg == "-w" && i + 1 < argc) {
            w = atof(argv[++i]);
        } else if (arg == "-M" && i + 1 < argc) {
            M_hyper = atoi(argv[++i]);
            M_pq = M_hyper;  // Also for PQ
        } else if (arg == "-probes" && i + 1 < argc) {
            probes = atoi(argv[++i]);
        } else if (arg == "-kclusters" && i + 1 < argc) {
            kclusters = atoi(argv[++i]);
        } else if (arg == "-nprobe" && i + 1 < argc) {
            nprobe = atoi(argv[++i]);
        } else if (arg == "-nbits" && i + 1 < argc) {
            nbits = atoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = atoi(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    // Validate required arguments
    if (method.empty() || dataset_file.empty() || query_file.empty() || output_file.empty()) {
        cerr << "Error: Missing required arguments\n\n";
        print_usage(argv[0]);
        return 1;
    }
    
    // Load data
    cout << "Loading dataset..." << endl;
    int n_data, dim_data;
    auto data = load_sift_binary(dataset_file, n_data, dim_data);
    
    cout << "Loading queries..." << endl;
    int n_query, dim_query;
    auto queries = load_sift_binary(query_file, n_query, dim_query);
    
    if (dim_data != dim_query) {
        cerr << "Error: Dimension mismatch between data and queries" << endl;
        return 1;
    }
    
    // Run selected method
    cout << "\nRunning " << method << " search..." << endl;
    cout << "Parameters: N=" << N << ", R=" << R << endl;
    
    if (method == "lsh") {
        cout << "LSH: k=" << k_lsh << ", L=" << L << ", w=" << w << endl;
        run_lsh(data, queries, N, R, k_lsh, L, w, seed, output_file, do_range);
        
    } else if (method == "hypercube") {
        cout << "Hypercube: kproj=" << kproj << ", M=" << M_hyper << ", probes=" << probes << endl;
        run_hypercube(data, queries, N, R, kproj, w, M_hyper, probes, seed, output_file, do_range);
        
    } else if (method == "ivfflat") {
        cout << "IVF-Flat: kclusters=" << kclusters << ", nprobe=" << nprobe << endl;
        run_ivfflat(data, queries, N, R, kclusters, nprobe, seed, output_file, do_range);
        
    } else if (method == "ivfpq") {
        cout << "IVF-PQ: kclusters=" << kclusters << ", nprobe=" << nprobe 
             << ", M=" << M_pq << ", nbits=" << nbits << endl;
        run_ivfpq(data, queries, N, R, kclusters, nprobe, M_pq, nbits, seed, output_file, do_range);
        
    } else {
        cerr << "Error: Unknown method '" << method << "'" << endl;
        print_usage(argv[0]);
        return 1;
    }
    
    cout << "\nSearch completed successfully!" << endl;
    cout << "Results written to: " << output_file << endl;
    
    return 0;
}