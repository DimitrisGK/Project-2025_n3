#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "../algorithms/ivfpq.h"

using namespace std;

vector<vector<float>> load_binary(const string& filename, int& n, int& dim) {
    ifstream file(filename, ios::binary);
    if (!file) { cerr << "Error: Cannot open " << filename << endl; exit(1); }
    file.read(reinterpret_cast<char*>(&n), sizeof(int));
    file.read(reinterpret_cast<char*>(&dim), sizeof(int));
    vector<vector<float>> data(n, vector<float>(dim));
    for (int i = 0; i < n; ++i)
        file.read(reinterpret_cast<char*>(data[i].data()), dim * sizeof(float));
    return data;
}

int main(int argc, char* argv[]) {
    string data_file, query_file, output_file;
    int N = 1, kclusters = 100, nprobe = 10, M = 8, nbits = 8, seed = 1;
    double R = 1000.0;
    bool do_range = false;
    
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-d" && i+1 < argc) data_file = argv[++i];
        else if (arg == "-q" && i+1 < argc) query_file = argv[++i];
        else if (arg == "-o" && i+1 < argc) output_file = argv[++i];
        else if (arg == "-N" && i+1 < argc) N = atoi(argv[++i]);
        else if (arg == "-R" && i+1 < argc) R = atof(argv[++i]);
        else if (arg == "-kclusters" && i+1 < argc) kclusters = atoi(argv[++i]);
        else if (arg == "-nprobe" && i+1 < argc) nprobe = atoi(argv[++i]);
        else if (arg == "-M" && i+1 < argc) M = atoi(argv[++i]);
        else if (arg == "-nbits" && i+1 < argc) nbits = atoi(argv[++i]);
        else if (arg == "-range" && i+1 < argc) do_range = (string(argv[++i]) == "true");
        else if (arg == "-type") ++i;
    }
    
    if (data_file.empty() || query_file.empty() || output_file.empty()) {
        cerr << "Usage: " << argv[0] << " -d <data> -q <query> -o <output> [options]" << endl;
        return 1;
    }
    
    int n_d, d_d, n_q, d_q;
    auto data = load_binary(data_file, n_d, d_d);
    auto queries = load_binary(query_file, n_q, d_q);
    
    cout << "[IVF-PQ] " << n_d << " data points, " << n_q << " queries, dim=" << d_d << endl;
    run_ivfpq(data, queries, N, R, kclusters, nprobe, M, nbits, seed, output_file, do_range);
    
    return 0;
}
