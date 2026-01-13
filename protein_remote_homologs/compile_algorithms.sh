#!/bin/bash
# Compilation script για C++ αλγορίθμους (με header-only utilities)

set -e  # Exit on error

echo "================================"
echo "Compiling ANN Algorithms"
echo "================================"

# Δημιουργία directories
mkdir -p build
mkdir -p mains

# Compiler settings
CXX=g++
CXXFLAGS="-std=c++17 -O3 -Wall -Wextra"
INCLUDES="-I. -I./utils -I./algorithms"

# Check if utils headers exist
echo ""
echo "Checking utils headers..."
if [ ! -f "utils/brute_force.h" ]; then
    echo "⚠️  Warning: utils/brute_force.h not found"
fi
if [ ! -f "utils/metrics.h" ]; then
    echo "⚠️  Warning: utils/metrics.h not found"
fi
if [ ! -f "utils/output_writer.h" ]; then
    echo "⚠️  Warning: utils/output_writer.h not found"
fi
if [ ! -f "utils/timer.h" ]; then
    echo "⚠️  Warning: utils/timer.h not found"
fi
if [ ! -f "utils/clustering.h" ]; then
    echo "⚠️  Warning: utils/clustering.h not found (needed for IVF)"
fi

echo ""
echo "Creating main files..."

# ============================================================================
# Δημιουργία ξεχωριστών main files
# ============================================================================

# LSH Main
cat > mains/lsh_main.cpp << 'EOF'
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "../algorithms/lsh.h"

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
    int N = 1, k = 4, L = 5, seed = 1;
    double R = 1000.0, w = 4.0;
    bool do_range = false;
    
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-d" && i+1 < argc) data_file = argv[++i];
        else if (arg == "-q" && i+1 < argc) query_file = argv[++i];
        else if (arg == "-o" && i+1 < argc) output_file = argv[++i];
        else if (arg == "-N" && i+1 < argc) N = atoi(argv[++i]);
        else if (arg == "-R" && i+1 < argc) R = atof(argv[++i]);
        else if (arg == "-k" && i+1 < argc) k = atoi(argv[++i]);
        else if (arg == "-L" && i+1 < argc) L = atoi(argv[++i]);
        else if (arg == "-w" && i+1 < argc) w = atof(argv[++i]);
        else if (arg == "-range" && i+1 < argc) do_range = (string(argv[++i]) == "true");
        else if (arg == "-type") ++i;  // Ignore type parameter
    }
    
    if (data_file.empty() || query_file.empty() || output_file.empty()) {
        cerr << "Usage: " << argv[0] << " -d <data> -q <query> -o <output> [options]" << endl;
        return 1;
    }
    
    int n_d, d_d, n_q, d_q;
    auto data = load_binary(data_file, n_d, d_d);
    auto queries = load_binary(query_file, n_q, d_q);
    
    cout << "[LSH] " << n_d << " data points, " << n_q << " queries, dim=" << d_d << endl;
    run_lsh(data, queries, N, R, k, L, w, seed, output_file, do_range);
    
    return 0;
}
EOF

# Hypercube Main
cat > mains/hypercube_main.cpp << 'EOF'
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "../algorithms/hypercube.h"

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
    int N = 1, kproj = 14, M = 10, probes = 2, seed = 1;
    double R = 1000.0, w = 4.0;
    bool do_range = false;
    
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-d" && i+1 < argc) data_file = argv[++i];
        else if (arg == "-q" && i+1 < argc) query_file = argv[++i];
        else if (arg == "-o" && i+1 < argc) output_file = argv[++i];
        else if (arg == "-N" && i+1 < argc) N = atoi(argv[++i]);
        else if (arg == "-R" && i+1 < argc) R = atof(argv[++i]);
        else if (arg == "-k" && i+1 < argc) kproj = atoi(argv[++i]);
        else if (arg == "-w" && i+1 < argc) w = atof(argv[++i]);
        else if (arg == "-M" && i+1 < argc) M = atoi(argv[++i]);
        else if (arg == "-probes" && i+1 < argc) probes = atoi(argv[++i]);
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
    
    cout << "[Hypercube] " << n_d << " data points, " << n_q << " queries, dim=" << d_d << endl;
    run_hypercube(data, queries, N, R, kproj, w, M, probes, seed, output_file, do_range);
    
    return 0;
}
EOF

# IVF-Flat Main
cat > mains/ivfflat_main.cpp << 'EOF'
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "../algorithms/ivfflat.h"

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
    int N = 1, kclusters = 100, nprobe = 10, seed = 1;
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
    
    cout << "[IVF-Flat] " << n_d << " data points, " << n_q << " queries, dim=" << d_d << endl;
    run_ivfflat(data, queries, N, R, kclusters, nprobe, seed, output_file, do_range);
    
    return 0;
}
EOF

# IVF-PQ Main
cat > mains/ivfpq_main.cpp << 'EOF'
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
EOF

echo "✓ Main files created"

# ============================================================================
# Compilation
# ============================================================================

echo ""
echo "Compiling algorithms..."

# Check if algorithm files exist
if [ ! -f "algorithms/lsh.cpp" ]; then
    echo "⚠️  Warning: algorithms/lsh.cpp not found"
fi

echo ""
echo "[1/4] Compiling LSH..."
if $CXX $CXXFLAGS $INCLUDES \
    mains/lsh_main.cpp \
    algorithms/lsh.cpp \
    -o build/lsh_search 2>&1 | tee build/lsh_errors.log; then
    echo "✓ LSH compiled successfully"
else
    echo "✗ LSH compilation failed. Check build/lsh_errors.log"
    cat build/lsh_errors.log
    exit 1
fi

echo "[2/4] Compiling Hypercube..."
if $CXX $CXXFLAGS $INCLUDES \
    mains/hypercube_main.cpp \
    algorithms/hypercube.cpp \
    -o build/hypercube_search 2>&1 | tee build/hypercube_errors.log; then
    echo "✓ Hypercube compiled successfully"
else
    echo "✗ Hypercube compilation failed. Check build/hypercube_errors.log"
    cat build/hypercube_errors.log
    exit 1
fi

echo "[3/4] Compiling IVF-Flat..."
if $CXX $CXXFLAGS $INCLUDES \
    mains/ivfflat_main.cpp \
    algorithms/ivfflat.cpp \
    -o build/ivfflat_search 2>&1 | tee build/ivfflat_errors.log; then
    echo "✓ IVF-Flat compiled successfully"
else
    echo "✗ IVF-Flat compilation failed. Check build/ivfflat_errors.log"
    cat build/ivfflat_errors.log
    exit 1
fi

echo "[4/4] Compiling IVF-PQ..."
if $CXX $CXXFLAGS $INCLUDES \
    mains/ivfpq_main.cpp \
    algorithms/ivfpq.cpp \
    -o build/ivfpq_search 2>&1 | tee build/ivfpq_errors.log; then
    echo "✓ IVF-PQ compiled successfully"
else
    echo "✗ IVF-PQ compilation failed. Check build/ivfpq_errors.log"
    cat build/ivfpq_errors.log
    exit 1
fi

echo ""
echo "[5/5] Making executables..."
chmod +x build/*

echo ""
echo "================================"
echo "✓ Compilation completed!"
echo "Executables in: ./build/"
echo "================================"
echo ""
echo "Available binaries:"
ls -lh build/ | grep search
echo ""
echo "Test with:"
echo "  ./build/lsh_search -h"