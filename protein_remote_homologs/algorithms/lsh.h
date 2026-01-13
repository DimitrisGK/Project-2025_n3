// #pragma once
// #include <vector>

// //βασική συνάρτηση για τον LSH αλγόριθμο
// void run_lsh(const std::vector<std::vector<float>>& data,
//              const std::vector<std::vector<float>>& queries,
//              int N, double R, 
//              int k = 4, int L = 5, double w = 4.0, int seed = 1);


#pragma once
#include <vector>
#include <string>

//signature με output file
void run_lsh(const std::vector<std::vector<float>>& data,
             const std::vector<std::vector<float>>& queries,
             int N, double R, 
             int k, int L, double w, int seed,
             const std::string& output_file, bool do_range);