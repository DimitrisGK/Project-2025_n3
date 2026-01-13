#pragma once
#include <vector>
#include <string>

//signature με output file και range
void run_hypercube(const std::vector<std::vector<float>>& data,
                   const std::vector<std::vector<float>>& queries,
                   int N, double R,
                   int kproj, double w, int M, int probes, int seed,
                   const std::string& output_file, bool do_range);
