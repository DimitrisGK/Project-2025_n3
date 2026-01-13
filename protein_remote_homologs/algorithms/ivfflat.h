#pragma once
#include <vector>
#include <string>

//signature με output file και range
void run_ivfflat(const std::vector<std::vector<float>>& data,
                 const std::vector<std::vector<float>>& queries,
                 int N, double R,
                 int kclusters, int nprobe, int seed,
                 const std::string& output_file, bool do_range);