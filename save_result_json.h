#ifndef SAVE_RESULT_JSON_H
#define SAVE_RESULT_JSON_H

#include "wire_analyzer.h"
#include <fstream>
#include <cstring>

// 函数声明
void save_results_binary(const std::vector<SolderWire>& results, const std::string& path);
int load_results_binary(const std::string& path, std::vector<SolderWire> &results);
void covert2template(std::vector<SolderWire> &results, const int template_left, const int template_top);

#endif