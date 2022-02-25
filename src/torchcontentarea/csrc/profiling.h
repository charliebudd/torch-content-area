#pragma once
#include <vector>
#include <string>

// uncomment to allow profiling
// #define PROFILE

void ADD_SAMPLE(std::string name, float time);
std::vector<std::pair<std::string, float>> GET_TIMES();
