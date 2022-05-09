#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <chrono>

// uncomment to allow profiling
#define PROFILE

struct CudaTimer
{
public:
    CudaTimer(std::string name);
    void stop();
private:
    std::string m_name;
    cudaEvent_t m_cuda_start, m_cuda_stop;
    std::chrono::_V2::system_clock::time_point m_overall_start;
};

void ADD_SAMPLE(std::string name, float cuda_time, float cpu_time, float overall_time);
std::vector<std::pair<std::string, std::vector<float>>> GET_TIMES();

