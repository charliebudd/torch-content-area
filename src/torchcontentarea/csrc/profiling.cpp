#include "profiling.h"
#include <map>

CudaTimer::CudaTimer(std::string name) : m_name(name)
{
    cudaEventCreate(&m_cuda_start);
    cudaEventCreate(&m_cuda_stop);

    cudaEventRecord(m_cuda_start, 0);

    m_overall_start = std::chrono::system_clock::now();
}

void CudaTimer::stop()
{
    auto m_cpu_stop = std::chrono::system_clock::now();

    cudaEventRecord(m_cuda_stop, 0);
    cudaEventSynchronize(m_cuda_stop);
    
    auto m_overall_stop = std::chrono::system_clock::now();
    
    float cuda_time;
    cudaEventElapsedTime(&cuda_time, m_cuda_start, m_cuda_stop);

    float cpu_time = std::chrono::duration<double>(m_cpu_stop - m_overall_start).count();
    float overall_time = std::chrono::duration<double>(m_overall_stop - m_overall_start).count();

    ADD_SAMPLE(m_name, cuda_time, cpu_time, overall_time);
}

struct Timer
{
    float cuda_time_sum, cpu_time_sum, overall_time_sum;
    int count;
};

std::map<std::string, Timer> TIMERS;

void ADD_SAMPLE(std::string name, float cuda_time, float cpu_time, float overall_time)
{
    auto itr = TIMERS.find(name);

    if (itr == TIMERS.end())
    {
        itr = TIMERS.insert(std::pair<std::string, Timer>(name, Timer())).first;
    }

    Timer* timer = &(itr->second);

    timer->cuda_time_sum += cuda_time;
    timer->cpu_time_sum += cpu_time;
    timer->overall_time_sum += overall_time;
    timer->count += 1;
}

std::vector<std::pair<std::string, std::vector<float>>> GET_TIMES()
{
    std::vector<std::pair<std::string, std::vector<float>>> vec;

    for(auto it = TIMERS.begin(); it != TIMERS.end(); ++it ) 
    {

        float cuda_time = 1e3 * it->second.cuda_time_sum / it->second.count;
        float cpu_time = 1e6 * it->second.cpu_time_sum / it->second.count;
        float overall_time = 1e6 * it->second.overall_time_sum / it->second.count;

        std::vector<float> times = std::vector<float>{cuda_time, cpu_time, overall_time};

        vec.push_back(std::pair<std::string, std::vector<float>>(it->first, times));
    }

    return vec;
}
