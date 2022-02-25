#include "profiling.h"

#include <map>

struct Timer
{
    float sum;
    int count;
};

std::map<std::string, Timer> TIMERS;

void ADD_SAMPLE(std::string name, float time)
{
    auto itr = TIMERS.find(name);

    if (itr == TIMERS.end())
    {
        itr = TIMERS.insert(std::pair<std::string, Timer>(name, Timer())).first;
    }

    Timer* timer = &(itr->second);

    timer->sum += time;
    timer->count += 1;
}

std::vector<std::pair<std::string, float>> GET_TIMES()
{
    std::vector<std::pair<std::string, float>> vec;

    for(auto it = TIMERS.begin(); it != TIMERS.end(); ++it ) {
        float time = it->second.sum / it->second.count;
        vec.push_back(std::pair<std::string, float>(it->first, time));
    }

    return vec;
}
