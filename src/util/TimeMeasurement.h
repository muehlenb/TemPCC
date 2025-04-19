#include <chrono>

using namespace std::chrono;

struct TimeMeasurement {
    std::map<std::string, high_resolution_clock::time_point> startTimeMeasureMap;
    std::map<std::string, float> timeMeasureMap;

    void startTimeMeasure(std::string str){
        auto time = high_resolution_clock::now();
        startTimeMeasureMap[str] = time;
    }

    void endTimeMeasure(std::string str){
        auto time = high_resolution_clock::now();

        float duration = duration_cast<microseconds>(time - startTimeMeasureMap[str]).count() / 1000.f;
        //timeMeasureMap[str] = duration;
        timeMeasureMap[str] = duration * 0.05f + timeMeasureMap[str] * 0.95f;
    }

    std::map<std::string, float> getTimeMeasuresInMilliSec(){
        return timeMeasureMap;
    }

    float getTimeMeasureInMilliSec(std::string str){
        return timeMeasureMap[str];
    }
};
