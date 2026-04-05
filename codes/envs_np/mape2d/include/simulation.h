#pragma once
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include "battlefield.h"

class Simulation
{
private:
    std::vector<std::unique_ptr<Battlefield>> battlefields; // 所有战场
    std::vector<std::thread> threads;                       // 线程池
    std::mutex mtx;                                         // 互斥锁
    std::atomic<bool> running;                              // 运行标志
    double simulationTime;                                  // 仿真时间
    double timeStep;                                        // 时间步长

public:
    Simulation();
    ~Simulation();

    // 初始化仿真
    void initialize(int numBattlefields, int numAircraft, int numMissiles, int maxSteps);

    // 运行仿真
    void run();

    // 停止仿真
    void stop();

    // 并行更新战场
    void updateBattlefield(Battlefield *battlefield);

    // 输出仿真结果
    void printResults() const;

    // 获取仿真时间
    double getSimulationTime() const;
};