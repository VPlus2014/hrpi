#include "simulation.h"
#include <iostream>
#include <chrono>

Simulation::Simulation() : running(false), simulationTime(0.0), timeStep(0.1) {}

Simulation::~Simulation()
{
    stop();
}

void Simulation::initialize(int numBattlefields, int numAircraft, int numMissiles, int maxSteps)
{
    for (int i = 0; i < numBattlefields; ++i)
    {
        auto battlefield = std::make_unique<Battlefield>(i, 1000.0, 1000.0, maxSteps);
        battlefield->initialize(numAircraft, numMissiles);
        battlefields.push_back(std::move(battlefield));
    }
}

void Simulation::run()
{
    running = true;

    // 创建线程池
    for (auto &battlefield : battlefields)
    {
        threads.emplace_back(&Simulation::updateBattlefield, this, battlefield.get());
    }

    // 主循环
    auto lastTime = std::chrono::high_resolution_clock::now();
    while (running)
    {
        auto currentTime = std::chrono::high_resolution_clock::now();
        double deltaTime = std::chrono::duration<double>(currentTime - lastTime).count();
        lastTime = currentTime;

        // 更新仿真时间
        simulationTime += deltaTime;

        // 渲染所有战场
        for (const auto &battlefield : battlefields)
        {
            std::lock_guard<std::mutex> lock(mtx);
            battlefield->render();
        }

        // 检查是否所有战场都结束
        bool allOver = true;
        for (const auto &battlefield : battlefields)
        {
            if (battlefield->isBattleOver())
            {
                std::lock_guard<std::mutex> lock(mtx);
                battlefield->render();
            }
            else
            {
                allOver = false;
            }
        }

        if (allOver)
        {
            stop();
            printResults();
        }

        // 控制渲染频率
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // 等待所有线程结束
    for (auto &thread : threads)
    {
        if (thread.joinable())
        {
            thread.join();
        }
    }
}

void Simulation::stop()
{
    running = false;
}

void Simulation::updateBattlefield(Battlefield *battlefield)
{
    while (running && !battlefield->isBattleOver())
    {
        battlefield->update(timeStep);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void Simulation::printResults() const
{
    std::cout << "=== Simulation Results ===" << std::endl;
    std::cout << "Total Simulation Time: " << simulationTime << "s" << std::endl;

    for (const auto &battlefield : battlefields)
    {
        int activeAircraft, activeMissiles, escapedEntities;
        battlefield->getStatistics(activeAircraft, activeMissiles, escapedEntities);

        std::cout << "Battlefield " << battlefield->getID() << ": "
                  << "Aircraft Survived: " << activeAircraft << ", "
                  << "Missiles Intercepted: " << escapedEntities << std::endl;
    }
}

double Simulation::getSimulationTime() const
{
    return simulationTime;
}