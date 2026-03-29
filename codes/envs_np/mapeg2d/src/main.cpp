#include <iostream>

#include "simulation.h"

int main()
{
    // 创建仿真环境
    Simulation simulation;

    // 初始化参数
    int numBattlefields = 1; // 并行战场数量
    int numAircraft = 1;     // 每个战场的友方飞机数量
    int numMissiles = 6;     // 每个战场的敌方导弹数量
    int maxSteps = 1000;     // 最大步数

    // 初始化仿真
    simulation.initialize(numBattlefields, numAircraft, numMissiles, maxSteps);

    // 运行仿真
    std::cout
        << "Starting multi-agent simulation with " << numBattlefields
        << " parallel battlefields..." << std::endl;
    try
    {
        simulation.run();
    }
    catch (const std::exception &e)
    {
        std::cout << "Simulation stopped with exception: " << e.what() << std::endl;
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "Simulation stopped." << std::endl;
    return 0;
}