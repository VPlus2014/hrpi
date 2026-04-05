#pragma once
#include <vector>
#include <memory>
#include <random>
#include "entity.h"
#include "aircraft.h"
#include "interceptor.h"
#include "decoy.h"
#include "missile.h"

class Battlefield
{
private:
    std::vector<std::unique_ptr<Entity>> entities; // 所有实体
    int battlefieldId;                             // 战场ID
    double width;                                  // 战场宽度
    double height;                                 // 战场高度
    double time;                                   // 仿真时间
    int steps;                                     // 仿真步数
    int maxSteps;                                  // 最大仿真步数
    bool isActive;                                 // 战场是否活跃

    // 添加随机数生成器成员
    std::mt19937 rng;                            // 随机数生成器
    std::uniform_real_distribution<double> dist; // 均匀分布

public:
    Battlefield(int id, double width, double height,int maxSteps);

    // 初始化战场
    void initialize(int numAircraft, int numMissiles);

    // 更新战场状态
    void update(double dt);

    // 渲染战场（文本输出）
    void render() const;

    // 检查战场是否结束
    bool isBattleOver() const;

    // 获取战场统计信息
    void getStatistics(int &activeAircraft, int &activeMissiles, int &escapedEntities) const;

    // 获取战场ID
    int getID() const;

    // 添加实体
    void addEntity(std::unique_ptr<Entity> entity);

    // 移除销毁的实体
    void removeDestroyedEntities();
};