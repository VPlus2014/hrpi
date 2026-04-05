#pragma once
#include "entity.h"

class Missile : public Entity
{
private:
    double maxSpeed; // 最大速度
    double damage;   // 伤害值
    Entity *target;  // 目标实体
    bool _isDecoyed; // 是否被诱饵吸引

public:
    Missile(double x, double y, double vx, double vy, double health, double detectionRange, int id);

    // 实体行为
    void behave(std::vector<Entity *> &allEntities, double dt) override;

    // 追踪目标
    void trackTarget(std::vector<Entity *> &allEntities, double dt);

    // 攻击目标
    void attackTarget();

    // 检查是否被诱饵吸引
    bool checkDecoyAttraction(std::vector<Entity *> &allEntities);

    void setVelocity(double vx, double vy);

    double getMaxSpeed() const;

    bool isDecoyed() const;
};