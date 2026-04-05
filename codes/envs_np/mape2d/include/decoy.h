#pragma once
#include "entity.h"

class Decoy : public Entity
{
private:
    double maxSpeed;        // 最大速度
    double attractionRange; // 吸引范围
    double lifetime;        // 生命周期

public:
    Decoy(double x, double y, double vx, double vy, double health, double detectionRange, int id);

    // 实体行为
    void behave(std::vector<Entity *> &allEntities, double dt) override;

    // 吸引导弹
    void attractMissiles(std::vector<Entity *> &allEntities, double dt);

    // 更新生命周期
    void updateLifetime(double dt);

    double getMaxSpeed() const;

    double getAttractionRange() const;

    double getLifetime() const;
};