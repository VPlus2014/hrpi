#pragma once
#include "entity.h"

class Aircraft : public Entity {
private:
    double missileCooldown;  // 导弹冷却时间
    double decoyCooldown;    // 诱饵冷却时间
    double maxSpeed;         // 最大速度
    double evasionSpeed;     // 躲避速度

public:
    Aircraft(double x, double y, double vx, double vy, double health, double detectionRange, int id);
    
    // 实体行为
    void behave(std::vector<Entity*>& allEntities, double dt) override;
    
    // 发射拦截弹
    void launchInterceptor(std::vector<Entity*>& allEntities);
    
    // 发射诱饵弹
    void launchDecoy(std::vector<Entity*>& allEntities);
    
    // 躲避导弹
    void evadeMissiles(std::vector<Entity*>& allEntities, double dt);
};