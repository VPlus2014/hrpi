#include "aircraft.h"
#include "interceptor.h"
#include "decoy.h"
#include "missile.h"

Aircraft::Aircraft(double x, double y, double vx, double vy, double health, double detectionRange, int id)
    : Entity(x, y, vx, vy, health, detectionRange, EntityType::AIRCRAFT, id),
      missileCooldown(0.0), decoyCooldown(0.0), maxSpeed(100.0), evasionSpeed(120.0) {}

void Aircraft::behave(std::vector<Entity *> &allEntities, double dt)
{
    if (state != EntityState::ACTIVE)
        return;

    // 更新冷却时间
    if (missileCooldown > 0)
        missileCooldown -= dt;
    if (decoyCooldown > 0)
        decoyCooldown -= dt;

    // 检测威胁
    auto detected = detectEntities(allEntities);

    // 躲避导弹
    evadeMissiles(detected, dt);

    // 发射拦截弹
    if (missileCooldown <= 0)
    {
        launchInterceptor(allEntities);
    }

    // 发射诱饵弹
    if (decoyCooldown <= 0)
    {
        launchDecoy(allEntities);
    }

    // 更新位置
    updatePosition(dt);
}

void Aircraft::launchInterceptor(std::vector<Entity *> &allEntities)
{
    // 查找最近的敌方导弹
    Entity *nearestMissile = nullptr;
    double minDist = std::numeric_limits<double>::max();

    for (auto &entity : allEntities)
    {
        if (entity->getType() == EntityType::MISSILE && entity->getState() == EntityState::ACTIVE)
        {
            double dist = distanceTo(entity);
            if (dist < minDist)
            {
                minDist = dist;
                nearestMissile = entity;
            }
        }
    }

    if (nearestMissile && minDist <= detectionRange)
    {
        // 创建拦截弹
        auto interceptor = std::make_unique<Interceptor>(
            x, y, vx, vy, 50.0, 150.0, static_cast<int>(allEntities.size()));
        allEntities.push_back(interceptor.release());
        missileCooldown = 2.0; // 冷却时间2秒
    }
}

void Aircraft::launchDecoy(std::vector<Entity *> &allEntities)
{
    // 创建诱饵弹
    auto decoy = std::make_unique<Decoy>(
        x, y, vx * 0.5, vy * 0.5, 10.0, 200.0, static_cast<int>(allEntities.size()));
    allEntities.push_back(decoy.release());
    decoyCooldown = 5.0; // 冷却时间5秒
}

void Aircraft::evadeMissiles(std::vector<Entity *> &allEntities, double dt)
{
    // 计算威胁方向
    double threatX = 0.0, threatY = 0.0;
    int threatCount = 0;

    for (auto &entity : allEntities)
    {
        if (entity->getType() == EntityType::MISSILE && entity->getState() == EntityState::ACTIVE)
        {
            double dist = distanceTo(entity);
            if (dist <= detectionRange * 0.8)
            { // 80%检测范围内视为威胁
                double dx = entity->getPosition().first - x;
                double dy = entity->getPosition().second - y;
                double norm = std::sqrt(dx * dx + dy * dy);
                if (norm > 0)
                {
                    threatX += dx / norm;
                    threatY += dy / norm;
                    threatCount++;
                }
            }
        }
    }

    if (threatCount > 0)
    {
        // 计算躲避方向（垂直于威胁方向）
        double evadeX = -threatY;
        double evadeY = threatX;

        // 设置躲避速度
        vx = evadeX * evasionSpeed;
        vy = evadeY * evasionSpeed;
    }
}