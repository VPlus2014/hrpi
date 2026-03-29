#include "interceptor.h"
#include "missile.h"
#include <algorithm>
#include <cmath>

Interceptor::Interceptor(double x, double y, double vx, double vy, double health, double detectionRange, int id)
    : Entity(x, y, vx, vy, health, detectionRange, EntityType::INTERCEPTOR, id),
      maxSpeed(200.0), damage(40.0), target(nullptr) {}

void Interceptor::behave(std::vector<Entity *> &allEntities, double dt)
{
    if (state != EntityState::ACTIVE)
        return;

    // 检测目标
    trackTarget(allEntities, dt);

    // 如果有目标，尝试攻击
    if (target && target->getState() == EntityState::ACTIVE)
    {
        double dist = distanceTo(target);
        if (dist <= 5.0)
        { // 攻击距离阈值
            attackTarget();
        }
    }

    // 更新位置
    updatePosition(dt);
}

void Interceptor::trackTarget(std::vector<Entity *> &allEntities, double dt)
{
    // 如果没有目标或目标已被摧毁，寻找新目标
    if (!target || target->getState() != EntityState::ACTIVE)
    {
        target = nullptr;
        double minDist = std::numeric_limits<double>::max();

        // 查找最近的敌方导弹
        for (auto &entity : allEntities)
        {
            if (entity->getType() == EntityType::MISSILE && entity->getState() == EntityState::ACTIVE)
            {
                double dist = distanceTo(entity);
                if (dist < minDist && dist <= detectionRange)
                {
                    minDist = dist;
                    target = entity;
                }
            }
        }
    }

    // 如果有目标，追踪目标
    if (target && target->getState() == EntityState::ACTIVE)
    {
        moveTo(target->getPosition().first, target->getPosition().second, maxSpeed);
    }
    else
    {
        // 没有目标时保持当前速度
        // 可以添加巡逻行为或其他默认行为
    }
}

void Interceptor::attackTarget()
{
    if (target && target->getState() == EntityState::ACTIVE)
    {
        // 对目标造成伤害
        target->takeDamage(damage);

        // 自毁（拦截弹是一次性的）
        destroy();
    }
}