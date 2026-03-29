#include "decoy.h"
#include "missile.h"
#include <algorithm>
#include <cmath>

Decoy::Decoy(double x, double y, double vx, double vy, double health, double detectionRange, int id)
    : Entity(x, y, vx, vy, health, detectionRange, EntityType::DECOY, id),
      maxSpeed(80.0), attractionRange(100.0), lifetime(10.0) {}

void Decoy::behave(std::vector<Entity *> &allEntities, double dt)
{
    if (state != EntityState::ACTIVE)
        return;

    // 吸引导弹
    attractMissiles(allEntities, dt);

    // 更新生命周期
    updateLifetime(dt);

    // 更新位置
    updatePosition(dt);
}

void Decoy::attractMissiles(std::vector<Entity *> &allEntities, double dt)
{
    // 检测范围内的导弹
    auto detected = detectEntities(allEntities);

    for (auto &entity : detected)
    {
        if (entity->getType() == EntityType::MISSILE)
        {
            Missile *missile = static_cast<Missile *>(entity);

            // 计算导弹到诱饵的距离
            double dist = distanceTo(missile);

            // 如果导弹在吸引范围内，尝试吸引它
            if (dist <= attractionRange)
            {
                // 计算吸引方向（从导弹指向诱饵）
                double dx = x - missile->getPosition().first;
                double dy = y - missile->getPosition().second;
                double norm = std::sqrt(dx * dx + dy * dy);

                if (norm > 0)
                {
                    // 归一化方向向量
                    dx /= norm;
                    dy /= norm;

                    // 计算吸引强度（距离越近，吸引力越强）
                    double attractionStrength = (1.0 - dist / attractionRange) * 50.0;

                    // 获取导弹当前速度
                    auto [mvx, mvy] = missile->getVelocity();

                    // 计算新速度（混合原速度和吸引方向）
                    double newVx = mvx * 0.7 + dx * attractionStrength;
                    double newVy = mvy * 0.7 + dy * attractionStrength;

                    // 限制速度不超过导弹最大速度
                    double speed = std::sqrt(newVx * newVx + newVy * newVy);
                    if (speed > missile->getMaxSpeed())
                    {
                        newVx = newVx / speed * missile->getMaxSpeed();
                        newVy = newVy / speed * missile->getMaxSpeed();
                    }

                    // 更新导弹速度（这里需要修改Missile类以支持速度设置）
                    missile->setVelocity(newVx, newVy);
                }
            }
        }
    }
}

void Decoy::updateLifetime(double dt)
{
    lifetime -= dt;

    // 生命周期结束，自毁
    if (lifetime <= 0.0)
    {
        destroy();
    }
}

// 获取最大速度
double Decoy::getMaxSpeed() const
{
    return maxSpeed;
}

// 获取吸引范围
double Decoy::getAttractionRange() const
{
    return attractionRange;
}

// 获取剩余生命周期
double Decoy::getLifetime() const
{
    return lifetime;
}