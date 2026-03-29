#include "Missile.h"
#include "Aircraft.h"
#include "Decoy.h"
#include <algorithm>
#include <cmath>

Missile::Missile(double x, double y, double vx, double vy, double health, double detectionRange, int id)
    : Entity(x, y, vx, vy, health, detectionRange, EntityType::MISSILE, id),
      maxSpeed(150.0), damage(50.0), target(nullptr), _isDecoyed(false) {}

void Missile::behave(std::vector<Entity *> &allEntities, double dt)
{
    if (state != EntityState::ACTIVE)
        return;

    // 检查是否被诱饵吸引
    if (!_isDecoyed)
    {
        _isDecoyed = checkDecoyAttraction(allEntities);
    }

    // 追踪目标
    trackTarget(allEntities, dt);

    // 如果有目标，尝试攻击
    if (target && target->getState() == EntityState::ACTIVE)
    {
        double dist = distanceTo(target);
        if (dist <= 3.0)
        { // 攻击距离阈值
            attackTarget();
        }
    }

    // 更新位置
    updatePosition(dt);
}

void Missile::trackTarget(std::vector<Entity *> &allEntities, double dt)
{
    // 如果没有目标或目标已被摧毁，寻找新目标
    if (!target || target->getState() != EntityState::ACTIVE)
    {
        target = nullptr;
        double minDist = std::numeric_limits<double>::max();

        // 查找最近的目标（飞机或诱饵）
        for (auto &entity : allEntities)
        {
            if ((entity->getType() == EntityType::AIRCRAFT ||
                 entity->getType() == EntityType::DECOY) &&
                entity->getState() == EntityState::ACTIVE)
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

void Missile::attackTarget()
{
    if (target && target->getState() == EntityState::ACTIVE)
    {
        // 对目标造成伤害
        target->takeDamage(damage);

        // 自毁（导弹是一次性的）
        destroy();
    }
}

bool Missile::checkDecoyAttraction(std::vector<Entity *> &allEntities)
{
    // to改进: 基于当前相对态势构建更复杂的随机诱骗模型

    // 检测范围内的诱饵
    auto detected = detectEntities(allEntities);

    for (auto &entity : detected)
    {
        if (entity->getType() == EntityType::DECOY)
        {
            Decoy *decoy = static_cast<Decoy *>(entity);

            // 计算导弹到诱饵的距离
            double dij = distanceTo(decoy);

            // 如果诱饵在吸引范围内，有概率被吸引
            if (dij <= decoy->getAttractionRange())
            {
                // 距离越近，被吸引的概率越高
                double attractionProbability = 1.0 - (dij / decoy->getAttractionRange());

                // 生成随机数判断是否被吸引
                if (dist(rng) < attractionProbability)
                {
                    // 如果被吸引，将目标设为诱饵
                    target = decoy;
                    return true;
                }
            }
        }
    }

    return false;
}

// 设置速度（由诱饵弹调用）
void Missile::setVelocity(double vx, double vy)
{
    this->vx = vx;
    this->vy = vy;
}

// 获取最大速度
double Missile::getMaxSpeed() const
{
    return maxSpeed;
}

// 获取是否被诱饵吸引
bool Missile::isDecoyed() const
{
    return _isDecoyed;
}