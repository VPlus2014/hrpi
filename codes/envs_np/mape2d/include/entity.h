#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

// 实体类型枚举
enum class EntityType {
    AIRCRAFT,
    INTERCEPTOR,
    DECOY,
    MISSILE
};

// 实体状态枚举
enum class EntityState {
    ACTIVE,
    DESTROYED,
    ESCAPED
};

class Entity {
protected:
    double x, y;          // 位置坐标
    double vx, vy;        // 速度分量
    double health;        // 生命值
    double detectionRange; // 检测范围
    EntityType type;      // 实体类型
    EntityState state;   // 实体状态
    int id;              // 实体ID

    // 随机数生成器
    static std::mt19937 rng;
    static std::uniform_real_distribution<double> dist;

public:
    Entity(double x, double y, double vx, double vy, double health, double detectionRange, EntityType type, int id);
    virtual ~Entity() = default;

    // 更新位置
    virtual void updatePosition(double dt);
    
    // 检测范围内的其他实体
    virtual std::vector<Entity*> detectEntities(std::vector<Entity*>& allEntities);
    
    // 获取位置
    std::pair<double, double> getPosition() const;
    
    // 获取速度
    std::pair<double, double> getVelocity() const;
    
    // 获取类型
    EntityType getType() const;
    
    // 获取状态
    EntityState getState() const;
    
    // 获取ID
    int getID() const;
    
    // 受到伤害
    virtual void takeDamage(double damage);
    
    // 销毁实体
    virtual void destroy();
    
    // 纯虚函数：实体行为
    virtual void behave(std::vector<Entity*>& allEntities, double dt) = 0;
    
    // 计算与另一个实体的距离
    double distanceTo(const Entity* other) const;
    
    // 移动到目标位置
    virtual void moveTo(double targetX, double targetY, double speed);
};