#include "entity.h"

// 静态成员初始化
std::mt19937 Entity::rng(std::random_device{}());
std::uniform_real_distribution<double> Entity::dist(0.0, 1.0);

Entity::Entity(double x, double y, double vx, double vy, double health, double detectionRange, EntityType type, int id)
    : x(x), y(y), vx(vx), vy(vy), health(health), detectionRange(detectionRange), type(type), state(EntityState::ACTIVE), id(id) {}

void Entity::updatePosition(double dt) {
    x += vx * dt;
    y += vy * dt;
}

std::vector<Entity*> Entity::detectEntities(std::vector<Entity*>& allEntities) {
    std::vector<Entity*> detected;
    for (auto& entity : allEntities) {
        if (entity != this && entity->getState() == EntityState::ACTIVE) {
            double dist = distanceTo(entity);
            if (dist <= detectionRange) {
                detected.push_back(entity);
            }
        }
    }
    return detected;
}

std::pair<double, double> Entity::getPosition() const {
    return {x, y};
}

std::pair<double, double> Entity::getVelocity() const {
    return {vx, vy};
}

EntityType Entity::getType() const {
    return type;
}

EntityState Entity::getState() const {
    return state;
}

int Entity::getID() const {
    return id;
}

void Entity::takeDamage(double damage) {
    health -= damage;
    if (health <= 0) {
        destroy();
    }
}

void Entity::destroy() {
    state = EntityState::DESTROYED;
}

double Entity::distanceTo(const Entity* other) const {
    double dx = x - other->x;
    double dy = y - other->y;
    return std::sqrt(dx * dx + dy * dy);
}

void Entity::moveTo(double targetX, double targetY, double speed) {
    double dx = targetX - x;
    double dy = targetY - y;
    double distance = std::sqrt(dx * dx + dy * dy);
    
    if (distance > 0) {
        vx = (dx / distance) * speed;
        vy = (dy / distance) * speed;
    }
}