#include "battlefield.h"
#include <algorithm>
#include <chrono>

Battlefield::Battlefield(int id, double width, double height, int maxSteps)
    : battlefieldId(id), width(width), height(height),
      isActive(true),
      time(0.0),
      maxSteps(maxSteps), steps(0),
      rng((unsigned int)std::chrono::system_clock::now().time_since_epoch().count()), // 使用时间作为种子
      dist(0.0, 1.0)                                                                  // 初始化均匀分布 [0, 1)
{
}

void Battlefield::initialize(int numAircraft, int numMissiles)
{
    // 创建友方飞机
    for (int i = 0; i < numAircraft; ++i)
    {
        double x = dist(rng) * width;
        double y = dist(rng) * height;
        double vx = (dist(rng) - 0.5) * 50;
        double vy = (dist(rng) - 0.5) * 50;

        entities.push_back(std::make_unique<Aircraft>(
            x, y, vx, vy, 100.0, 200.0, i));
    }

    // 创建敌方导弹
    for (int i = 0; i < numMissiles; ++i)
    {
        double x = dist(rng) * width;
        double y = dist(rng) * height;
        double vx = (dist(rng) - 0.5) * 80;
        double vy = (dist(rng) - 0.5) * 80;

        entities.push_back(std::make_unique<Missile>(
            x, y, vx, vy, 30.0, 150.0, numAircraft + i));
    }
}

void Battlefield::update(double dt)
{
    if (!isActive)
        return;

    // 更新所有实体
    std::vector<Entity *> entityPtrs;
    for (auto &entity : entities)
    {
        entityPtrs.push_back(entity.get());
    }

    for (auto &entity : entities)
    {
        entity->behave(entityPtrs, dt);
    }

    // 移除销毁的实体
    removeDestroyedEntities();

    // 更新时间
    time += dt;
    steps++;

    // 检查战场是否结束
    if (isBattleOver())
    {
        isActive = false;
    }
}

void Battlefield::render() const
{
    std::cout << "=== Battlefield " << battlefieldId
              << " Time=" << time << "s" << " Steps=" << steps
              << " ===" << std::endl;

    const int nrows(10), ncols(2 * nrows), nr2(nrows + 2), nc2(ncols + 2);
    char cells[nr2][nc2 + 1];
    char wall('*'), blanck(' '), cell;
    
    for (int x = 0; x < nc2; ++x)
    {
        cells[0][x] = cells[nrows + 1][x] = wall;
    }
    cells[0][nc2] = cells[nrows + 1][nc2] = '\0';
    for (int y = 1; y < nr2 - 1; ++y)
    {
        memset(cells[y], blanck, nc2);
        cells[y][0] = cells[y][ncols + 1] = wall;
        cells[y][ncols + 2] = '\0';
    }

    // 绘制实体
    for (const auto &entity : entities)
    {
        if (entity->getState() == EntityState::ACTIVE)
        {
            auto pos = entity->getPosition();
            int ex = static_cast<int>(ncols * pos.first / width);
            int ey = static_cast<int>(nrows * pos.second / height);

            if (ex >= 0 && ex < ncols && ey >= 0 && ey < nrows)
            {
                // std::cout << "Entity " << entity->getID() << " at (" << ex << ", " << ey << ")" << std::endl;
                cell = '?';
                switch (entity->getType())
                {
                case EntityType::AIRCRAFT:
                    cell = 'A';
                    break;
                case EntityType::INTERCEPTOR:
                    cell = 'I';
                    break;
                case EntityType::DECOY:
                    cell = 'D';
                    break;
                case EntityType::MISSILE:
                    cell = 'M';
                    break;
                }
                cells[ey + 1][ex + 1] = cell;
            }
        }
    }

    // 输出画面
    for (int y = 0; y < nrows + 2; ++y)
        std::cout << cells[y] << '\n';

    // 输出统计信息
    int activeAircraft = 0, activeMissiles = 0, escapedEntities = 0;
    getStatistics(activeAircraft, activeMissiles, escapedEntities);

    std::cout << "Active Aircraft: " << activeAircraft
              << ", Active Missiles: " << activeMissiles
              << ", Escaped Entities: " << escapedEntities << '\n';
    std::cout << std::endl;
}

bool Battlefield::isBattleOver() const
{
    if (steps >= maxSteps)
    {
        return true;
    }

    int activeAircraft = 0, activeMissiles = 0;
    for (const auto &entity : entities)
    {
        if (entity->getState() == EntityState::ACTIVE)
        {
            if (entity->getType() == EntityType::AIRCRAFT)
                activeAircraft++;
            else if (entity->getType() == EntityType::MISSILE)
                activeMissiles++;
        }
    }

    // 战场结束条件：所有飞机被摧毁或所有导弹被拦截
    return activeAircraft == 0 || activeMissiles == 0;
}

void Battlefield::getStatistics(int &activeAircraft, int &activeMissiles, int &escapedEntities) const
{
    activeAircraft = 0;
    activeMissiles = 0;
    escapedEntities = 0;

    for (const auto &entity : entities)
    {
        if (entity->getState() == EntityState::ACTIVE)
        {
            if (entity->getType() == EntityType::AIRCRAFT)
                activeAircraft++;
            else if (entity->getType() == EntityType::MISSILE)
                activeMissiles++;
        }
        else if (entity->getState() == EntityState::ESCAPED)
        {
            escapedEntities++;
        }
    }
}

int Battlefield::getID() const
{
    return battlefieldId;
}

void Battlefield::addEntity(std::unique_ptr<Entity> entity)
{
    entities.push_back(std::move(entity));
}

void Battlefield::removeDestroyedEntities()
{
    entities.erase(
        std::remove_if(entities.begin(), entities.end(),
                       [](const std::unique_ptr<Entity> &entity)
                       {
                           return entity->getState() == EntityState::DESTROYED;
                       }),
        entities.end());
}