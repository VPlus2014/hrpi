import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
from typing import List, Tuple, Dict, Any
import math

class UAV:
    """UAV智能体类"""
    def __init__(self, id: int, x: float, y: float, team: str, max_speed: float = 5.0):
        self.id = id
        self.x = x
        self.y = y
        self.team = team  # 'red' or 'blue'
        self.max_speed = max_speed
        self.alive = True
        self.health = 100.0
        self.ammo = 10
        self.detection_range = 50.0
        self.attack_range = 20.0
        self.communication_range = 30.0
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        
    def get_position(self) -> Tuple[float, float]:
        return self.x, self.y
    
    def move(self, action: np.ndarray, dt: float = 0.1):
        """根据动作移动UAV"""
        if not self.alive:
            return
            
        # action: [dx, dy, attack] 归一化到[-1, 1]
        dx, dy = action[0], action[1]
        
        # 限制速度
        speed = np.sqrt(dx**2 + dy**2)
        if speed > self.max_speed:
            dx = dx / speed * self.max_speed
            dy = dy / speed * self.max_speed
            
        self.velocity_x = dx
        self.velocity_y = dy
        self.x += dx * dt
        self.y += dy * dt
        
    def attack(self, target: 'UAV') -> bool:
        """攻击目标"""
        if not self.alive or not target.alive or self.ammo <= 0:
            return False
            
        distance = self.distance_to(target)
        if distance <= self.attack_range:
            self.ammo -= 1
            damage = max(0, 30 - distance)  # 距离越近伤害越高
            target.take_damage(damage)
            return True
        return False
    
    def take_damage(self, damage: float):
        """受到伤害"""
        self.health -= damage
        if self.health <= 0:
            self.alive = False
            
    def distance_to(self, other: 'UAV') -> float:
        """计算到另一个UAV的距离"""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def can_see(self, other: 'UAV') -> bool:
        """检查是否能探测到另一个UAV"""
        return self.alive and other.alive and self.distance_to(other) <= self.detection_range

class MultiUAVEnvironment:
    """多UAV对抗环境"""
    def __init__(self, map_size: Tuple[int, int] = (100, 100), red_uavs: int = 3, blue_uavs: int = 3):
        self.map_width, self.map_height = map_size
        self.red_uavs_count = red_uavs
        self.blue_uavs_count = blue_uavs
        self.uavs = []
        self.obstacles = []
        self.time_step = 0
        self.max_episode_steps = 1000
        
        # 添加一些障碍物
        self.add_obstacles()
        self.reset()
        
    def add_obstacles(self):
        """添加地图障碍物"""
        # 添加一些矩形障碍物
        obstacles = [
            (30, 30, 10, 10),
            (60, 20, 8, 15),
            (20, 70, 12, 8),
            (70, 80, 15, 10),
            (45, 50, 6, 6)
        ]
        self.obstacles = obstacles
        
    def reset(self) -> Dict[str, np.ndarray]:
        """重置环境"""
        self.uavs = []
        self.time_step = 0
        
        # 创建红队UAV
        for i in range(self.red_uavs_count):
            x = np.random.uniform(5, 25)
            y = np.random.uniform(5, self.map_height - 5)
            uav = UAV(i, x, y, 'red')
            self.uavs.append(uav)
            
        # 创建蓝队UAV
        for i in range(self.blue_uavs_count):
            x = np.random.uniform(self.map_width - 25, self.map_width - 5)
            y = np.random.uniform(5, self.map_height - 5)
            uav = UAV(i + self.red_uavs_count, x, y, 'blue')
            self.uavs.append(uav)
            
        return self.get_observations()
    
    def get_observations(self) -> Dict[str, np.ndarray]:
        """获取所有UAV的观测"""
        observations = {}
        
        for uav in self.uavs:
            if not uav.alive:
                continue
                
            obs = self._get_uav_observation(uav)
            observations[f"{uav.team}_{uav.id}"] = obs
            
        return observations
    
    def _get_uav_observation(self, uav: UAV) -> np.ndarray:
        """获取单个UAV的观测"""
        obs = []
        
        # 自身状态 [x, y, vx, vy, health, ammo] (6维)
        obs.extend([
            uav.x / self.map_width,
            uav.y / self.map_height,
            uav.velocity_x / uav.max_speed,
            uav.velocity_y / uav.max_speed,
            uav.health / 100.0,
            uav.ammo / 10.0
        ])
        
        # 队友信息 (最多2个队友，每个4维：相对位置和状态)
        teammates = [u for u in self.uavs if u.team == uav.team and u.id != uav.id and u.alive]
        for i in range(2):  # 假设最多2个队友
            if i < len(teammates):
                teammate = teammates[i]
                rel_x = (teammate.x - uav.x) / self.map_width
                rel_y = (teammate.y - uav.y) / self.map_height
                obs.extend([rel_x, rel_y, teammate.health / 100.0, teammate.ammo / 10.0])
            else:
                obs.extend([0, 0, 0, 0])
                
        # 敌人信息 (最多3个敌人，每个4维)
        enemies = [u for u in self.uavs if u.team != uav.team and u.alive and uav.can_see(u)]
        for i in range(3):  # 假设最多能看到3个敌人
            if i < len(enemies):
                enemy = enemies[i]
                rel_x = (enemy.x - uav.x) / self.map_width
                rel_y = (enemy.y - uav.y) / self.map_height
                obs.extend([rel_x, rel_y, enemy.health / 100.0, uav.distance_to(enemy) / uav.detection_range])
            else:
                obs.extend([0, 0, 0, 0])
        
        # 地图边界信息 (4维)
        obs.extend([
            uav.x / self.map_width,  # 到左边界距离
            (self.map_width - uav.x) / self.map_width,  # 到右边界距离
            uav.y / self.map_height,  # 到下边界距离
            (self.map_height - uav.y) / self.map_height   # 到上边界距离
        ])
        
        return np.array(obs, dtype=np.float32)
    
    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, Any]]:
        """环境步进"""
        rewards = {}
        dones = {}
        infos = {}
        
        # 移动所有UAV
        for uav in self.uavs:
            if not uav.alive:
                continue
                
            key = f"{uav.team}_{uav.id}"
            if key in actions:
                action = actions[key]
                uav.move(action[:2])  # 移动动作
                
                # 边界检查
                uav.x = np.clip(uav.x, 0, self.map_width)
                uav.y = np.clip(uav.y, 0, self.map_height)
                
                # 攻击动作
                if len(action) > 2 and action[2] > 0.5:  # 攻击阈值
                    self._handle_attack(uav)
        
        # 计算奖励
        for uav in self.uavs:
            key = f"{uav.team}_{uav.id}"
            rewards[key] = self._calculate_reward(uav)
            dones[key] = not uav.alive
            infos[key] = {'health': uav.health, 'ammo': uav.ammo, 'alive': uav.alive}
        
        self.time_step += 1
        
        # 检查回合结束条件
        red_alive = sum(1 for uav in self.uavs if uav.team == 'red' and uav.alive)
        blue_alive = sum(1 for uav in self.uavs if uav.team == 'blue' and uav.alive)
        
        episode_done = (red_alive == 0 or blue_alive == 0 or self.time_step >= self.max_episode_steps)
        
        if episode_done:
            for key in rewards:
                dones[key] = True
        
        observations = self.get_observations()
        return observations, rewards, dones, infos
    
    def _handle_attack(self, attacker: UAV):
        """处理攻击逻辑"""
        enemies = [uav for uav in self.uavs if uav.team != attacker.team and uav.alive]
        
        # 寻找攻击范围内的敌人
        targets_in_range = [enemy for enemy in enemies if attacker.distance_to(enemy) <= attacker.attack_range]
        
        if targets_in_range:
            # 攻击最近的敌人
            target = min(targets_in_range, key=lambda e: attacker.distance_to(e))
            attacker.attack(target)
    
    def _calculate_reward(self, uav: UAV) -> float:
        """计算UAV的奖励"""
        if not uav.alive:
            return -100.0  # 死亡惩罚
            
        reward = 0.0
        
        # 生存奖励
        reward += 1.0
        
        # 健康状态奖励
        reward += (uav.health / 100.0) * 0.5
        
        # 团队协作奖励
        teammates = [u for u in self.uavs if u.team == uav.team and u.id != uav.id and u.alive]
        if teammates:
            # 鼓励保持合理距离（不要太近也不要太远）
            avg_distance = np.mean([uav.distance_to(teammate) for teammate in teammates])
            optimal_distance = 25.0
            distance_reward = -abs(avg_distance - optimal_distance) / optimal_distance
            reward += distance_reward * 0.3
        
        # 对敌奖励
        enemies = [u for u in self.uavs if u.team != uav.team and u.alive]
        if enemies:
            # 鼓励接近敌人（在探测范围内）
            min_enemy_distance = min([uav.distance_to(enemy) for enemy in enemies])
            if min_enemy_distance <= uav.detection_range:
                reward += (uav.detection_range - min_enemy_distance) / uav.detection_range * 2.0
        
        # 边界惩罚
        border_penalty = 0
        if uav.x < 5 or uav.x > self.map_width - 5:
            border_penalty += 0.5
        if uav.y < 5 or uav.y > self.map_height - 5:
            border_penalty += 0.5
        reward -= border_penalty
        
        return reward

class DQNNetwork(nn.Module):
    """深度Q网络"""
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 9):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class MultiAgentDQN:
    """多智能体DQN训练器"""
    def __init__(self, obs_dim: int, action_dim: int = 9, lr: float = 0.001):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 为每个队伍创建网络
        self.red_q_network = DQNNetwork(obs_dim, output_dim=action_dim).to(self.device)
        self.red_target_network = DQNNetwork(obs_dim, output_dim=action_dim).to(self.device)
        self.blue_q_network = DQNNetwork(obs_dim, output_dim=action_dim).to(self.device)
        self.blue_target_network = DQNNetwork(obs_dim, output_dim=action_dim).to(self.device)
        
        self.red_optimizer = optim.Adam(self.red_q_network.parameters(), lr=lr)
        self.blue_optimizer = optim.Adam(self.blue_q_network.parameters(), lr=lr)
        
        # 经验回放缓冲区
        self.red_memory = deque(maxlen=10000)
        self.blue_memory = deque(maxlen=10000)
        
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.gamma = 0.99
        self.update_target_freq = 100
        self.training_step = 0
        
        # 初始化目标网络
        self.update_target_networks()
    
    def update_target_networks(self):
        """更新目标网络"""
        self.red_target_network.load_state_dict(self.red_q_network.state_dict())
        self.blue_target_network.load_state_dict(self.blue_q_network.state_dict())
    
    def get_action(self, obs: np.ndarray, team: str, training: bool = True) -> int:
        """获取动作"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        if team == 'red':
            q_values = self.red_q_network(obs_tensor)
        else:
            q_values = self.blue_q_network(obs_tensor)
            
        return q_values.argmax().item()
    
    def action_to_continuous(self, action: int) -> np.ndarray:
        """将离散动作转换为连续动作"""
        # 9个离散动作：8个移动方向 + 攻击
        actions = [
            [0, 0, 0],      # 0: 停止
            [1, 0, 0],      # 1: 右
            [-1, 0, 0],     # 2: 左
            [0, 1, 0],      # 3: 上
            [0, -1, 0],     # 4: 下
            [0.7, 0.7, 0],  # 5: 右上
            [-0.7, 0.7, 0], # 6: 左上
            [0.7, -0.7, 0], # 7: 右下
            [-0.7, -0.7, 0], # 8: 左下
        ]
        
        # 随机决定是否攻击
        base_action = actions[action]
        if np.random.random() < 0.1:  # 10%概率攻击
            base_action[2] = 1.0
            
        return np.array(base_action)
    
    def store_experience(self, team: str, obs: np.ndarray, action: int, reward: float, 
                        next_obs: np.ndarray, done: bool):
        """存储经验"""
        experience = (obs, action, reward, next_obs, done)
        if team == 'red':
            self.red_memory.append(experience)
        else:
            self.blue_memory.append(experience)
    
    def train(self):
        """训练网络"""
        if len(self.red_memory) < self.batch_size or len(self.blue_memory) < self.batch_size:
            return
        
        # 训练红队
        self._train_team('red')
        # 训练蓝队
        self._train_team('blue')
        
        self.training_step += 1
        
        # 更新目标网络
        if self.training_step % self.update_target_freq == 0:
            self.update_target_networks()
        
        # 更新epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def _train_team(self, team: str):
        """训练指定队伍"""
        memory = self.red_memory if team == 'red' else self.blue_memory
        q_network = self.red_q_network if team == 'red' else self.blue_q_network
        target_network = self.red_target_network if team == 'red' else self.blue_target_network
        optimizer = self.red_optimizer if team == 'red' else self.blue_optimizer
        
        batch = random.sample(memory, self.batch_size)
        obs_batch = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        action_batch = torch.LongTensor([e[1] for e in batch]).to(self.device)
        reward_batch = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_obs_batch = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        done_batch = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        current_q_values = q_network(obs_batch).gather(1, action_batch.unsqueeze(1))
        next_q_values = target_network(next_obs_batch).max(1)[0].detach()
        target_q_values = reward_batch + (self.gamma * next_q_values * ~done_batch)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

class UAVSimulationVisualizer:
    """UAV仿真可视化器"""
    def __init__(self, env: MultiUAVEnvironment):
        self.env = env
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.setup_plot()
        
    def setup_plot(self):
        """设置绘图"""
        self.ax.set_xlim(0, self.env.map_width)
        self.ax.set_ylim(0, self.env.map_height)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('多UAV对抗仿真环境', fontsize=16, fontweight='bold')
        
        # 绘制障碍物
        for obs in self.env.obstacles:
            x, y, w, h = obs
            rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                   edgecolor='black', facecolor='gray', alpha=0.7)
            self.ax.add_patch(rect)
    
    def update_plot(self):
        """更新绘图"""
        # 清除之前的UAV
        for artist in self.ax.lines + self.ax.collections:
            if hasattr(artist, '_uav_artist'):
                artist.remove()
        
        # 绘制UAV
        for uav in self.env.uavs:
            if not uav.alive:
                continue
                
            color = 'red' if uav.team == 'red' else 'blue'
            alpha = uav.health / 100.0
            
            # UAV位置
            circle = plt.Circle((uav.x, uav.y), 2, color=color, alpha=alpha)
            circle._uav_artist = True
            self.ax.add_patch(circle)
            
            # UAV ID
            self.ax.text(uav.x + 3, uav.y + 3, f'{uav.team[0].upper()}{uav.id}', 
                        fontsize=8, color=color)
            
            # 探测范围（半透明圆圈）
            detection_circle = plt.Circle((uav.x, uav.y), uav.detection_range, 
                                        color=color, alpha=0.1, fill=True)
            detection_circle._uav_artist = True
            self.ax.add_patch(detection_circle)
            
            # 攻击范围
            attack_circle = plt.Circle((uav.x, uav.y), uav.attack_range, 
                                     color=color, alpha=0.2, fill=False, linestyle='--')
            attack_circle._uav_artist = True
            self.ax.add_patch(attack_circle)
        
        # 显示统计信息
        red_alive = sum(1 for uav in self.env.uavs if uav.team == 'red' and uav.alive)
        blue_alive = sum(1 for uav in self.env.uavs if uav.team == 'blue' and uav.alive)
        
        self.ax.text(5, self.env.map_height - 5, f'红队存活: {red_alive}', 
                    fontsize=12, color='red', fontweight='bold')
        self.ax.text(5, self.env.map_height - 10, f'蓝队存活: {blue_alive}', 
                    fontsize=12, color='blue', fontweight='bold')
        self.ax.text(5, self.env.map_height - 15, f'时间步: {self.env.time_step}', 
                    fontsize=12, color='black')
        
        plt.draw()
        plt.pause(0.05)

class UAVTrainingManager:
    """UAV训练管理器"""
    def __init__(self):
        self.env = MultiUAVEnvironment()
        self.visualizer = UAVSimulationVisualizer(self.env)
        
        # 计算观测维度
        sample_obs = self.env.reset()
        obs_dim = len(list(sample_obs.values())[0])
        
        self.agent = MultiAgentDQN(obs_dim)
        self.episode_rewards = {'red': [], 'blue': []}
        self.win_rates = {'red': [], 'blue': []}
        
    def run_episode(self, render: bool = False, training: bool = True):
        """运行一个回合"""
        obs = self.env.reset()
        episode_rewards = {key: 0 for key in obs.keys()}
        done = False
        step_count = 0
        
        if render:
            self.visualizer.update_plot()
        
        while not done and step_count < 1000:
            actions = {}
            
            # 获取所有智能体的动作
            for key, observation in obs.items():
                team = key.split('_')[0]
                action_idx = self.agent.get_action(observation, team, training)
                actions[key] = self.agent.action_to_continuous(action_idx)
            
            # 执行动作
            next_obs, rewards, dones, infos = self.env.step(actions)
            
            # 存储经验（仅在训练时）
            if training:
                for key in obs.keys():
                    if key in next_obs:  # 确保智能体还存活
                        team = key.split('_')[0]
                        action_idx = self.agent.get_action(obs[key], team, False)  # 不使用随机动作获取动作索引
                        self.agent.store_experience(
                            team, obs[key], action_idx, rewards[key], 
                            next_obs[key], dones[key]
                        )
            
            # 累计奖励
            for key, reward in rewards.items():
                if key in episode_rewards:
                    episode_rewards[key] += reward
            
            obs = next_obs
            done = all(dones.values()) or len(obs) == 0
            step_count += 1
            
            if render:
                self.visualizer.update_plot()
        
        return episode_rewards, step_count
    
    def train(self, episodes: int = 1000, render_freq: int = 100):
        """训练智能体"""
        print("开始训练多UAV对抗系统...")
        
        for episode in range(episodes):
            render = (episode % render_freq == 0)
            episode_rewards, steps = self.run_episode(render=render, training=True)
            
            # 训练网络
            if episode > 50:  # 预热
                self.agent.train()
            
            # 记录奖励
            red_total = sum(reward for key, reward in episode_rewards.items() if 'red' in key)
            blue_total = sum(reward for key, reward in episode_rewards.items() if 'blue' in key)
            
            self.episode_rewards['red'].append(red_total)
            self.episode_rewards['blue'].append(blue_total)
            
            # 计算胜率
            red_alive = sum(1 for uav in self.env.uavs if uav.team == 'red' and uav.alive)
            blue_alive = sum(1 for uav in self.env.uavs if uav.team == 'blue' and uav.alive)
            
            if red_alive > blue_alive:
                winner = 'red'
            elif blue_alive > red_alive:
                winner = 'blue'
            else:
                winner = 'draw'
            
            # 打印训练信息
            if episode % 50 == 0:
                avg_red_reward = np.mean(self.episode_rewards['red'][-50:]) if len(self.episode_rewards['red']) >= 50 else red_total
                avg_blue_reward = np.mean(self.episode_rewards['blue'][-50:]) if len(self.episode_rewards['blue']) >= 50 else blue_total
                
                print(f"Episode {episode}: "
                      f"红队奖励: {avg_red_reward:.2f}, "
                      f"蓝队奖励: {avg_blue_reward:.2f}, "
                      f"胜者: {winner}, "
                      f"Epsilon: {self.agent.epsilon:.3f}, "
                      f"步数: {steps}")
        
        print("训练完成!")
    
    def evaluate(self, episodes: int = 10):
        """评估训练结果"""
        print("开始评估...")
        results = {'red_wins': 0, 'blue_wins': 0, 'draws': 0}
        
        for episode in range(episodes):
            episode_rewards, steps = self.run_episode(render=True, training=False)
            
            red_alive = sum(1 for uav in self.env.uavs if uav.team == 'red' and uav.alive)
            blue_alive = sum(1 for uav in self.env.uavs if uav.team == 'blue' and uav.alive)
            
            if red_alive > blue_alive:
                results['red_wins'] += 1
                winner = '红队胜利'
            elif blue_alive > red_alive:
                results['blue_wins'] += 1
                winner = '蓝队胜利'
            else:
                results['draws'] += 1
                winner = '平局'
            
            print(f"评估回合 {episode + 1}: {winner}, 步数: {steps}")
        
        print(f"\n评估结果:")
        print(f"红队胜率: {results['red_wins']/episodes*100:.1f}%")
        print(f"蓝队胜率: {results['blue_wins']/episodes*100:.1f}%")
        print(f"平局率: {results['draws']/episodes*100:.1f}%")
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 奖励曲线
        episodes = range(len(self.episode_rewards['red']))
        ax1.plot(episodes, self.episode_rewards['red'], label='红队', color='red', alpha=0.7)
        ax1.plot(episodes, self.episode_rewards['blue'], label='蓝队', color='blue', alpha=0.7)
        
        # 移动平均
        if len(self.episode_rewards['red']) > 50:
            red_ma = np.convolve(self.episode_rewards['red'], np.ones(50)/50, mode='valid')
            blue_ma = np.convolve(self.episode_rewards['blue'], np.ones(50)/50, mode='valid')
            ax1.plot(range(49, len(red_ma) + 49), red_ma, label='红队(移动平均)', color='darkred', linewidth=2)
            ax1.plot(range(49, len(blue_ma) + 49), blue_ma, label='蓝队(移动平均)', color='darkblue', linewidth=2)
        
        ax1.set_xlabel('回合')
        ax1.set_ylabel('累计奖励')
        ax1.set_title('训练奖励曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 胜率统计（最近100回合的滑动窗口）
        window_size = 100
        if len(self.episode_rewards['red']) >= window_size:
            red_wins = []
            blue_wins = []
            draws = []
            
            for i in range(window_size, len(self.episode_rewards['red']) + 1):
                red_slice = self.episode_rewards['red'][i-window_size:i]
                blue_slice = self.episode_rewards['blue'][i-window_size:i]
                
                red_win_count = sum(1 for j in range(len(red_slice)) if red_slice[j] > blue_slice[j])
                blue_win_count = sum(1 for j in range(len(blue_slice)) if blue_slice[j] > red_slice[j])
                draw_count = window_size - red_win_count - blue_win_count
                
                red_wins.append(red_win_count / window_size)
                blue_wins.append(blue_win_count / window_size)
                draws.append(draw_count / window_size)
            
            episodes_range = range(window_size, len(self.episode_rewards['red']) + 1)
            ax2.plot(episodes_range, red_wins, label='红队胜率', color='red')
            ax2.plot(episodes_range, blue_wins, label='蓝队胜率', color='blue')
            ax2.plot(episodes_range, draws, label='平局率', color='gray')
        
        ax2.set_xlabel('回合')
        ax2.set_ylabel('胜率')
        ax2.set_title(f'胜率趋势(滑动窗口={window_size})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.show()

class ConfigurableScenario:
    """可配置的对抗场景"""
    def __init__(self):
        self.scenarios = {
            'balanced': {'red_uavs': 3, 'blue_uavs': 3, 'map_size': (100, 100)},
            'outnumbered': {'red_uavs': 2, 'blue_uavs': 4, 'map_size': (100, 100)},
            'large_scale': {'red_uavs': 5, 'blue_uavs': 5, 'map_size': (150, 150)},
            'asymmetric': {'red_uavs': 4, 'blue_uavs': 2, 'map_size': (80, 120)},
        }
    
    def create_scenario(self, scenario_name: str) -> MultiUAVEnvironment:
        """创建指定场景"""
        if scenario_name not in self.scenarios:
            raise ValueError(f"未知场景: {scenario_name}")
        
        config = self.scenarios[scenario_name]
        return MultiUAVEnvironment(
            map_size=config['map_size'],
            red_uavs=config['red_uavs'],
            blue_uavs=config['blue_uavs']
        )

# 主程序执行部分
def main():
    """主训练程序"""
    print("=" * 60)
    print("多UAV对抗场景仿真与群体决策强化学习系统")
    print("=" * 60)
    
    # 创建场景管理器
    scenario_manager = ConfigurableScenario()
    
    # 选择场景
    print("\n可用场景:")
    for name, config in scenario_manager.scenarios.items():
        print(f"  {name}: 红队{config['red_uavs']}架 vs 蓝队{config['blue_uavs']}架, "
              f"地图{config['map_size']}")
    
    scenario_name = 'balanced'  # 默认使用平衡场景
    print(f"\n使用场景: {scenario_name}")
    
    # 创建环境和训练管理器
    env = scenario_manager.create_scenario(scenario_name)
    trainer = UAVTrainingManager()
    trainer.env = env
    trainer.visualizer = UAVSimulationVisualizer(env)
    
    # 重新初始化智能体（因为观测维度可能不同）
    sample_obs = env.reset()
    obs_dim = len(list(sample_obs.values())[0])
    trainer.agent = MultiAgentDQN(obs_dim)
    
    print(f"\n环境信息:")
    print(f"  地图大小: {env.map_width} x {env.map_height}")
    print(f"  UAV数量: 红队{env.red_uavs_count}架, 蓝队{env.blue_uavs_count}架")
    print(f"  观测维度: {obs_dim}")
    print(f"  动作维度: {trainer.agent.action_dim}")
    
    # 训练选项
    train_episodes = 500  # 可以调整训练回合数
    
    print(f"\n开始训练 {train_episodes} 回合...")
    
    # 执行训练
    trainer.train(episodes=train_episodes, render_freq=100)
    
    # 绘制训练曲线
    trainer.plot_training_curves()
    
    # 评估性能
    print("\n开始性能评估...")
    trainer.evaluate(episodes=10)
    
    print("\n训练和评估完成!")
    return trainer

# 高级功能扩展
class AdvancedUAVFeatures:
    """高级UAV功能扩展"""
    
    @staticmethod
    def add_formation_control(uavs: List[UAV], formation_type: str = 'triangle'):
        """添加编队控制"""
        if formation_type == 'triangle' and len(uavs) >= 3:
            # 三角形编队逻辑
            leader = uavs[0]
            if len(uavs) > 1:
                follower1 = uavs[1]
                follower1.target_x = leader.x - 10
                follower1.target_y = leader.y - 10
            if len(uavs) > 2:
                follower2 = uavs[2]
                follower2.target_x = leader.x - 10
                follower2.target_y = leader.y + 10
                
    @staticmethod
    def calculate_swarm_intelligence_bonus(team_uavs: List[UAV]) -> float:
        """计算群体智能奖励"""
        if len(team_uavs) < 2:
            return 0.0
        
        # 计算团队凝聚力
        positions = [(uav.x, uav.y) for uav in team_uavs if uav.alive]
        if len(positions) < 2:
            return 0.0
        
        center_x = sum(pos[0] for pos in positions) / len(positions)
        center_y = sum(pos[1] for pos in positions) / len(positions)
        
        avg_distance_to_center = sum(
            np.sqrt((pos[0] - center_x)**2 + (pos[1] - center_y)**2) 
            for pos in positions
        ) / len(positions)
        
        # 适度的凝聚力有奖励
        optimal_distance = 20.0
        cohesion_bonus = max(0, 1 - abs(avg_distance_to_center - optimal_distance) / optimal_distance)
        
        return cohesion_bonus * 5.0

class PerformanceAnalyzer:
    """性能分析器"""
    def __init__(self):
        self.battle_logs = []
        
    def log_battle(self, env: MultiUAVEnvironment, episode: int):
        """记录战斗日志"""
        red_status = [(uav.health, uav.ammo) for uav in env.uavs if uav.team == 'red']
        blue_status = [(uav.health, uav.ammo) for uav in env.uavs if uav.team == 'blue']
        
        log_entry = {
            'episode': episode,
            'time_step': env.time_step,
            'red_alive': sum(1 for uav in env.uavs if uav.team == 'red' and uav.alive),
            'blue_alive': sum(1 for uav in env.uavs if uav.team == 'blue' and uav.alive),
            'red_total_health': sum(status[0] for status in red_status),
            'blue_total_health': sum(status[0] for status in blue_status),
            'red_total_ammo': sum(status[1] for status in red_status),
            'blue_total_ammo': sum(status[1] for status in blue_status),
        }
        
        self.battle_logs.append(log_entry)
    
    def generate_report(self):
        """生成分析报告"""
        if not self.battle_logs:
            print("没有可用的战斗日志")
            return
        
        print("\n" + "=" * 50)
        print("战斗分析报告")
        print("=" * 50)
        
        total_battles = len(self.battle_logs)
        red_wins = sum(1 for log in self.battle_logs if log['red_alive'] > log['blue_alive'])
        blue_wins = sum(1 for log in self.battle_logs if log['blue_alive'] > log['red_alive'])
        draws = total_battles - red_wins - blue_wins
        
        print(f"总战斗次数: {total_battles}")
        print(f"红队胜利: {red_wins} ({red_wins/total_battles*100:.1f}%)")
        print(f"蓝队胜利: {blue_wins} ({blue_wins/total_battles*100:.1f}%)")
        print(f"平局: {draws} ({draws/total_battles*100:.1f}%)")
        
        avg_duration = np.mean([log['time_step'] for log in self.battle_logs])
        print(f"平均战斗时长: {avg_duration:.1f} 步")
        
        # 资源利用效率
        avg_red_health_remaining = np.mean([log['red_total_health'] for log in self.battle_logs])
        avg_blue_health_remaining = np.mean([log['blue_total_health'] for log in self.battle_logs])
        
        print(f"红队平均剩余血量: {avg_red_health_remaining:.1f}")
        print(f"蓝队平均剩余血量: {avg_blue_health_remaining:.1f}")

# 使用示例和测试
def run_demonstration():
    """运行演示"""
    print("运行多UAV对抗仿真演示...")
    
    # 创建训练管理器
    trainer = UAVTrainingManager()
    
    # 运行几个测试回合
    print("运行测试回合...")
    for i in range(3):
        print(f"\n测试回合 {i+1}:")
        episode_rewards, steps = trainer.run_episode(render=True, training=False)
        
        red_total = sum(reward for key, reward in episode_rewards.items() if 'red' in key)
        blue_total = sum(reward for key, reward in episode_rewards.items() if 'blue' in key)
        
        red_alive = sum(1 for uav in trainer.env.uavs if uav.team == 'red' and uav.alive)
        blue_alive = sum(1 for uav in trainer.env.uavs if uav.team == 'blue' and uav.alive)
        
        print(f"  红队奖励: {red_total:.2f}, 蓝队奖励: {blue_total:.2f}")
        print(f"  存活UAV: 红队{red_alive}架, 蓝队{blue_alive}架")
        print(f"  回合步数: {steps}")
    
    return trainer

# 专门的训练配置类
class TrainingConfig:
    """训练配置管理"""
    def __init__(self):
        self.scenarios = ['balanced', 'outnumbered', 'large_scale', 'asymmetric']
        self.hyperparameters = {
            'learning_rate': 0.001,
            'epsilon_start': 1.0,
            'epsilon_end': 0.1,
            'epsilon_decay': 0.995,
            'batch_size': 32,
            'memory_size': 10000,
            'gamma': 0.99,
            'update_target_freq': 100,
            'hidden_dim': 256
        }
        
    def create_custom_training(self, scenario: str, episodes: int = 1000):
        """创建自定义训练配置"""
        scenario_manager = ConfigurableScenario()
        env = scenario_manager.create_scenario(scenario)
        
        trainer = UAVTrainingManager()
        trainer.env = env
        trainer.visualizer = UAVSimulationVisualizer(env)
        
        # 重新初始化智能体
        sample_obs = env.reset()
        obs_dim = len(list(sample_obs.values())[0])
        trainer.agent = MultiAgentDQN(obs_dim, lr=self.hyperparameters['learning_rate'])
        
        return trainer, episodes

# 多场景训练管理器
class MultiScenarioTrainer:
    """多场景训练管理器"""
    def __init__(self):
        self.config = TrainingConfig()
        self.results = {}
        self.analyzer = PerformanceAnalyzer()
        
    def train_all_scenarios(self, episodes_per_scenario: int = 500):
        """在所有场景上训练"""
        print("开始多场景训练...")
        
        for scenario in self.config.scenarios:
            print(f"\n{'='*20} 训练场景: {scenario} {'='*20}")
            
            trainer, episodes = self.config.create_custom_training(scenario, episodes_per_scenario)
            
            # 训练
            trainer.train(episodes=episodes, render_freq=episodes//5)
            
            # 评估
            print(f"\n评估场景: {scenario}")
            trainer.evaluate(episodes=20)
            
            # 记录结果
            self.results[scenario] = {
                'trainer': trainer,
                'final_red_reward': np.mean(trainer.episode_rewards['red'][-50:]),
                'final_blue_reward': np.mean(trainer.episode_rewards['blue'][-50:])
            }
            
            # 记录到分析器
            for i in range(20):  # 最后20个回合的日志
                self.analyzer.log_battle(trainer.env, len(trainer.episode_rewards['red']) - 20 + i)
        
        # 生成总体报告
        self.generate_multi_scenario_report()
    
    def generate_multi_scenario_report(self):
        """生成多场景分析报告"""
        print("\n" + "=" * 70)
        print("多场景训练总结报告")
        print("=" * 70)
        
        for scenario, result in self.results.items():
            print(f"\n场景: {scenario}")
            print(f"  红队最终平均奖励: {result['final_red_reward']:.2f}")
            print(f"  蓝队最终平均奖励: {result['final_blue_reward']:.2f}")
            
            if result['final_red_reward'] > result['final_blue_reward']:
                print(f"  优势方: 红队")
            elif result['final_blue_reward'] > result['final_red_reward']:
                print(f"  优势方: 蓝队")
            else:
                print(f"  结果: 平衡")
        
        # 生成性能分析
        self.analyzer.generate_report()

# 如果直接运行此文件
if __name__ == "__main__":
    # 选择运行模式
    print("多UAV对抗仿真与强化学习系统")
    print("1. 快速演示")
    print("2. 单场景训练")
    print("3. 多场景训练")
    
    mode = input("请选择模式 (1/2/3): ").strip()
    
    if mode == '1':
        # 快速演示
        trainer = run_demonstration()
    elif mode == '2':
        # 单场景训练
        trainer = main()
    elif mode == '3':
        # 多场景训练
        multi_trainer = MultiScenarioTrainer()
        multi_trainer.train_all_scenarios(episodes_per_scenario=300)
    else:
        print("运行默认演示...")
        trainer = run_demonstration()
    
    print("\n程序执行完成!")