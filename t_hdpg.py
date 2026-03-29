"""
《Feudal Netwrks算法实现》
时间:2024.10.05
环境:CartPole
作者:不去幼儿园
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

# 超参数
GAMMA = 0.99
LEARNING_RATE = 0.001
MANAGER_UPDATE_FREQUENCY = 10  # 高层更新频率
WORKER_UPDATE_FREQUENCY = 1  # 低层更新频率
NUM_EPISODES = 50000
MIN_EPSILON = 0.1
EPSILON_DECAY = 0.995


# 高层（Manager）网络
class ManagerNetwork(nn.Module):
    def __init__(self, state_dim, goal_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, goal_dim)

    def forward(self, state):
        x = self.fc1(state)
        x = torch.relu(x)
        goal = self.fc2(x)
        return goal


# 低层（Worker）网络
class WorkerNetwork(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + goal_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, state, goal):
        x = torch.cat((state, goal), dim=-1)
        x = torch.relu(self.fc1(x))
        action_logits = self.fc2(x)
        return action_logits


# FeUdal Networks 智能体
class FeudalAgent:
    def __init__(self, state_dim, action_dim, goal_dim):
        self.manager_net = ManagerNetwork(state_dim, goal_dim)
        self.worker_net = WorkerNetwork(state_dim, goal_dim, action_dim)
        self.manager_optimizer = optim.Adam(
            self.manager_net.parameters(), lr=LEARNING_RATE
        )
        self.worker_optimizer = optim.Adam(
            self.worker_net.parameters(), lr=LEARNING_RATE
        )
        self.epsilon = 1.0

    def select_worker_action(self, state, goal, epsilon):
        if random.random() < epsilon:
            return random.choice([0, 1])  # CartPole 动作空间为 2（0 或 1）
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            goal = torch.FloatTensor(goal).unsqueeze(0)
            action_logits = self.worker_net(state, goal)
            action_probs = torch.softmax(action_logits, dim=-1)
            return torch.argmax(action_probs).item()

    # 修改后的 update_manager 方法
    def update_manager(self, state, next_state, goal, reward):
        state = (
            torch.FloatTensor(state).unsqueeze(0).requires_grad_(True)
        )  # 启用 requires_grad
        next_state = (
            torch.FloatTensor(next_state).unsqueeze(0).requires_grad_(True)
        )  # 启用 requires_grad
        goal = (
            torch.FloatTensor(goal).unsqueeze(0).requires_grad_(True)
        )  # 启用 requires_grad

        # 计算内在奖励
        deltaX = next_state - state  # (1, dimX)
        # intrinsic_reward = torch.dot(
        #     goal.squeeze(), next_state.squeeze() - state.squeeze()
        # )
        intrinsic_cost = torch.norm(deltaX - goal, p=2)

        # 反向传播损失
        loss = intrinsic_cost - reward
        self.manager_optimizer.zero_grad()
        loss.backward()
        self.manager_optimizer.step()

    def update_worker(self, state, goal, action, reward):
        state = torch.FloatTensor(state).unsqueeze(0)
        goal = torch.FloatTensor(goal).unsqueeze(0)
        action_logits = self.worker_net(state, goal)
        action_probs = torch.softmax(action_logits, dim=-1)
        log_action_probs = torch.log(action_probs)
        policy_loss = -log_action_probs[0, action] * reward
        self.worker_optimizer.zero_grad()
        policy_loss.backward()
        self.worker_optimizer.step()

    def train(self, env, num_episodes):
        goal_dim = self.manager_net.fc2.out_features

        train_results = []

        for episode in range(num_episodes):
            state, _ = env.reset()
            goal = (
                self.manager_net(torch.FloatTensor(state).unsqueeze(0))
                .detach()
                .numpy()
                .squeeze()
            )
            done = False
            episode_reward = 0
            steps = 0

            while not done:
                steps += 1
                action = self.select_worker_action(state, goal, self.epsilon)
                next_state, reward, done, _, __ = env.step(action)

                # 更新低层（Worker）
                self.update_worker(state, goal, action, reward)

                # 每隔 MANAGER_UPDATE_FREQUENCY 更新一次高层（Manager）
                if steps % MANAGER_UPDATE_FREQUENCY == 0:
                    new_goal = (
                        self.manager_net(torch.FloatTensor(next_state).unsqueeze(0))
                        .detach()
                        .numpy()
                        .squeeze()
                    )
                    self.update_manager(state, next_state, goal, reward)
                    goal = new_goal

                state = next_state
                episode_reward += reward

            self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)
            print(f"Episode {episode + 1}: Total Reward: {episode_reward}")
            train_results.append(episode_reward)
        return train_results


def main():
    # 创建 CartPole 环境并训练智能体
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    goal_dim = 4  # 设置一个目标维度
    agent = FeudalAgent(state_dim, action_dim, goal_dim)
    rst = agent.train(env, NUM_EPISODES)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(rst)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    plt.show()


if __name__ == "__main__":
    main()
