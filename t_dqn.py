from __future__ import annotations
from pathlib import Path
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import swanlab


# 定义Q网络
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.fc(x)


# DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim, seed=None):
        self.q_net = QNetwork(state_dim, action_dim)  # 当前网络
        self.target_net = QNetwork(state_dim, action_dim)  # 目标网络
        self.target_net.load_state_dict(
            self.q_net.state_dict()
        )  # 将目标网络和当前网络初始化一致，避免网络不一致导致的训练波动
        self.best_net = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)
        self.replay_buffer = deque(maxlen=10000)  # 经验回放缓冲区
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 0.1
        self.update_target_freq = 100  # 目标网络更新频率
        self.step_count = 0
        self.best_reward = 0
        self.best_avg_reward = 0
        self.eval_episodes = 5  # 评估时的episode数量
        self.loss_func = nn.MSELoss()  # 损失函数

        self.rng = np.random.default_rng(seed)

    def choose_action(self, state):
        rng = self.rng
        if rng.random() < self.epsilon:
            return rng.integers(0, 2)  # CartPole有2个动作（左/右）
        else:
            state_tensor = torch.FloatTensor(state)
            q_values: torch.Tensor = self.q_net(state_tensor)
            return q_values.detach().cpu().argmax().numpy()

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train(self, max_iters=5):
        n = len(self.replay_buffer)
        if n < self.batch_size:
            return
        rng = self.rng
        rb = self.replay_buffer

        # 从缓冲区随机采样
        idxs = rng.choice(n, self.batch_size, replace=False)
        batch = [rb[i] for i in idxs]
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.asarray(states))  # (B,dimX)
        actions = torch.LongTensor(np.asarray(actions)).unsqueeze(-1)  # (B,1)
        next_states = torch.FloatTensor(np.asarray(next_states))  # (B,dimX)
        rewards = torch.FloatTensor(np.asarray(rewards))  # (B,)
        dones = torch.FloatTensor(np.asarray(dones))  # (B,)

        # 计算目标Q值（使用目标网络）
        with torch.no_grad():
            next_q: torch.Tensor = self.target_net(next_states)
            next_q = next_q.max(-1)[0]  # (B,)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        for _itr in range(max_iters):
            # 计算当前Q值
            current_q: torch.Tensor = self.q_net(states)
            current_q = current_q.gather(-1, actions).squeeze()

            # 计算损失并更新网络
            loss = self.loss_func(current_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # 定期更新目标网络
        self.step_count += 1
        if self.step_count % self.update_target_freq == 0:
            # 使用深拷贝更新目标网络参数
            self.target_net.load_state_dict(
                {k: v.clone() for k, v in self.q_net.state_dict().items()}
            )

    def save_model(self, path="./output/best_model.pth"):
        po = Path(path).resolve()
        if not po.parent.exists():
            po.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.q_net.state_dict(), po)
        print(f"Model saved to {po}")

    def evaluate(self, env):
        """评估当前模型的性能"""
        original_epsilon = self.epsilon
        self.epsilon = 0  # 关闭探索
        total_rewards = []

        for _ in range(self.eval_episodes):
            state = env.reset()[0]
            episode_reward = 0
            while True:
                action = self.choose_action(state)
                next_state, reward, done, _, _ = env.step(action)
                episode_reward += reward
                state = next_state
                if done or episode_reward > 2e4:
                    break
            total_rewards.append(episode_reward)

        self.epsilon = original_epsilon  # 恢复探索
        return np.mean(total_rewards)


def main():
    # 设置随机数种子
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    # 训练过程
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)

    # 初始化SwanLab日志记录器
    swanlab.init(
        project="RL-All-In-One",
        experiment_name="DQN-CartPole-v1",
        config={
            "state_dim": state_dim,
            "action_dim": action_dim,
            "batch_size": agent.batch_size,
            "gamma": agent.gamma,
            "epsilon": agent.epsilon,
            "update_target_freq": agent.update_target_freq,
            "replay_buffer_size": agent.replay_buffer.maxlen,
            "learning_rate": agent.optimizer.param_groups[0]["lr"],
            "episode": 200,
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay": 0.995,
        },
        description="增加了初始化目标网络和当前网络一致，避免网络不一致导致的训练波动",
    )

    # ========== 训练阶段 ==========

    agent.epsilon = swanlab.config["epsilon_start"]

    for episode in range(swanlab.config["episode"]):
        seed = SEED if episode == 0 else None  # 随机种子
        state = env.reset(seed=seed)[0]
        total_reward = 0.0

        while True:
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.store_experience(state, action, reward, next_state, done)
            agent.train()

            total_reward += reward
            state = next_state
            if done or total_reward > 2e4:
                break

        # epsilon是探索系数，随着每一轮训练，epsilon 逐渐减小
        agent.epsilon = max(
            swanlab.config["epsilon_end"],
            agent.epsilon * swanlab.config["epsilon_decay"],
        )

        # 每10个episode评估一次模型
        if episode % 10 == 0:
            eval_env = gym.make("CartPole-v1")
            avg_reward = agent.evaluate(eval_env)
            eval_env.close()

            if avg_reward > agent.best_avg_reward:
                agent.best_avg_reward = avg_reward
                # 深拷贝当前最优模型的参数
                agent.best_net.load_state_dict(
                    {k: v.clone() for k, v in agent.q_net.state_dict().items()}
                )
                agent.save_model(path=f"./output/best_model.pth")
                print(f"New best model saved with average reward: {avg_reward}")

        print(
            f"Episode: {episode}, Train Reward: {total_reward}, Best Eval Avg Reward: {agent.best_avg_reward}"
        )

        swanlab.log(
            {
                "train/reward": total_reward,
                "eval/best_avg_reward": agent.best_avg_reward,
                "train/epsilon": agent.epsilon,
            },
            step=episode,
        )

    # 测试并录制视频
    agent.epsilon = 0  # 关闭探索策略
    test_env = gym.make("CartPole-v1", render_mode="rgb_array")
    test_env = RecordVideo(
        test_env, "./dqn_videos", episode_trigger=lambda x: True
    )  # 保存所有测试回合
    agent.q_net.load_state_dict(agent.best_net.state_dict())  # 使用最佳模型

    for episode in range(3):  # 录制3个测试回合
        seed = SEED if episode == 0 else None  # 随机种子
        state = test_env.reset(seed=seed)[0]
        total_reward = 0
        steps = 0

        while True:
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = test_env.step(action)
            total_reward += reward
            state = next_state
            steps += 1

            # 限制每个episode最多1500步,约30秒,防止录制时间过长
            if done or steps >= 1500:
                break

        print(f"Test Episode: {episode}, Reward: {total_reward}")

    test_env.close()


if __name__ == "__main__":
    main()
