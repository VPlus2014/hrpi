"""
Dreamerv3 World Model 使用示例

展示如何:
1. 从NPSyncVecEnv初始化世界模型
2. 收集训练数据
3. 训练世界模型
4. 使用世界模型进行想象
"""
import torch
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

# 导入world model模块
from codes.world_model import (
    WorldModel,
    Dreamer,
    WorldModelCollector,
    RSSMState,
)


def create_world_model_from_env(env, config=None):
    """
    从NPSyncVecEnv创建世界模型

    Args:
        env: NPSyncVecEnv实例 (如NavHeadingEnv)
        config: 配置字典

    Returns:
        WorldModel实例
    """
    if config is None:
        config = {}

    # 获取环境信息
    obs_shape = env.single_observation_space.shape
    action_dim = env.single_action_space.shape[0]

    # 默认配置
    embed_dim = config.get("embed_dim", 256)
    hidden_dim = config.get("hidden_dim", 256)
    latent_dim = config.get("latent_dim", 32)
    reward_hidden_dims = config.get("reward_hidden_dims", (256, 256))
    use_cnn = config.get("use_cnn", False)

    world_model = WorldModel(
        obs_shape=obs_shape,
        action_dim=action_dim,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        reward_hidden_dims=reward_hidden_dims,
        use_cnn=use_cnn,
    )

    return world_model


def train_world_model_example():
    """世界模型训练示例"""
    # 1. 创建环境
    from codes.envs_np import NavHeadingEnv

    env = NavHeadingEnv(
        num_envs=4,
        agent_step_size_ms=100,
        sim_step_size_ms=20,
    )

    # 2. 创建世界模型
    world_model = create_world_model_from_env(env)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world_model = world_model.to(device)

    print(f"World Model 参数量: {sum(p.numel() for p in world_model.parameters())}")

    # 3. 创建数据收集器
    collector = WorldModelCollector(
        env=env,
        device=device,
        normalize_obs=True,
    )

    # 4. 训练循环
    optimizer = torch.optim.Adam(world_model.parameters(), lr=3e-4)

    num_epochs = 100
    rollout_steps = 64
    batch_size = 16

    for epoch in range(num_epochs):
        # 收集数据 (使用随机策略)
        data = collector.collect_rollout(num_steps=rollout_steps)

        # 训练世界模型
        world_model.train()
        T, B = data["rewards"].shape

        # 随机选择起始时间步
        start_idx = torch.randint(0, T - 1, (batch_size,))
        batch_obs = []
        batch_action = []
        batch_reward = []
        batch_continue = []

        for i in range(batch_size):
            t = start_idx[i].item()
            batch_obs.append(data["observations"][t])
            batch_action.append(data["actions"][t])
            batch_reward.append(data["rewards"][t])
            batch_continue.append(data["dones_float"][t])

        obs_batch = torch.stack(batch_obs, dim=0).to(device)
        action_batch = torch.stack(batch_action, dim=0).to(device)
        reward_batch = torch.stack(batch_reward, dim=0).to(device)
        continue_batch = torch.stack(batch_continue, dim=0).to(device)

        # 计算损失
        total_loss, loss_dict = world_model.compute_loss(
            obs=obs_batch,
            action=action_batch,
            reward=reward_batch,
            continue_flag=1.0 - continue_batch,  # invert: 1=continue, 0=done
        )

        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(world_model.parameters(), 100.0)
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: {loss_dict}")

    print("训练完成!")


def dreamer_agent_example():
    """Dreamer智能体示例"""
    # 1. 创建环境
    from codes.envs_np import NavHeadingEnv

    env = NavHeadingEnv(
        num_envs=4,
        agent_step_size_ms=100,
        sim_step_size_ms=20,
    )

    # 2. 创建世界模型
    world_model = create_world_model_from_env(env)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world_model = world_model.to(device)

    # 3. 创建Dreamer智能体
    action_dim = env.single_action_space.shape[0]
    action_scale = (env.single_action_space.high - env.single_action_space.low) / 2

    dreamer = Dreamer(
        world_model=world_model,
        action_dim=action_dim,
        actor_hidden_dims=(256, 256),
        critic_hidden_dims=(256, 256),
        gamma=0.99,
        lam=0.95,
        horizon=15,
        action_scale=action_scale,
    ).to(device)

    # 4. 创建收集器
    collector = WorldModelCollector(env=env, device=device)

    # 5. 训练Dreamer
    world_optimizer = torch.optim.Adam(world_model.parameters(), lr=3e-4)
    actor_optimizer = torch.optim.Adam(dreamer.actor.parameters(), lr=1e-4)
    critic_optimizer = torch.optim.Adam(dreamer.critic.parameters(), lr=1e-4)

    num_iterations = 1000

    for i in range(num_iterations):
        # 收集真实数据
        data = collector.collect_rollout(num_steps=64)

        # 训练世界模型
        world_model.train()
        T, B = data["rewards"].shape

        # 随机批次
        start_idx = torch.randint(0, T - 1, (16,))
        obs_batch = torch.stack([data["observations"][t] for t in start_idx]).to(device)
        action_batch = torch.stack([data["actions"][t] for t in start_idx]).to(device)
        reward_batch = torch.stack([data["rewards"][t] for t in start_idx]).to(device)
        continue_batch = torch.stack([data["dones_float"][t] for t in start_idx]).to(device)

        # 世界模型损失
        total_loss, _ = world_model.compute_loss(
            obs=obs_batch,
            action=action_batch,
            reward=reward_batch,
            continue_flag=1.0 - continue_batch,
        )

        world_optimizer.zero_grad()
        total_loss.backward()
        world_optimizer.step()

        # 想象 trajectories 并训练 actor-critic
        world_model.eval()
        with torch.no_grad():
            # 初始化状态
            init_state = world_model.initial_state(batch_size=B, device=device)

            # 想象未来
            imagined_actions, imagined_rewards, imagined_continues = dreamer.imagine(
                init_state, num_steps=15
            )

            # 收集想象状态
            imagined_states = [init_state]
            curr_state = init_state
            for t in range(15):
                action = imagined_actions[t]
                curr_state = world_model.rssm.imagine(curr_state, action)
                imagined_states.append(curr_state)

            # 更新 actor-critic
            total_loss, loss_dict = dreamer.update_actor_critic(
                imagined_rewards,
                1.0 - imagined_continues,
                imagined_states,
            )

            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            total_loss.backward()
            actor_optimizer.step()
            critic_optimizer.step()

        if i % 100 == 0:
            print(f"Iteration {i}: world_loss={total_loss.item():.4f}")


def simple_inference_example():
    """简单的推理示例 - 仅使用世界模型"""
    from codes.envs_np import NavHeadingEnv

    bsz = 2
    horizon = 5

    # 创建环境
    env = NavHeadingEnv(num_envs=bsz, agent_step_size_ms=100)

    # 创建世界模型
    world_model = create_world_model_from_env(env)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world_model = world_model.to(device)
    world_model.eval()

    # 初始化状态
    state = world_model.initial_state(batch_size=bsz, device=device)

    # 收集一些数据
    collector = WorldModelCollector(env=env, device=device)
    data = collector.collect_rollout(num_steps=10)

    # 使用世界模型进行观察
    obs = data["observations"].to(device)  # (T, B, *obs_shape)
    action = data["actions"].to(device)     # (T, B, action_dim)

    with torch.no_grad():
        # 观察
        states, outputs = world_model.observe(obs, action, state)

        print(f"观察序列长度: {states.h.shape[0]}")
        print(f"隐状态形状: {states.h.shape}")
        print(f"潜在变量形状: {states.z.shape}")
        print(f"奖励预测形状: {outputs.reward_pred.shape}")

        # 想象未来
        imagined_actions = torch.randn(horizon, bsz, env.single_action_space.shape[0], device=device)
        imagined_states, imagined_outputs = world_model.imagine(state, imagined_actions)

        print(f"\n想象序列长度: {imagined_states.h.shape[0]}")
        print(f"想象奖励预测: {imagined_outputs.reward_pred}")


if __name__ == "__main__":
    print("=== 简单推理示例 ===")
    simple_inference_example()
