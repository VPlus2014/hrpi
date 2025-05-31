import torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from environments import NavigationEnv, EvasionEnv
from agents import PPOContinuous


def main():
    writer = SummaryWriter()
    # train_env = NavigationEnv(
    #     agent_step_size_ms = 50, 
    #     sim_step_size_ms = 50, 
    #     navigation_points_total_num = 10, 
    #     navigation_points_visible_num = 1,
    #     position_min_limit = [-5000, -5000, -10000],
    #     position_max_limit = [5000, 5000, 0],
    #     render_mode = "tacview",
    #     render_dir = Path.cwd() / "results",
    #     num_envs = 2000, 
    #     device = torch.device("cuda"),
    #     mode = "pytorch"
    # )
    train_env = EvasionEnv(
        agent_step_size_ms = 100, 
        sim_step_size_ms = 10, 
        position_min_limit = [-10000, -10000, -10000],
        position_max_limit = [10000, 10000, 0],
        writer = writer,
        render_mode = "tacview",
        render_dir = Path.cwd() / "results_1",
        num_envs = 1000, 
        device = torch.device("cuda"),
        mode = "pytorch"
    )
    
    batch_size = 500
    
    agent = PPOContinuous(
        name = "planner",
        observation_space = train_env.observation_space,
        action_space = train_env.action_space,
        batch_size = batch_size, 
        mini_batch_size = 100,
        lr_a = 1e-4,
        actor_hidden_sizes = [128, 128, 128, 128, 128],
        lr_c = 1e-4,
        critic_hidden_sizes = [128, 128, 128, 128, 128],
        gamma = 0.99,
        gae_lambda = 0.95,
        epsilon = 0.2,
        policy_entropy_coef = 0.01,
        use_grad_clip = False,
        use_adv_norm = False,
        repeat = 10,
        adam_eps = 1e-5,
        num_envs = train_env.num_envs, 
        writer = writer,
        device = train_env.device
    )

    obs, _ = train_env.reset()

    # train
    max_train_episodes = int(1e8)
    progress_bar = tqdm(total=max_train_episodes, desc="Training Progress")
    global_step = 0
    pre_udpate_n = progress_bar.n
    while progress_bar.n < max_train_episodes:
        # 更新智能体
        if pre_udpate_n != progress_bar.n and global_step % (agent.replay_buffer.max_size) == 0:
            agent.update(progress_bar.n)
        
        # 存储智能体
        if progress_bar.n % 10000 == 0:
            torch.save(agent.actor, "actor.pt")
            torch.save(agent.actor.state_dict(), "actor.pth")

        # obs_norm = normalize(obs, low, high)
        act, act_log_prob = agent.choose_action(obs)
        
        obs_next, rew, terminated, truncated, _ = train_env.step(act, progress_bar.n)
        train_env.render()
        agent.post_act(obs_next, act, act_log_prob, rew, terminated, truncated, progress_bar.n)
        
        obs = obs_next
        done = torch.logical_or(truncated, terminated)
        progress_bar.update(done.sum().item())

        global_step += 1


if __name__ == "__main__":
    main()
    
