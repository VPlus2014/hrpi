import numpy as np
import torch
from pathlib import Path
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from environments import NavigationEnv
from agents import HPPOContinuous
from environments.utils.tacview_render import ObjectState, AircraftAttr, WaypointAttr


as_tsr = torch.asarray


def init_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    print(f"Seed initialized to {seed}")


def main():
    np_float = np.float32
    tsr_float = torch.float32
    init_seed(10086)

    train_env = NavigationEnv(
        agent_step_size_ms=50,
        sim_step_size_ms=50,
        waypoints_total_num=10,
        waypoints_visible_num=1,
        position_min_limit=[-5000, -5000, -10000],
        position_max_limit=[5000, 5000, 0],
        render_mode="tacview",
        render_dir=Path.cwd() / "results_h",
        num_envs=1000,
        device=torch.device("cpu"),
        dtype=tsr_float,
        np_float=np_float,
    )
    writer = SummaryWriter()

    batch_size = 50
    temporal_extension = 20
    agent = HPPOContinuous(
        env_observation_space_dict=train_env._observation_space,
        env_action_space_dict=train_env._action_space,
        h_batch_size=batch_size,
        h_mini_batch_size=50,
        h_lr_a=1e-4,
        h_actor_hidden_sizes=[128, 128, 128],
        h_lr_c=1e-4,
        h_critic_hidden_sizes=[128, 128, 128, 128],
        h_gamma=0.99,
        h_gae_lambda=0.95,
        h_epsilon=0.2,
        h_policy_entropy_coef=0.01,
        h_use_grad_clip=False,
        h_use_adv_norm=False,
        h_repeat=10,
        h_adam_eps=1e-5,
        l_batch_size=batch_size * temporal_extension,
        l_mini_batch_size=50,
        l_lr_a=1e-4,
        l_actor_hidden_sizes=[128, 128, 128],
        l_lr_c=1e-4,
        l_critic_hidden_sizes=[128, 128, 128, 128],
        l_gamma=0.99,
        l_gae_lambda=0.95,
        l_epsilon=0.2,
        l_policy_entropy_coef=0.01,
        l_use_grad_clip=False,
        l_use_adv_norm=False,
        l_repeat=10,
        l_adam_eps=1e-5,
        temporal_extension=temporal_extension,
        num_envs=train_env.num_envs,
        writer=writer,
        device=train_env.device,
        dtype=tsr_float,
    )

    obs, _ = train_env.reset()

    # train
    max_train_episodes = int(1e6)
    progress_bar = tqdm(total=max_train_episodes, desc="Training Progress")
    global_step = 0
    while progress_bar.n < max_train_episodes:
        # 更新智能体
        if global_step and global_step % (agent.tracker.replay_buffer.max_size) == 0:
            agent.update(global_step)

        # obs_norm = normalize(obs, low, high)
        act, act_log_prob = agent.choose_action(obs)
        # render
        aircraft_state = ObjectState(
            sim_time_s=0.001 * train_env.sim_time_ms[0].item(),
            name="hppo",
            attr=AircraftAttr(
                Color="Blue",
                CallSign="PPOAgent",
            ),
            pos_ned=agent.expected_aircraft_position_g[0, ...].detach().cpu(),
        )
        train_env.render_object_state(aircraft_state)
        # render
        # aircraft_state = ObjectState(
        #     sim_time_s = 0.001*train_env.sim_time_ms[0].item(),
        #     name = "hermite",
        #     attr = AircraftAttr(
        #         Color = "Blue"
        #     ),
        #     pos_ned = agent.l_obs_dict["expected_aircraft_position_g"][0, ...].detach().cpu()
        # )
        # train_env.render_object_state(aircraft_state)

        obs_next, rew, terminated, truncated, _ = train_env.step(act)
        #
        obs_next = as_tsr(obs_next, dtype=tsr_float, device=train_env.device)
        rew = as_tsr(rew, dtype=tsr_float, device=train_env.device)
        terminated = as_tsr(terminated, dtype=torch.bool, device=train_env.device)
        truncated = as_tsr(truncated, dtype=torch.bool, device=train_env.device)
        #
        agent.post_act(
            obs, obs_next, act, act_log_prob, rew, terminated, truncated, global_step
        )

        obs = obs_next
        done = truncated | terminated
        progress_bar.update(done.sum().item())

        global_step += 1


if __name__ == "__main__":
    main()
