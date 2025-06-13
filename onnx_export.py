import onnx
import torch
import torch.nn as nn
from pathlib import Path
from environments_th import EvasionEnv
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    writer = SummaryWriter()
    env = EvasionEnv(
        agent_step_size_ms=50,
        sim_step_size_ms=50,
        position_min_limit=[-10000, -10000, -10000],
        position_max_limit=[10000, 10000, 0],
        writer=writer,
        render_mode="tacview",
        render_dir=Path.cwd() / "results",
        num_envs=1,
        device=torch.device("cpu"),
        mode="pytorch",
    )
    obs = env.observation_space.sample()
    obs = torch.from_numpy(obs).to(device=torch.device("cuda"))
    model = torch.load("actor.pt", weights_only=False)
    model = model
    model.eval()
    print(model)
    torch.onnx.export(model, obs, "actor.onnx")
