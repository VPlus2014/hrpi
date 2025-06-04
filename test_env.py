from pathlib import Path
import time
from typing import cast
import numpy as np
import torch
import gymnasium as gym
from tqdm import tqdm
from contextlib import ContextDecorator


class ConextTimer(ContextDecorator):
    def __init__(self, name: str):
        self.name = name
        self.t = 0
        self.dt = 0
        self._lv = 0

    def reset(self):
        self.t = 0
        self.dt = 0

    def __enter__(self):
        self.push()

    def __exit__(self, *exc):
        self.pop()

    def push(self):
        self._lv += 1
        if self._lv == 1:
            self._t0 = time.time()

    def pop(self):
        if self._lv > 0:
            self._lv -= 1
            if self._lv == 0:
                self.dt = dt = time.time() - self._t0
                self.t += dt


def init_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    # torch.set_num_interop_threads(1)
    print(f"Seed initialized to {seed}")


def as_np(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    else:
        return x


def main():
    use_cuda = False
    dv = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    dtp = torch.float32
    nenvs = 1000
    total_frames = 1000000
    bufsz = 2 * nenvs
    buftmax_step = 100
    global_max_steps = total_frames // nenvs
    env_desc_ms = 50
    env_sim_dt_ms = env_desc_ms
    env_max_steps = 1000
    render_mode = [
        None,
        "tacview",
    ][-1]
    render_dir = Path(__file__).parent / "tmp"

    from environments.navigation import NavigationEnv

    envcls = NavigationEnv
    for out_numpy in [True]:
        init_seed(0)

        env = envcls(
            agent_step_size_ms=env_desc_ms,
            sim_step_size_ms=env_sim_dt_ms,
            max_sim_ms=env_sim_dt_ms * env_max_steps,
            waypoints_total_num=10,
            waypoints_visible_num=1,
            position_min_limit=[-5000, -5000, -10000],
            position_max_limit=[5000, 5000, 0],
            render_mode=render_mode,
            render_dir=render_dir,
            num_envs=nenvs,
            device=dv,
            out_torch=bool(out_numpy),
        )

        from replay_buffer.trajbuffer import RETrajReplayBuffer

        buffer = RETrajReplayBuffer(
            max_steps=buftmax_step,
            obs_shape=env.observation_space.shape,
            act_shape=env.action_space.shape,
            float_dtype=np.float32,
            max_trajs=bufsz,
            num_envs=nenvs,
        )
        qbar = tqdm(range(global_max_steps))
        tmr_env = ConextTimer("env")
        tmr_infer = ConextTimer("infer")
        tmr_buffer = ConextTimer("buffer")

        _t0 = time.time()
        _echo_k = 0
        _echo_k0 = 0

        with tmr_env:
            obs, info = env.reset()

        with tmr_buffer:
            obs = as_np(obs)
        obss = [obs]

        for itr in qbar:
            with tmr_infer:
                action = env.action_space.sample()
                action = torch.asarray(action, device=dv, dtype=dtp)

            with tmr_env:
                obs_next, rew, term, trunc, info = env.step(action)
            #
            with tmr_buffer:
                obs_next = as_np(obs_next)
                act = as_np(action)
                rew = as_np(rew)
                term = as_np(term)
                trunc = as_np(trunc)
                logpa = act * 0
                buffer.add(obs, act, obs_next, rew, term, trunc, act_log_prob=logpa)

                obs = obs_next
                if obss is not None:
                    obss.append(obs_next)

            if obss is not None:
                if buffer._ptrE2N[0] == 0:
                    obss_ = np.stack(obss, axis=0)  # (T,B,...)
                    obss_ = obss_[:, 0, 0]
                    obscmpr = buffer._obs[: obss_.shape[0], 0, 0]
                    strg = np.not_equal(obss_, obscmpr)
                    if np.any(strg):
                        whe = np.where(strg)[0]
                        print(obss_, "---", obscmpr, sep="\n")
                        print(whe)
                        whe
                else:
                    obss.clear()
                    obss = None

            _echo_k = int((time.time() - _t0) / 1)
            if _echo_k > _echo_k0:
                _echo_k0 = _echo_k
                ts = np.asarray([tmr_env.t, tmr_infer.t, tmr_buffer.t])
                tsum = max(ts.sum(), 1e-6)
                tr = ts / tsum
                fps = nenvs * (itr + 1) / tsum
                spb = tsum / (itr + 1)

                qbar.set_postfix(
                    env=f"{tr[0]:.0%}",
                    infer=f"{tr[1]:.0%}",
                    buffer=f"{tr[2]:.0%}",
                    fps=f"{fps:.0g}",
                    mspb=f"{int(spb*1e3)}",
                )

        dt = time.time() - _t0
        print(f"Time elapsed: {dt:.2f}s, FPS: {nenvs*global_max_steps/dt:.2f}")


if __name__ == "__main__":
    main()
