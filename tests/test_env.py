from copy import deepcopy
from datetime import datetime
import logging
from pathlib import Path
import time
from typing import cast
import gymnasium
import numpy as np
import torch
from tqdm import tqdm
import gymnasium


def _setup():
    import sys
    from pathlib import Path

    __FILE = Path(__file__)
    ROOT = __FILE.parents[1]  # /../..
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    return ROOT


ROOT = _setup()


from codes.utils import log_ext
from util_tools import init_seed, as_np, as_tsr, ConextTimer


def main():

    # log_ext.reset_root_logger(logging.DEBUG)
    seed = 10086
    use_cuda = False
    dv = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
    dtp = torch.float32
    nenvs = 4000
    total_frames = int(1e6)
    bufsz = 2 * nenvs + 1
    buftmax_step = 100
    global_max_steps = max(1, total_frames // nenvs)
    env_sim_dt_ms = 10
    env_desc_ms = 5 * env_sim_dt_ms
    env_desc_max_steps = 1000
    render_mode = [
        None,
        "tacview_remote",
        "tacview_local",
    ][-1]
    run_dir = ROOT / "tmp" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    render_dir = run_dir / "acmi"

    from codes.envs_np.nav_heading import NavHeadingEnv as TestEnv

    for out_numpy in [True]:
        init_seed(0)

        env = TestEnv(
            num_envs=nenvs,
            agent_step_size_ms=env_desc_ms,
            sim_step_size_ms=env_sim_dt_ms,
            max_sim_ms=env_desc_ms * env_desc_max_steps,
            waypoints_total_num=1,
            waypoints_visible_num=1,
            pos_e_nvec=[100, 100, 100],
            render_mode=render_mode,
            render_dir=str(render_dir),
            device=dv,
            easy_mode=True,
            debug=True,
            logconfig=log_ext.LogConfig(
                __name__, level=logging.DEBUG, file_path=str(run_dir / "env.log")
            ),
        )
        env.seeding(seed)

        from codes.envs_np.wrappers.disc_action import LinspaceActionWrapper

        from codes.agents.replay_buffer.np_trajbuffer import RETrajReplayBuffer

        buffer = RETrajReplayBuffer(
            max_steps=buftmax_step,
            obs_shape=env.observation_space.shape,
            act_shape=env.action_space.shape,
            float_dtype=np.float32,
            max_trajs=bufsz,
            size_stack_in=nenvs,
            logger=log_ext.reset_logger(
                "test_buffer",
                level=logging.DEBUG,
                file_path=str(run_dir / "buffer.log"),
            ),
            debug=True,
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

        with tmr_infer:
            obs = as_np(obs)

        for itr in qbar:
            with tmr_infer:
                action = env.action_space.sample()
                action = action.reshape(1, -1)
                action = np.repeat(action, nenvs, axis=0)
                # action = torch.asarray(action, device=dv, dtype=dtp)

            with tmr_env:
                obs_next, rew, term, trunc, info = env.step(action)
                done = term | trunc
                anydone = done.any()
                if anydone:
                    msk = done.ravel()
                    obs_, _ = env.reset(msk)

            #
            with tmr_buffer:
                obs_next = as_np(obs_next)
                act = as_np(action)
                rew = as_np(rew)
                term = as_np(term)
                trunc = as_np(trunc)
                logpa = act * 0
                buffer.add(
                    obs, act, obs_next, rew, term, trunc, act_log_prob=logpa, done=done
                )

            with tmr_infer:
                obs = deepcopy(obs_next)
                if anydone:
                    obs[msk, :] = obs_

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
