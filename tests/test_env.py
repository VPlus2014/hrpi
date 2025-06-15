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

from envs_np.utils.log_ext import LogConfig
from tools import init_seed, as_np, as_tsr, ConextTimer


def main():
    seed = 10086
    use_cuda = False
    dv = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
    dtp = torch.float32
    nenvs = 1
    total_frames = int(1e3)
    bufsz = 2 * nenvs
    buftmax_step = 100
    global_max_steps = total_frames // nenvs
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

    from envs_np.nav_heading import NavHeadingEnv as TestEnv

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
            render_dir=render_dir,
            device=dv,
            easy_mode=True,
            debug=True,
            logconfig=LogConfig(
                __name__, level=logging.DEBUG, file_path=str(run_dir / "env.log")
            ),
        )
        env.seeding(seed)

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

                obs = obs_next
                if obss is not None:
                    obss.append(obs_next.copy())

                if anydone:
                    obs[msk, :] = obs_

            with tmr_env:
                pass

            if obss is not None:
                if buffer._ptrE2N[0] == 0:
                    obss_ = np.stack(obss, axis=0)  # (T,B,...)
                    obss_ = obss_[:, 0, 0]
                    obscmpr = buffer._obs[: obss_.shape[0], 0, 0]
                    strg = np.not_equal(obss_, obscmpr)
                    if np.any(strg):
                        msk = np.where(strg)[0]
                        print(obss_, "---", obscmpr, sep="\n")
                        print(msk)
                        msk
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
