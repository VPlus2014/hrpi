#!/usr/bin/env python3

import embodied
import jax
import jax.numpy as jnp
import numpy as np

from .model import DreamerLight


def train():
    # Dummy spaces for vectorized obs
    obs_space = {
        'vector': embodied.Space(np.float32, (10,)),  # Example vector obs
        'reward': embodied.Space(np.float32, ()),
        'is_last': embodied.Space(bool, ()),
    }
    act_space = {
        'action': embodied.Space(np.float32, (4,)),  # Example continuous action
    }

    config = embodied.Config.load('configs.yaml')
    model = DreamerLight(obs_space, act_space, config)

    # Initialize
    carry = model.init_train(config.batch_size)

    # Dummy batch
    batch_obs = {
        'vector': jnp.ones((config.batch_size, config.batch_length, 10)),
        'reward': jnp.zeros((config.batch_size, config.batch_length)),
        'is_last': jnp.zeros((config.batch_size, config.batch_length), dtype=bool),
    }
    batch_act = {
        'action': jnp.zeros((config.batch_size, config.batch_length, 4)),
    }
    reset = jnp.zeros((config.batch_size, config.batch_length), dtype=bool)

    # Train step
    carry, entries, losses, metrics = model.train(carry, batch_obs, batch_act, reset)

    print("Training step completed!")
    print("Losses:", {k: float(v.mean()) for k, v in losses.items()})


if __name__ == '__main__':
    train()