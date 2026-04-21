import embodied.jax.nets as nn
import ninjax as nj

from .rssm import RSSM
from .encoder import Encoder
from .decoder import Decoder
from .reward_predictor import RewardPredictor
from .continuation_predictor import ContinuationPredictor
from .policy_value import Policy, Value


class DreamerLight(nj.Module):

  def __init__(self, obs_space, act_space, config):
    self.obs_space = obs_space
    self.act_space = act_space
    self.config = config

    self.rssm = RSSM(act_space, **config.dyn)
    self.encoder = Encoder(obs_space, **config.enc)
    self.decoder = Decoder(obs_space, **config.dec)
    self.reward = RewardPredictor(**config.rewhead)
    self.continuation = ContinuationPredictor(**config.conhead)
    self.policy = Policy(act_space, **config.policy)
    self.value = Value(**config.value)

  def init_train(self, batch_size):
    carry = self.rssm.initial(batch_size)
    return carry

  def train(self, carry, obs, action, reset, training=True):
    tokens = self.encoder(obs)
    carry, entries, losses, feat, metrics = self.rssm.loss(
        carry, tokens, action, reset, training)

    # Reconstruction loss
    recon = self.decoder(feat['deter'])
    recon_loss = {}
    for key in self.obs_space:
      recon_loss[key] = -recon[key].log_prob(obs[key])

    # Reward and continuation losses
    rew_pred = self.reward(feat['deter'])
    rew_loss = -rew_pred.log_prob(obs['reward'])

    con_pred = self.continuation(feat['deter'])
    con_loss = -con_pred.log_prob(obs['is_last'])

    losses.update({
        'recon': sum(recon_loss.values()),
        'rew': rew_loss,
        'con': con_loss,
    })

    metrics.update({
        'rew_mse': jnp.mean((rew_pred.mean - obs['reward']) ** 2),
    })

    return carry, entries, losses, metrics

  def policy(self, carry, obs, training=True):
    tokens = self.encoder(obs)
    carry, _, feat = self.rssm.observe(carry, tokens, {}, False, training, single=True)
    action = self.policy(feat['deter'])
    return carry, action

  def imagine(self, carry, length, training=True):
    def policy_fn(feat):
      return self.policy(feat)
    carry, feat, action = self.rssm.imagine(carry, policy_fn, length, training)
    rew_pred = self.reward(feat['deter'])
    val_pred = self.value(feat['deter'])
    return carry, feat, action, rew_pred, val_pred