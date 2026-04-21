import embodied.jax.nets as nn
import ninjax as nj


class Encoder(nj.Module):

  units: int = 512  # Reduced
  norm: str = 'rms'
  act: str = 'gelu'
  layers: int = 2  # Reduced

  def __init__(self, obs_space, **kw):
    # Assume vectorized observations
    self.obs_space = obs_space
    self.kw = kw

  def __call__(self, obs):
    x = nn.DictConcat(self.obs_space, 1)(obs)
    for i in range(self.layers):
      x = self.sub(f'enc{i}', nn.Linear, self.units, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'enc{i}norm', nn.Norm, self.norm)(x))
    return x