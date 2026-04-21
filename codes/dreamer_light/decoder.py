import embodied.jax.nets as nn
import ninjax as nj


class Decoder(nj.Module):

  units: int = 512
  norm: str = 'rms'
  act: str = 'gelu'
  layers: int = 2

  def __init__(self, obs_space, **kw):
    self.obs_space = obs_space
    self.kw = kw

  def __call__(self, feat):
    x = feat
    for i in range(self.layers):
      x = self.sub(f'dec{i}', nn.Linear, self.units, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'dec{i}norm', nn.Norm, self.norm)(x))
    # Output reconstruction for vector obs
    out = {}
    for key, space in self.obs_space.items():
      out[key] = self.sub(f'out_{key}', nn.Linear, space.size, **self.kw)(x)
    return out