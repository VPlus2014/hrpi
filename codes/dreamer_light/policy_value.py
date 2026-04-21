import embodied.jax.nets as nn
import embodied.jax.outs as outs
import ninjax as nj


class Policy(nj.Module):

  units: int = 512
  norm: str = 'rms'
  act: str = 'gelu'
  layers: int = 2
  minstd: float = 0.1
  maxstd: float = 1.0
  outscale: float = 0.01
  unimix: float = 0.01

  def __init__(self, act_space, **kw):
    self.act_space = act_space
    self.kw = kw

  def __call__(self, feat):
    x = feat
    for i in range(self.layers):
      x = self.sub(f'pol{i}', nn.Linear, self.units, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'pol{i}norm', nn.Norm, self.norm)(x))
    kw = dict(**self.kw, outscale=self.outscale)
    out = {}
    for key, space in self.act_space.items():
      if space.discrete:
        out[key] = self.sub(f'pol_{key}', nn.Linear, space.size, **kw)(x)
        out[key] = outs.OneHot(out[key], self.unimix)
      else:
        mean = self.sub(f'pol_{key}_mean', nn.Linear, space.size, **kw)(x)
        std = self.sub(f'pol_{key}_std', nn.Linear, space.size, **kw)(x)
        std = nn.Clamp(self.minstd, self.maxstd)(jax.nn.softplus(std))
        out[key] = outs.BoundedNormal(mean, std)
    return out


class Value(nj.Module):

  units: int = 512
  norm: str = 'rms'
  act: str = 'gelu'
  layers: int = 2
  output: str = 'symexp_twohot'
  outscale: float = 0.0
  bins: int = 255

  def __init__(self, **kw):
    self.kw = kw

  def __call__(self, feat):
    x = feat
    for i in range(self.layers):
      x = self.sub(f'val{i}', nn.Linear, self.units, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'val{i}norm', nn.Norm, self.norm)(x))
    kw = dict(**self.kw, outscale=self.outscale)
    x = self.sub('valout', nn.Linear, self.bins, **kw)(x)
    return outs.SymExpTwoHot(x)