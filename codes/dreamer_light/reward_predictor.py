import embodied.jax.nets as nn
import embodied.jax.outs as outs
import ninjax as nj


class RewardPredictor(nj.Module):

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
      x = self.sub(f'rew{i}', nn.Linear, self.units, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'rew{i}norm', nn.Norm, self.norm)(x))
    kw = dict(**self.kw, outscale=self.outscale)
    x = self.sub('rewout', nn.Linear, self.bins, **kw)(x)
    return outs.SymExpTwoHot(x, self.bins)