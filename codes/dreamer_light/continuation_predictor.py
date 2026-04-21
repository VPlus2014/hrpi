import embodied.jax.nets as nn
import embodied.jax.outs as outs
import ninjax as nj


class ContinuationPredictor(nj.Module):

    units: int = 512
    norm: str = "rms"
    act: str = "gelu"
    layers: int = 2
    outscale: float = 1.0

    def __init__(self, **kw):
        self.kw = kw

    def __call__(self, feat):
        x = feat
        for i in range(self.layers):
            x = self.sub(f"con{i}", nn.Linear, self.units, **self.kw)(x)
            x = nn.act(self.act)(self.sub(f"con{i}norm", nn.Norm, self.norm)(x))
        kw = dict(**self.kw, outscale=self.outscale)
        x = self.sub("conout", nn.Linear, 1, **kw)(x)
        return outs.Binary(x)
