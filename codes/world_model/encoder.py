"""
观测编码器 (Encoder)
将原始观测编码为嵌入向量
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Union, Sequence


class Encoder(nn.Module):
    """
    观测编码器

    将原始观测（可以是图像或特征向量）编码为嵌入向量
    支持两种模式:
    - flatten: 将展平的观测通过MLP编码
    - cnn: 将图像通过CNN编码 (如需要处理图像观测)
    """

    def __init__(
        self,
        obs_shape: Union[int, Sequence[int]],  # 观测形状
        embed_dim: int = 256,                   # 嵌入维度
        hidden_dims: Sequence[int] = (256, 256), # 隐藏层维度
        use_cnn: bool = False,                   # 是否使用CNN（处理图像）
        cnn_channels: Sequence[int] = (32, 64, 128), # CNN通道数
    ):
        super().__init__()
        self.obs_shape = obs_shape
        self.embed_dim = embed_dim
        self.use_cnn = use_cnn

        if use_cnn:
            # CNN编码器 (用于图像观测)
            self._build_cnn_encoder(obs_shape, embed_dim, cnn_channels)
        else:
            # MLP编码器 (用于向量观测)
            self._build_mlp_encoder(obs_shape, embed_dim, hidden_dims)

        self._init_weights()

    def _build_mlp_encoder(
        self,
        obs_shape: Union[int, Sequence[int]],
        embed_dim: int,
        hidden_dims: Sequence[int],
    ):
        """构建MLP编码器"""
        if isinstance(obs_shape, int):
            input_dim = obs_shape
        else:
            input_dim = 1
            for s in obs_shape:
                input_dim *= s

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.SiLU(),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, embed_dim))
        layers.append(nn.SiLU())

        self.net = nn.Sequential(*layers)

    def _build_cnn_encoder(
        self,
        obs_shape: Sequence[int],
        embed_dim: int,
        cnn_channels: Sequence[int],
    ):
        """构建CNN编码器"""
        # obs_shape: (C, H, W) 或 (H, W, C)
        if len(obs_shape) == 3:
            in_channels = obs_shape[0]
        else:
            in_channels = obs_shape[-1]

        layers = []
        prev_channels = in_channels
        for out_channels in cnn_channels:
            layers.extend([
                nn.Conv2d(prev_channels, out_channels, kernel_size=4, stride=2),
                nn.SiLU(),
            ])
            prev_channels = out_channels

        self.cnn = nn.Sequential(*layers)

        # 计算CNN输出尺寸
        with torch.no_grad():
            dummy = torch.zeros(1, *obs_shape)
            cnn_out = self.cnn(dummy)
            cnn_out_size = cnn_out.flatten(1).shape[1]

        # 展平后通过MLP映射到嵌入维度
        self.fc = nn.Sequential(
            nn.Linear(cnn_out_size, embed_dim),
            nn.SiLU(),
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, obs: Tensor) -> Tensor:
        """
        编码观测

        Args:
            obs: 观测张量, shape: (B, *obs_shape) 或 (T, B, *obs_shape)

        Returns:
            观测嵌入, shape: (B, embed_dim) 或 (T, B, embed_dim)
        """
        is_sequence = obs.dim() == 3
        if is_sequence:
            T, B = obs.shape[:2]
            obs_flat = obs.flatten(0, 1)  # (T*B, *obs_shape)
        else:
            obs_flat = obs  # (B, *obs_shape)

        if self.use_cnn:
            # CNN编码
            embeds = self.fc(self.cnn(obs_flat).flatten(1))
        else:
            # MLP编码
            obs_flat = obs_flat.flatten(1)
            embeds = self.net(obs_flat)

        if is_sequence:
            embeds = embeds.view(T, B, -1)  # (T, B, embed_dim)

        return embeds
