"""
观测解码器 (Decoder)
从隐状态重建观测
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Union, Sequence, Optional


class Decoder(nn.Module):
    """
    观测解码器

    从RSSM状态表示重建原始观测
    支持两种模式:
    - flatten: 重建向量观测
    - cnn: 重建图像观测 (解卷积)
    """

    def __init__(
        self,
        state_dim: int,                # RSSM状态维度 (hidden + latent)
        obs_shape: Union[int, Sequence[int]],  # 原始观测形状
        hidden_dims: Sequence[int] = (256, 256), # 隐藏层维度
        embed_dim: int = 256,           # 嵌入维度 (与Encoder一致)
        use_cnn: bool = False,         # 是否使用解卷积（重建图像）
        cnn_channels: Sequence[int] = (128, 64, 32), # 解卷积通道数
        use_batch_norm: bool = True,    # 是否使用批归一化
    ):
        super().__init__()
        self.state_dim = state_dim
        self.obs_shape = obs_shape
        self.embed_dim = embed_dim
        self.use_cnn = use_cnn

        if use_cnn:
            self._build_cnn_decoder(state_dim, obs_shape, embed_dim, cnn_channels)
        else:
            self._build_mlp_decoder(state_dim, obs_shape, hidden_dims)

        self._init_weights()

    def _get_output_dim(self, obs_shape: Sequence[int]) -> int:
        """计算输出维度"""
        dim = 1
        for s in obs_shape:
            dim *= s
        return dim

    def _build_mlp_decoder(
        self,
        state_dim: int,
        obs_shape: Union[int, Sequence[int]],
        hidden_dims: Sequence[int],
    ):
        """构建MLP解码器"""
        output_dim = self._get_output_dim(obs_shape)

        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.SiLU(),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def _build_cnn_decoder(
        self,
        state_dim: int,
        obs_shape: Sequence[int],
        embed_dim: int,
        cnn_channels: Sequence[int],
    ):
        """构建CNN解码器"""
        # 假设输入是 (B, state_dim)
        # 先通过FC映射到 embed_dim
        self.fc = nn.Sequential(
            nn.Linear(state_dim, embed_dim * 4 * 4),
            nn.SiLU(),
        )

        # 假设原始图像形状为 (C, H, W)
        if len(obs_shape) == 3:
            out_channels = obs_shape[0]
            h, w = obs_shape[1], obs_shape[2]
        else:
            out_channels = obs_shape[-1]
            h, w = obs_shape[0], obs_shape[1]

        # 解卷积层
        layers = []
        in_channels = embed_dim
        for out_ch in cnn_channels:
            layers.extend([
                nn.ConvTranspose2d(in_channels, out_ch, kernel_size=4, stride=2),
                nn.SiLU(),
            ])
            in_channels = out_ch

        # 最后一层
        layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2))

        self.deconv = nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.ConvTranspose2d)):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        state: Tensor,
        mean: bool = False,
    ) -> Tensor:
        """
        从状态重建观测

        Args:
            state: RSSM状态表示, shape: (B, state_dim) 或 (T, B, state_dim)
            mean: 是否返回均值（用于生成模式）

        Returns:
            重建的观测, shape: (B, *obs_shape) 或 (T, B, *obs_shape)
        """
        is_sequence = state.dim() == 3
        if is_sequence:
            T, B = state.shape[:2]
            state_flat = state.flatten(0, 1)  # (T*B, state_dim)
        else:
            state_flat = state  # (B, state_dim)

        if self.use_cnn:
            # CNN解码
            x = self.fc(state_flat)
            x = x.view(-1, self.embed_dim, 4, 4)
            output = self.deconv(x)
        else:
            # MLP解码
            output = self.net(state_flat)

        if is_sequence:
            output = output.view(T, B, *self.obs_shape)

        return output

    def log_prob(
        self,
        state: Tensor,
        target_obs: Tensor,
    ) -> Tensor:
        """
        计算重建观测的对数概率

        Args:
            state: RSSM状态表示
            target_obs: 目标观测

        Returns:
            对数概率
        """
        # 简单的高斯分布假设
        recon_obs = self.forward(state, mean=True)
        # 假设观测值在[-1, 1]范围内，使用恒定方差
        log_prob = -0.5 * ((recon_obs - target_obs) ** 2).sum(dim=-1)
        return log_prob
