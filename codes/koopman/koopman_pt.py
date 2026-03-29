import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Union, Callable


class PyTorchKoopmanOperator(nn.Module):
    """
    基于PyTorch的Koopman算子实现
    用于从数据中学习受控系统的线性表示
    
    系统模型: x_{t+1} = F(x_t, u_t), y_t = g(x_t)
    Koopman算子使用观测历史 h_t = (y_{t-i})_{i=0}^{m-1} 和控制输入 u_t 预测下一步观测 y_{t+1}
    """

    def __init__(
        self,
        observation_dim: int,
        control_dim: int,
        history_length: int = 1,
        basis_type: str = "polynomial",
        basis_order: int = 2,
        hidden_dim: int = 64,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        初始化PyTorch Koopman算子

        参数:
        observation_dim: 观测向量维度
        control_dim: 控制向量维度
        history_length: 历史观测长度 m
        basis_type: 基函数类型 ('polynomial', 'neural')
        basis_order: 多项式基函数阶数 (仅用于polynomial)
        hidden_dim: 神经网络隐藏层维度 (仅用于neural)
        device: 计算设备 ('cpu' 或 'cuda')
        """
        super().__init__()

        self.observation_dim = observation_dim
        self.control_dim = control_dim
        self.history_length = history_length
        self.basis_type = basis_type
        self.basis_order = basis_order
        self.hidden_dim = hidden_dim
        self.device = device

        # 计算基函数维度
        if basis_type == "polynomial":
            # 多项式基函数维度计算
            # 对于d维向量和n阶多项式，维度为 C(n+d,n)
            d = observation_dim * history_length
            n = basis_order
            self.basis_dim = 1
            for i in range(1, n + 1):
                self.basis_dim += self._combination(d + i - 1, i)
        elif basis_type == "neural":
            # 神经网络基函数，输出维度自定义
            self.basis_dim = hidden_dim
        else:
            raise ValueError(f"Unknown basis type: {basis_type}")

        # 初始化基函数网络
        self._init_basis_network()

        # 初始化Koopman算子 (线性层)
        # 输入: 提升后的状态 + 控制输入
        # 输出: 下一步的提升状态
        self.koopman_layer = nn.Linear(
            self.basis_dim + control_dim, self.basis_dim, bias=True
        )

        # 初始化解码器 (从提升空间映射回观测空间)
        self.decoder = nn.Linear(self.basis_dim, observation_dim, bias=True)

        # 数据归一化参数
        self.register_buffer("obs_mean", torch.zeros(observation_dim))
        self.register_buffer("obs_std", torch.ones(observation_dim))
        self.register_buffer("control_mean", torch.zeros(control_dim))
        self.register_buffer("control_std", torch.ones(control_dim))

        # 移动模型到指定设备
        self.to(device)

    def _combination(self, n: int, k: int) -> int:
        """计算组合数 C(n,k)"""
        if k < 0 or k > n:
            return 0
        if k == 0 or k == n:
            return 1
        k = min(k, n - k)
        c = 1
        for i in range(k):
            c = c * (n - i) // (i + 1)
        return c

    def _init_basis_network(self):
        """初始化基函数网络"""
        input_dim = self.observation_dim * self.history_length

        if self.basis_type == "polynomial":
            # 多项式基函数不需要额外的网络层
            self.basis_network = None
        elif self.basis_type == "neural":
            # 神经网络基函数
            self.basis_network = nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.Tanh(),
            )
        else:
            raise ValueError(f"Unknown basis type: {self.basis_type}")

    def polynomial_basis(self, x: torch.Tensor) -> torch.Tensor:
        """
        多项式基函数
        
        参数:
        x: 输入张量 [batch_size, observation_dim * history_length]
        
        返回:
        psi: 提升后的张量 [batch_size, basis_dim]
        """
        batch_size = x.shape[0]
        psi = torch.ones(batch_size, 1, device=self.device)

        # 添加一阶项 (原始状态)
        psi = torch.cat([psi, x], dim=1)

        if self.basis_order >= 2:
            # 添加二阶项 (所有可能的两两相乘)
            n = x.shape[1]
            for i in range(n):
                for j in range(i, n):
                    psi = torch.cat([psi, (x[:, i] * x[:, j]).unsqueeze(1)], dim=1)

        if self.basis_order >= 3:
            # 添加三阶项
            for i in range(n):
                for j in range(i, n):
                    for k in range(j, n):
                        psi = torch.cat(
                            [psi, (x[:, i] * x[:, j] * x[:, k]).unsqueeze(1)], dim=1
                        )

        # 注意: 更高阶的多项式可以类似添加，但计算量会迅速增加

        return psi

    def neural_basis(self, x: torch.Tensor) -> torch.Tensor:
        """
        神经网络基函数
        
        参数:
        x: 输入张量 [batch_size, observation_dim * history_length]
        
        返回:
        psi: 提升后的张量 [batch_size, basis_dim]
        """
        return self.basis_network(x)

    def lift_state(self, observations_history: torch.Tensor) -> torch.Tensor:
        """
        将观测历史提升到高维空间
        
        参数:
        observations_history: 观测历史 [batch_size, history_length, observation_dim]
        
        返回:
        psi: 提升后的状态 [batch_size, basis_dim]
        """
        batch_size = observations_history.shape[0]

        # 重塑观测历史为 [batch_size, history_length * observation_dim]
        x = observations_history.reshape(batch_size, -1)

        # 归一化
        x = (x - self.obs_mean.repeat(self.history_length)) / (
            self.obs_std.repeat(self.history_length) + 1e-8
        )

        # 应用基函数
        if self.basis_type == "polynomial":
            return self.polynomial_basis(x)
        elif self.basis_type == "neural":
            return self.neural_basis(x)
        else:
            raise ValueError(f"Unknown basis type: {self.basis_type}")

    def forward(
        self, observations_history: torch.Tensor, control: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播: 预测下一步观测
        
        参数:
        observations_history: 观测历史 [batch_size, history_length, observation_dim]
        control: 控制输入 [batch_size, control_dim]
        
        返回:
        next_observation: 预测的下一步观测 [batch_size, observation_dim]
        """
        # 归一化控制输入
        control_norm = (control - self.control_mean) / (self.control_std + 1e-8)

        # 提升状态
        psi = self.lift_state(observations_history)

        # 连接提升状态和控制输入
        psi_u = torch.cat([psi, control_norm], dim=1)

        # 应用Koopman算子
        next_psi = self.koopman_layer(psi_u)

        # 解码回观测空间
        next_obs_norm = self.decoder(next_psi)

        # 反归一化
        next_observation = next_obs_norm * self.obs_std + self.obs_mean

        return next_observation

    def fit(
        self,
        observations: torch.Tensor,
        controls: torch.Tensor,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        verbose: bool = True,
    ) -> dict:
        """
        训练Koopman算子
        
        参数:
        observations: 观测序列 [seq_length, observation_dim]
        controls: 控制序列 [seq_length, control_dim]
        epochs: 训练轮数
        batch_size: 批大小
        learning_rate: 学习率
        weight_decay: 权重衰减
        verbose: 是否打印训练信息
        
        返回:
        history: 训练历史
        """
        # 确保输入是PyTorch张量
        if not isinstance(observations, torch.Tensor):
            observations = torch.tensor(
                observations, dtype=torch.float32, device=self.device
            )
        if not isinstance(controls, torch.Tensor):
            controls = torch.tensor(controls, dtype=torch.float32, device=self.device)

        # 计算归一化参数
        self.obs_mean = observations.mean(dim=0)
        self.obs_std = observations.std(dim=0)
        self.control_mean = controls.mean(dim=0)
        self.control_std = controls.std(dim=0)

        # 准备训练数据
        dataset = self._prepare_training_data(observations, controls)

        # 创建数据加载器
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        # 定义优化器和损失函数
        optimizer = optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        criterion = nn.MSELoss()

        # 训练历史
        history = {"loss": [], "val_loss": []}

        # 训练循环
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            num_batches = 0

            for batch in dataloader:
                obs_history, control, next_obs = batch

                # 前向传播
                pred_next_obs = self(obs_history, control)

                # 计算损失
                loss = criterion(pred_next_obs, next_obs)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_epoch_loss = epoch_loss / num_batches
            history["loss"].append(avg_epoch_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.6f}")

        return history

    def _prepare_training_data(
        self, observations: torch.Tensor, controls: torch.Tensor
    ) -> torch.utils.data.TensorDataset:
        """
        准备训练数据
        
        参数:
        observations: 观测序列 [seq_length, observation_dim]
        controls: 控制序列 [seq_length, control_dim]
        
        返回:
        dataset: 训练数据集
        """
        seq_length = observations.shape[0]
        m = self.history_length

        # 创建训练样本
        obs_history_list = []
        control_list = []
        next_obs_list = []

        for t in range(m, seq_length - 1):
            # 观测历史 [t-m, t-m+1, ..., t-1, t]
            obs_history = observations[t - m + 1 : t + 1]
            # 当前控制输入
            control = controls[t]
            # 下一步观测
            next_obs = observations[t + 1]

            obs_history_list.append(obs_history)
            control_list.append(control)
            next_obs_list.append(next_obs)

        # 转换为张量
        obs_history_tensor = torch.stack(obs_history_list)
        control_tensor = torch.stack(control_list)
        next_obs_tensor = torch.stack(next_obs_list)

        # 创建数据集
        dataset = torch.utils.data.TensorDataset(
            obs_history_tensor, control_tensor, next_obs_tensor
        )

        return dataset

    def predict(
        self,
        initial_observations: torch.Tensor,
        controls: torch.Tensor,
        steps: int = 1,
    ) -> torch.Tensor:
        """
        多步预测
        
        参数:
        initial_observations: 初始观测历史 [history_length, observation_dim]
        controls: 控制序列 [steps, control_dim]
        steps: 预测步数
        
        返回:
        predictions: 预测序列 [steps, observation_dim]
        """
        self.eval()

        # 确保输入是PyTorch张量
        if not isinstance(initial_observations, torch.Tensor):
            initial_observations = torch.tensor(
                initial_observations, dtype=torch.float32, device=self.device
            )
        if not isinstance(controls, torch.Tensor):
            controls = torch.tensor(controls, dtype=torch.float32, device=self.device)

        # 初始观测历史
        obs_history = initial_observations.unsqueeze(0)  # [1, history_length, observation_dim]

        # 预测结果
        predictions = []

        # 逐步预测
        for t in range(steps):
            # 获取当前控制输入
            control = controls[t].unsqueeze(0)  # [1, control_dim]

            # 预测下一步观测
            next_obs = self(obs_history, control)  # [1, observation_dim]
            predictions.append(next_obs.squeeze(0))

            # 更新观测历史 (滑动窗口)
            obs_history = torch.cat(
                [obs_history[:, 1:], next_obs.unsqueeze(1)], dim=1
            )

        # 返回预测序列
        return torch.stack(predictions)

    def get_koopman_matrix(self) -> torch.Tensor:
        """
        获取Koopman矩阵
        
        返回:
        K: Koopman矩阵 [basis_dim, basis_dim + control_dim]
        """
        return self.koopman_layer.weight

    def get_eigenvalues(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算Koopman矩阵的特征值和特征向量
        
        返回:
        eigenvalues: 特征值
        eigenvectors: 特征向量
        """
        # 提取Koopman矩阵 (仅状态部分)
        K = self.koopman_layer.weight[:, : self.basis_dim].detach().cpu()

        # 计算特征值和特征向量
        eigenvalues, eigenvectors = torch.linalg.eig(K)

        return eigenvalues, eigenvectors

    def plot_eigenvalues(self, title: str = "PyTorch Koopman Eigenvalues"):
        """
        绘制特征值
        
        参数:
        title: 图表标题
        """
        eigenvalues, _ = self.get_eigenvalues()
        eigenvalues = eigenvalues.numpy()

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.scatter(eigenvalues.real, eigenvalues.imag, alpha=0.7, s=50)
        plt.axhline(y=0, color="k", linestyle="--", alpha=0.3)
        plt.axvline(x=0, color="k", linestyle="--", alpha=0.3)

        # 单位圆
        theta = np.linspace(0, 2 * np.pi, 100)
        plt.plot(np.cos(theta), np.sin(theta), "r--", alpha=0.5, label="Unit Circle")

        plt.xlabel("Real Part")
        plt.ylabel("Imaginary Part")
        plt.title(f"{title} - Complex Plane")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis("equal")

        plt.subplot(1, 2, 2)
        magnitudes = np.abs(eigenvalues)
        plt.scatter(range(len(magnitudes)), magnitudes, alpha=0.7, s=50)
        plt.axhline(
            y=1, color="r", linestyle="--", alpha=0.5, label="Stability Boundary"
        )
        plt.xlabel("Eigenvalue Index")
        plt.ylabel("Magnitude")
        plt.title(f"{title} - Magnitudes")
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.show()

    def save(self, path: str):
        """
        保存模型
        
        参数:
        path: 保存路径
        """
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "obs_mean": self.obs_mean,
                "obs_std": self.obs_std,
                "control_mean": self.control_mean,
                "control_std": self.control_std,
                "observation_dim": self.observation_dim,
                "control_dim": self.control_dim,
                "history_length": self.history_length,
                "basis_type": self.basis_type,
                "basis_order": self.basis_order,
                "hidden_dim": self.hidden_dim,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        加载模型
        
        参数:
        path: 模型路径
        device: 计算设备
        
        返回:
        model: 加载的模型
        """
        checkpoint = torch.load(path, map_location=device)

        model = cls(
            observation_dim=checkpoint["observation_dim"],
            control_dim=checkpoint["control_dim"],
            history_length=checkpoint["history_length"],
            basis_type=checkpoint["basis_type"],
            basis_order=checkpoint["basis_order"],
            hidden_dim=checkpoint["hidden_dim"],
            device=device,
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.obs_mean = checkpoint["obs_mean"]
        model.obs_std = checkpoint["obs_std"]
        model.control_mean = checkpoint["control_mean"]
        model.control_std = checkpoint["control_std"]

        return model


def test_pytorch_koopman():
    """测试PyTorch Koopman算子"""
    import numpy as np
    from aircraft_dynamics import RigidBodyAircraft, sinusoidal_control

    # 创建飞机模型
    aircraft = RigidBodyAircraft()
    trim_state, _ = aircraft.trim_condition()

    # 生成训练数据
    t, trajectory = aircraft.simulate(
        initial_state=trim_state,
        time_span=(0, 20),
        controls_func=sinusoidal_control,
        dt=0.1,
    )

    # 提取观测 (这里假设观测是状态的前6个分量)
    observations = trajectory[:, :6]
    
    # 生成控制序列
    controls = np.array([sinusoidal_control(ti) for ti in t])

    print(f"Generated trajectory with {len(trajectory)} time steps")
    print(f"Observations shape: {observations.shape}")
    print(f"Controls shape: {controls.shape}")

    # 转换为PyTorch张量
    observations_tensor = torch.tensor(observations, dtype=torch.float32)
    controls_tensor = torch.tensor(controls, dtype=torch.float32)

    # 创建PyTorch Koopman算子
    koopman_pt = PyTorchKoopmanOperator(
        observation_dim=6,
        control_dim=4,
        history_length=3,
        basis_type="neural",
        hidden_dim=32,
    )

    # 训练模型
    history = koopman_pt.fit(
        observations_tensor,
        controls_tensor,
        epochs=50,
        batch_size=32,
        learning_rate=1e-3,
    )

    # 测试预测
    test_idx = 100
    initial_obs = observations_tensor[test_idx:test_idx+3]
    test_controls = controls_tensor[test_idx+3:test_idx+13]
    
    predictions = koopman_pt.predict(initial_obs, test_controls, steps=10)
    true_observations = observations_tensor[test_idx+3:test_idx+13]
    
    # 计算预测误差
    error = torch.mean(torch.norm(predictions - true_observations, dim=1))
    print(f"Mean prediction error: {error.item():.4f}")

    # 绘制特征值
    koopman_pt.plot_eigenvalues("PyTorch Neural Koopman Operator")

    # 绘制预测结果
    plt.figure(figsize=(12, 8))
    
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.plot(t[test_idx+3:test_idx+13], true_observations[:, i].numpy(), 'b-', label='True')
        plt.plot(t[test_idx+3:test_idx+13], predictions[:, i].detach().numpy(), 'r--', label='Predicted')
        plt.title(f'State {i+1}')
        plt.grid(True)
        if i == 0:
            plt.legend()
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_pytorch_koopman()