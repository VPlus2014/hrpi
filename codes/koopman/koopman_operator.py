import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from scipy.linalg import eig, pinv
import matplotlib.pyplot as plt


class KoopmanOperator:
    """
    Koopman算子实现类
    用于从数据中学习系统的线性表示
    """

    def __init__(self, basis_type="polynomial", basis_order=2, regularization=1e-6):
        """
        初始化Koopman算子

        参数:
        basis_type: 基函数类型 ('polynomial', 'rbf', 'fourier')
        basis_order: 基函数阶数
        regularization: 正则化参数
        """
        self.basis_type = basis_type
        self.basis_order = basis_order
        self.regularization = regularization

        # Koopman矩阵
        self.K = None
        self.eigenvalues = None
        self.eigenvectors = None

        # 基函数相关
        self.poly_features = None
        self.n_basis_functions = None
        self.state_dim = None

        # 数据统计
        self.state_mean = None
        self.state_std = None
        self.normalize_data = True

    def polynomial_basis(self, states):
        """多项式基函数"""
        if self.poly_features is None:
            self.poly_features = PolynomialFeatures(
                degree=self.basis_order, include_bias=True
            )
            # 拟合并变换
            psi = self.poly_features.fit_transform(states)
        else:
            psi = self.poly_features.transform(states)

        return psi

    def rbf_basis(self, states, centers=None, sigma=1.0):
        """径向基函数"""
        n_samples, n_states = states.shape

        if centers is None:
            # 使用K-means或随机选择中心点
            n_centers = min(50, n_samples // 2)
            idx = np.random.choice(n_samples, n_centers, replace=False)
            centers = states[idx]
            self.rbf_centers = centers
        else:
            centers = self.rbf_centers

        n_centers = centers.shape[0]
        psi = np.zeros((n_samples, n_centers + n_states + 1))

        # 原始状态
        psi[:, :n_states] = states

        # RBF基函数
        for i, center in enumerate(centers):
            distances = np.sum((states - center) ** 2, axis=1)
            psi[:, n_states + i] = np.exp(-distances / (2 * sigma**2))

        # 常数项
        psi[:, -1] = 1.0

        return psi

    def fourier_basis(self, states, n_frequencies=5):
        """傅里叶基函数"""
        n_samples, n_states = states.shape

        # 基函数数量
        n_basis = n_states + 2 * n_frequencies * n_states + 1
        psi = np.zeros((n_samples, n_basis))

        # 原始状态
        psi[:, :n_states] = states

        # 正弦和余弦项
        idx = n_states
        for k in range(1, n_frequencies + 1):
            for i in range(n_states):
                psi[:, idx] = np.sin(k * states[:, i])
                psi[:, idx + 1] = np.cos(k * states[:, i])
                idx += 2

        # 常数项
        psi[:, -1] = 1.0

        return psi

    def lift_state(self, states):
        """
        将状态提升到观测空间
        states: (n_samples, n_states)
        返回: (n_samples, n_basis_functions)
        """
        if self.normalize_data and self.state_mean is not None:
            states = (states - self.state_mean) / (self.state_std + 1e-8)

        if self.basis_type == "polynomial":
            return self.polynomial_basis(states)
        elif self.basis_type == "rbf":
            return self.rbf_basis(states)
        elif self.basis_type == "fourier":
            return self.fourier_basis(states)
        else:
            raise ValueError(f"Unknown basis type: {self.basis_type}")

    def fit(self, states, next_states, controls=None):
        """
        从数据中学习Koopman算子

        参数:
        states: 当前状态 (n_samples, n_states)
        next_states: 下一步状态 (n_samples, n_states)
        controls: 控制输入 (n_samples, n_controls) - 可选
        """
        states = np.array(states)
        next_states = np.array(next_states)

        if len(states.shape) == 1:
            states = states.reshape(1, -1)
        if len(next_states.shape) == 1:
            next_states = next_states.reshape(1, -1)

        self.state_dim = states.shape[1]

        # 数据标准化
        if self.normalize_data:
            self.state_mean = np.mean(states, axis=0)
            self.state_std = np.std(states, axis=0)
            states_norm = (states - self.state_mean) / (self.state_std + 1e-8)
            next_states_norm = (next_states - self.state_mean) / (self.state_std + 1e-8)
        else:
            states_norm = states
            next_states_norm = next_states

        # 提升状态到观测空间
        psi_X = self.lift_state(states_norm)
        psi_Y = self.lift_state(next_states_norm)

        self.n_basis_functions = psi_X.shape[1]

        # 如果有控制输入，扩展观测向量
        if controls is not None:
            controls = np.array(controls)
            controls = controls.reshape(1, -1)

            n_controls = controls.shape[1]
            psi_X_extended = np.concatenate([psi_X, controls], axis=1)
            self.n_basis_functions += n_controls
        else:
            psi_X_extended = psi_X

        # 计算Koopman算子 K: psi_Y = K @ psi_X
        # 使用伪逆求解: K = psi_Y @ pinv(psi_X)
        try:
            if self.regularization > 0:
                # 使用脊回归进行正则化
                n_basis = psi_X_extended.shape[1]
                A = psi_X_extended.T @ psi_X_extended + self.regularization * np.eye(
                    n_basis
                )
                B = psi_X_extended.T @ psi_Y
                self.K = np.linalg.solve(A, B).T
            else:
                self.K = psi_Y @ pinv(psi_X_extended)
        except np.linalg.LinAlgError:
            print("Warning: Using regularized solution due to singular matrix")
            n_basis = psi_X_extended.shape[1]
            A = psi_X_extended.T @ psi_X_extended + 1e-6 * np.eye(n_basis)
            B = psi_X_extended.T @ psi_Y
            self.K = np.linalg.solve(A, B).T

        # 特征值分解
        try:
            self.eigenvalues, self.eigenvectors = eig(
                self.K[:, : self.n_basis_functions]
            )
        except:
            print("Warning: Eigenvalue decomposition failed")
            self.eigenvalues = None
            self.eigenvectors = None

        return self

    def predict(self, initial_state, n_steps, controls=None):
        """
        多步预测

        参数:
        initial_state: 初始状态 (n_states,)
        n_steps: 预测步数
        controls: 控制序列 (n_steps, n_controls) - 可选

        返回:
        predictions: 预测轨迹 (n_steps+1, n_states)
        """
        if self.K is None:
            raise ValueError("Model not fitted yet!")

        initial_state = np.array(initial_state).reshape(1, -1)
        predictions = [initial_state[0]]

        current_state = initial_state.copy()

        for step in range(n_steps):
            # 提升当前状态
            psi = self.lift_state(current_state)

            # 添加控制输入
            if controls is not None:
                control = controls[step].reshape(1, -1)
                psi_extended = np.concatenate([psi, control], axis=1)
            else:
                psi_extended = psi

            # Koopman预测
            psi_next = (self.K @ psi_extended.T).T

            # 提取状态部分 (假设前n_states个基函数对应原始状态)
            if self.normalize_data:
                next_state = (
                    psi_next[0, : self.state_dim] * (self.state_std + 1e-8)
                    + self.state_mean
                )
            else:
                next_state = psi_next[0, : self.state_dim]

            predictions.append(next_state)
            current_state = next_state.reshape(1, -1)

        return np.array(predictions)

    def get_koopman_modes(self, n_modes=None):
        """获取Koopman模态"""
        if self.eigenvalues is None:
            return None, None

        if n_modes is None:
            n_modes = len(self.eigenvalues)

        # 按特征值模长排序
        eigenval_magnitudes = np.abs(self.eigenvalues)
        sorted_indices = np.argsort(eigenval_magnitudes)[::-1]

        dominant_eigenvals = self.eigenvalues[sorted_indices[:n_modes]]
        dominant_modes = self.eigenvectors[:, sorted_indices[:n_modes]]

        return dominant_eigenvals, dominant_modes

    def reconstruction_error(self, states, next_states, controls=None):
        """计算重构误差"""
        predictions = []

        for i, state in enumerate(states):
            if controls is not None:
                control = controls[i : i + 1]
            else:
                control = None

            pred = self.predict(state, 1, control)
            predictions.append(pred[1])  # 下一步预测

        predictions = np.array(predictions)
        error = np.mean(np.linalg.norm(predictions - next_states, axis=1))

        return error

    def spectral_analysis(self):
        """频谱分析"""
        if self.eigenvalues is None:
            return None

        eigenvals = self.eigenvalues
        frequencies = np.angle(eigenvals) / (2 * np.pi)  # 频率
        growth_rates = np.real(np.log(eigenvals))  # 增长率

        return frequencies, growth_rates

    def plot_eigenvalues(self, title="Koopman Eigenvalues"):
        """绘制特征值"""
        if self.eigenvalues is None:
            print("No eigenvalues to plot")
            return

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.scatter(
            np.real(self.eigenvalues), np.imag(self.eigenvalues), alpha=0.7, s=50
        )
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
        magnitudes = np.abs(self.eigenvalues)
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


class MultiStepKoopman:
    """多步观测Koopman算子"""

    def __init__(self, delay_steps=1, **koopman_kwargs):
        """
        初始化多步观测Koopman算子

        参数:
        delay_steps: 延迟嵌入步数 M: h_t=(Y_{t-k})_{0<=k<M}
        **koopman_kwargs: KoopmanOperator的参数
        """
        self._M = delay_steps
        self.koopman = KoopmanOperator(**koopman_kwargs)

    def create_delay_embedding(self, trajectory):
        """
        创建延迟嵌入
        trajectory: (n_timesteps, n_states)
        返回: (n_samples, delay_steps * n_states)
        """
        n_timesteps, n_states = trajectory.shape
        n_samples = n_timesteps - self._M

        embedded_states = np.zeros((n_samples, self._M * n_states))

        for i in range(n_samples):
            for j in range(self._M):
                start_idx = j * n_states
                end_idx = (j + 1) * n_states
                embedded_states[i, start_idx:end_idx] = trajectory[i + j]

        return embedded_states

    def fit(self, trajectory, controls=None):
        """
        训练多步观测Koopman算子

        参数:
        trajectory: 轨迹数据 (n_timesteps, n_states)
        controls: 控制输入 (n_timesteps, n_controls)
        """
        # 创建延迟嵌入
        embedded_states = self.create_delay_embedding(trajectory)

        # 下一步的延迟嵌入状态
        next_embedded_states = embedded_states[1:]
        embedded_states = embedded_states[:-1]

        # 对应的控制输入
        if controls is not None:
            embedded_controls = controls[self._M - 1 : -1]
        else:
            embedded_controls = None

        # 训练Koopman算子
        self.koopman.fit(embedded_states, next_embedded_states, embedded_controls)

        return self

    def predict(self, initial_trajectory, n_steps, controls=None):
        """
        多步预测

        参数:
        initial_trajectory: 初始轨迹段 (delay_steps, n_states)
        n_steps: 预测步数
        controls: 控制序列 (n_steps, n_controls)

        返回:
        predictions: 预测轨迹 (n_steps, n_states)
        """
        initial_trajectory = np.array(initial_trajectory)
        if initial_trajectory.shape[0] != self._M:
            raise ValueError(
                f"Initial trajectory must have {self._M} time steps"
            )

        n_states = initial_trajectory.shape[1]
        predictions = []

        # 创建初始延迟嵌入状态
        current_embedded = initial_trajectory.flatten().reshape(1, -1)

        for step in range(n_steps):
            # 使用Koopman算子预测下一步
            if controls is not None:
                control = controls[step : step + 1]
            else:
                control = None

            next_embedded = self.koopman.predict(current_embedded[0], 1, control)
            next_embedded_state = next_embedded[1].reshape(1, -1)

            # 提取最新的状态 (最后n_states个元素)
            newest_state = next_embedded_state[0, -n_states:]
            predictions.append(newest_state)

            # 更新延迟嵌入状态 (滑动窗口)
            current_embedded = next_embedded_state

        return np.array(predictions)


def main():
    # 测试代码
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

    print(f"Generated trajectory with {len(trajectory)} time steps")

    # 测试多项式基函数Koopman算子
    koopman_poly = KoopmanOperator(basis_type="polynomial", basis_order=2)

    # 准备训练数据
    states = trajectory[:-1]  # 当前状态
    next_states = trajectory[1:]  # 下一步状态

    # 训练
    koopman_poly.fit(states, next_states)
    print(f"Koopman matrix shape: {koopman_poly.K.shape}")
    print(f"Number of basis functions: {koopman_poly.n_basis_functions}")

    # 测试预测
    test_initial = trajectory[100]
    prediction = koopman_poly.predict(test_initial, 50)
    true_trajectory = trajectory[100:151]

    # 计算误差
    if len(true_trajectory) >= len(prediction):
        error = np.mean(
            np.linalg.norm(prediction[: len(true_trajectory)] - true_trajectory, axis=1)
        )
        print(f"Mean prediction error: {error:.4f}")

    # 绘制特征值
    koopman_poly.plot_eigenvalues("Polynomial Koopman Operator")

    # 测试多步观测Koopman算子
    print("\nTesting Multi-Step Koopman Operator...")
    multistep_koopman = MultiStepKoopman(
        delay_steps=5, basis_type="polynomial", basis_order=2
    )
    multistep_koopman.fit(trajectory)

    # 多步预测
    initial_segment = trajectory[100:105]  # 5步初始轨迹
    multistep_prediction = multistep_koopman.predict(initial_segment, 30)

    print(f"Multi-step prediction shape: {multistep_prediction.shape}")

    # 计算多步预测误差
    true_multistep = trajectory[105:135]
    if len(true_multistep) >= len(multistep_prediction):
        multistep_error = np.mean(
            np.linalg.norm(
                multistep_prediction[: len(true_multistep)] - true_multistep, axis=1
            )
        )
        print(f"Multi-step prediction error: {multistep_error:.4f}")

    plt.show()


if __name__ == "__main__":
    main()
