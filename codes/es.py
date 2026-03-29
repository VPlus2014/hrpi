import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import Any, Callable, Tuple
import random


class NonConvexRegion:
    """n维非凸区域E的构造器"""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.circles = []
        self.rectangles = []
        self.polynomials = []
        self.indi_funcs: list[Callable[[np.ndarray], bool]] = []

    def add_circle(self, center: np.ndarray, radius: float, inside: bool = True):
        """添加圆形/球形区域"""
        center = np.array(center).reshape(self.dimension)
        self.circles.append({"center": center, "radius": radius, "inside": inside})

        self.indi_funcs.append(
            lambda x: np.sum(np.square(center - center)) <= radius**2
        )

    def add_rectangle(
        self, min_bounds: np.ndarray, max_bounds: np.ndarray, inside: bool = True
    ):
        """添加矩形/超矩形区域"""
        min_bounds = np.array(min_bounds).reshape(self.dimension)
        max_bounds = np.array(max_bounds).reshape(self.dimension)
        self.rectangles.append(
            {"min_bounds": min_bounds, "max_bounds": max_bounds, "inside": inside}
        )
        self.indi_funcs.append(
            lambda x: (np.all(x >= min_bounds) & np.all(x <= max_bounds)).item()
        )

    def add_polynomial(self, W: np.ndarray, b: np.ndarray):
        """添加多面体约束区域 (仅支持2次项以内的2D多项式)"""
        self.polynomials.append({"W": W, "b": b})
        self.indi_funcs.append(lambda x: np.all(np.dot(W, x) + b > 0).item())

    def indicator_function(self, x: np.ndarray):
        """
        示性函数f：判断点x是否在非凸区域E内
        Args:
            x: 输入点 shape=(..., dimX)
        Returns:
            yes: shape=(...,1)
        """
        if x.shape[-1] != self.dimension:
            raise ValueError(f"输入维度不匹配，期望{self.dimension}维")
        shphd = x.shape[:-1]
        x = x.reshape(-1, self.dimension)  # 矩阵化

        results: list[bool] = []
        for point in x:
            in_region = False
            for f in self.indi_funcs:
                y = f(point)
                if y:
                    in_region = True
                    break

            results.append(in_region)

        rst = np.reshape(np.asarray(results, np.bool_), shphd + (1,))
        return rst


class ParametricAffineModel(nn.Module):
    """参数化仿射集模型 C(W,b)"""

    def __init__(self, input_dim: int, num_hyperplanes: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.num_hyperplanes = num_hyperplanes

        # 权重矩阵W和偏置向量b
        self.W = nn.Parameter(torch.randn(num_hyperplanes, input_dim) * 0.1)
        self.b = nn.Parameter(torch.randn(num_hyperplanes) * 0.1)

    def forward(self, x):
        """前向传播：计算 Wx + b"""
        return torch.matmul(x, self.W.t()) + self.b

    def classify(self, x):
        """分类判别函数：Wx + b > 0"""
        scores = self.forward(x)
        # 所有超平面都满足条件才分类为正类
        return torch.all(scores > 0, dim=-1)


class MisclassificationLoss(nn.Module):
    """误分类测度代价函数"""

    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin

    def forward(self, predictions, targets):
        """
        计算误分类损失
        predictions: 模型输出 (batch_size, num_hyperplanes)
        targets: 真实标签 (batch_size,)
        """
        batch_size = predictions.shape[0]
        num_hyperplanes = predictions.shape[1]

        # 对于正样本，希望所有超平面输出都>margin
        # 对于负样本，希望至少一个超平面输出<-margin

        losses = []
        for i in range(batch_size):
            if targets[i] == 1:  # 正样本
                # 所有超平面都应该>margin
                pos_loss = torch.sum(torch.clamp(self.margin - predictions[i], min=0))
                losses.append(pos_loss)
            else:  # 负样本
                # 至少一个超平面应该<-margin
                neg_loss = torch.clamp(torch.min(predictions[i] + self.margin), min=0)
                losses.append(neg_loss)

        return torch.mean(torch.stack(losses))


class MLExpertSystem:
    """可机器学习的专家系统"""

    def __init__(self, dimension: int, num_hyperplanes: int = 3):
        self.dimension = dimension
        self.region = NonConvexRegion(dimension)
        self.model = ParametricAffineModel(dimension, num_hyperplanes)
        self.loss_fn = MisclassificationLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

    def create_sample_region_2d(self):
        """创建2D示例非凸区域"""
        if self.dimension != 2:
            raise ValueError("此方法仅适用于2D情况")

        # 添加几个圆形区域组成非凸形状
        self.region.add_circle(np.array([2, 2]), 1.0, inside=True)
        self.region.add_circle(np.array([4, 2]), 1.2, inside=True)
        self.region.add_circle(np.array([3, 4]), 0.8, inside=True)

        # 添加一个矩形区域
        self.region.add_rectangle(np.array([1, 0]), np.array([5, 1]), inside=True)

    def generate_training_data(
        self, num_samples: int = 1000
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成训练数据"""
        # 在合理范围内随机生成点
        if self.dimension == 2:
            x_range = (-1, 6)
            y_range = (-1, 6)
            X = np.random.uniform(
                [x_range[0], y_range[0]], [x_range[1], y_range[1]], (num_samples, 2)
            )
        else:
            X = np.random.uniform(-3, 3, (num_samples, self.dimension))

        # 计算标签
        y = np.array([1 if self.region.indicator_function(point) else 0 for point in X])

        return torch.FloatTensor(X), torch.LongTensor(y)

    def train(self, epochs: int = 500, num_samples: int = 1000):
        """训练模型"""
        X_train, y_train = self.generate_training_data(num_samples)

        self.model.train()
        losses = []

        for epoch in range(epochs):
            self.optimizer.zero_grad()

            # 前向传播
            predictions = self.model(X_train)

            # 计算损失
            loss = self.loss_fn(predictions, y_train)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

            if (epoch + 1) % 100 == 0:
                # 计算准确率
                with torch.no_grad():
                    pred_labels = self.model.classify(X_train)
                    accuracy = (pred_labels == y_train).float().mean()
                    print(
                        f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}"
                    )

        return losses

    def visualize_2d_results(self, resolution: int = 200):
        """2维情况下可视化分类结果"""
        if self.dimension != 2:
            raise ValueError("可视化仅支持2D情况")

        # 创建网格
        x_min, x_max = -1, 6
        y_min, y_max = -1, 6
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
        )

        grid_points = np.c_[xx.ravel(), yy.ravel()]

        # 真实标签 (专家系统)
        true_labels = np.array(
            [self.region.indicator_function(point) for point in grid_points]
        )
        true_labels = true_labels.reshape(xx.shape)

        # 模型预测
        self.model.eval()
        with torch.no_grad():
            grid_tensor = torch.FloatTensor(grid_points)
            pred_labels = self.model.classify(grid_tensor).numpy()
            pred_labels = pred_labels.reshape(xx.shape)

        # 可视化
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 真实区域
        axes[0].contourf(
            xx, yy, true_labels, levels=1, alpha=0.8, colors=["lightcoral", "lightblue"]
        )
        axes[0].contour(xx, yy, true_labels, levels=1, colors="black", linewidths=2)
        axes[0].set_title("真实非凸区域 E", fontsize=14)
        axes[0].set_xlabel("x₁")
        axes[0].set_ylabel("x₂")
        axes[0].grid(True, alpha=0.3)

        # 模型预测
        axes[1].contourf(
            xx, yy, pred_labels, levels=1, alpha=0.8, colors=["lightcoral", "lightblue"]
        )
        axes[1].contour(xx, yy, pred_labels, levels=1, colors="black", linewidths=2)
        axes[1].set_title("模型分类结果 C(W,b)", fontsize=14)
        axes[1].set_xlabel("x₁")
        axes[1].set_ylabel("x₂")
        axes[1].grid(True, alpha=0.3)

        # 误差可视化
        error_mask = (true_labels != pred_labels).astype(int)
        axes[2].contourf(
            xx, yy, error_mask, levels=1, alpha=0.8, colors=["white", "red"]
        )
        axes[2].set_title("分类误差区域", fontsize=14)
        axes[2].set_xlabel("x₁")
        axes[2].set_ylabel("x₂")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # 输出模型参数
        print("\n模型参数：")
        print("权重矩阵 W:")
        print(self.model.W.detach().numpy())
        print("偏置向量 b:")
        print(self.model.b.detach().numpy())

        # 计算总体准确率
        accuracy = 1 - np.mean(error_mask)
        print(f"\n总体分类准确率: {accuracy:.4f}")


def main():
    """主函数：演示系统使用"""
    print("=== 可机器学习的专家系统演示 ===\n")

    # 创建2D专家系统
    system = MLExpertSystem(dimension=2, num_hyperplanes=4)

    # 构造非凸区域
    print("1. 构造2D非凸区域...")
    system.create_sample_region_2d()

    # 生成并显示训练数据
    print("2. 生成训练数据...")
    X_sample, y_sample = system.generate_training_data(100)
    pos_samples = X_sample[y_sample == 1]
    neg_samples = X_sample[y_sample == 0]
    print(f"   正样本数量: {len(pos_samples)}")
    print(f"   负样本数量: {len(neg_samples)}")

    # 训练模型
    print("\n3. 训练参数化仿射集模型...")
    losses = system.train(epochs=300, num_samples=1500)

    # 可视化结果
    print("\n4. 可视化分类结果...")
    system.visualize_2d_results()

    # 展示训练过程
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title("训练损失曲线")
    plt.xlabel("Epoch")
    plt.ylabel("Misclassification Loss")
    plt.grid(True, alpha=0.3)
    plt.show()

    print("\n=== 演示完成 ===")


if __name__ == "__main__":
    main()
