import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from koopman_operator import KoopmanOperator, MultiStepKoopman
import pickle
import warnings

class KoopmanLearner:
    """
    Koopman算子学习算法集合
    包含各种训练策略、超参数优化、模型选择等功能
    """

    def __init__(self):
        self.models = {}
        self.best_model = None
        self.training_history = {}

    def grid_search(self, trajectory, param_grid, cv_folds=5, scoring='mse'):
        """
        网格搜索最佳超参数
        
        参数:
        trajectory: 轨迹数据 (n_timesteps, n_states)  
        param_grid: 参数网格 dict
        cv_folds: 交叉验证折数
        scoring: 评分方法 ('mse', 'r2', 'prediction_horizon')
        
        返回:
        best_params: 最佳参数
        results: 所有参数组合的结果
        """
        states = trajectory[:-1]
        next_states = trajectory[1:]

        # 分割数据用于交叉验证
        n_samples = len(states)
        fold_size = n_samples // cv_folds

        results = []
        best_score = float('inf') if scoring == 'mse' else -float('inf')
        best_params = None

        # 生成参数组合
        param_combinations = self._generate_param_combinations(param_grid)

        print(f"Testing {len(param_combinations)} parameter combinations...")

        for i, params in enumerate(param_combinations):
            print(f"Progress: {i+1}/{len(param_combinations)} - Testing: {params}")

            cv_scores = []

            # 交叉验证
            for fold in range(cv_folds):
                try:
                    # 创建训练/验证分割
                    val_start = fold * fold_size
                    val_end = (fold + 1) * fold_size if fold < cv_folds - 1 else n_samples

                    val_indices = np.arange(val_start, val_end)
                    train_indices = np.concatenate([np.arange(0, val_start), 
                                                   np.arange(val_end, n_samples)])

                    train_states = states[train_indices]
                    train_next_states = next_states[train_indices]
                    val_states = states[val_indices]
                    val_next_states = next_states[val_indices]

                    # 训练模型
                    model = KoopmanOperator(**params)
                    model.fit(train_states, train_next_states)

                    # 评估
                    if scoring == 'mse':
                        score = model.reconstruction_error(val_states, val_next_states)
                    elif scoring == 'r2':
                        predictions = []
                        for state in val_states:
                            pred = model.predict(state, 1)
                            predictions.append(pred[1])
                        predictions = np.array(predictions)
                        score = r2_score(val_next_states.flatten(), predictions.flatten())
                    elif scoring == 'prediction_horizon':
                        score = self._evaluate_prediction_horizon(model, val_states, 
                                                                trajectory, val_start)

                    cv_scores.append(score)

                except Exception as e:
                    print(f"Error in fold {fold}: {e}")
                    cv_scores.append(float('inf') if scoring == 'mse' else -float('inf'))

            # 计算平均分数
            avg_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)

            result = {
                'params': params,
                'mean_score': avg_score,
                'std_score': std_score,
                'scores': cv_scores
            }
            results.append(result)

            # 更新最佳参数
            if ((scoring == 'mse' and avg_score < best_score) or 
                (scoring != 'mse' and avg_score > best_score)):
                best_score = avg_score
                best_params = params.copy()

        # 训练最佳模型
        print(f"\nBest parameters: {best_params}")
        print(f"Best score: {best_score:.6f}")
        assert best_params is not None, "No valid parameter combination found."
        self.best_model = KoopmanOperator(**best_params)
        self.best_model.fit(states, next_states)

        return best_params, results

    def _generate_param_combinations(self, param_grid):
        """生成参数组合"""
        keys = list(param_grid.keys())
        values = list(param_grid.values())

        combinations = []
        self._recursive_combinations(keys, values, 0, {}, combinations)

        return combinations

    def _recursive_combinations(self, keys, values, index, current, combinations):
        """递归生成参数组合"""
        if index == len(keys):
            combinations.append(current.copy())
            return

        key = keys[index]
        for value in values[index]:
            current[key] = value
            self._recursive_combinations(keys, values, index + 1, current, combinations)

    def _evaluate_prediction_horizon(self, model, val_states, full_trajectory, start_idx):
        """评估预测视野长度"""
        horizon_scores = []
        max_horizon = min(50, len(full_trajectory) - start_idx - 20)

        for i in range(0, min(len(val_states), 10)):  # 测试前10个状态
            initial_state = val_states[i]
            true_trajectory = full_trajectory[start_idx + i:start_idx + i + max_horizon]

            # 逐步增加预测长度，直到误差超过阈值
            horizon = 1
            threshold = 1.0  # 误差阈值

            while horizon < max_horizon:
                try:
                    prediction = model.predict(initial_state, horizon)
                    if horizon <= len(true_trajectory):
                        error = np.mean(np.linalg.norm(
                            prediction[1:horizon+1] - true_trajectory[:horizon], axis=1))
                        if error > threshold:
                            break
                    horizon += 1
                except:
                    break

            horizon_scores.append(horizon)

        return np.mean(horizon_scores)

    def adaptive_basis_selection(self, trajectory, max_order=5, tolerance=1e-4):
        """
        自适应基函数选择
        自动确定最适合的基函数类型和阶数
        """
        basis_types = ['polynomial', 'rbf', 'fourier']
        states = trajectory[:-1]
        next_states = trajectory[1:]

        best_model = None
        best_score = float('inf')
        best_config = None

        results = []

        for basis_type in basis_types:
            print(f"\nTesting {basis_type} basis functions...")

            if basis_type == 'polynomial':
                orders = range(1, max_order + 1)
            elif basis_type == 'rbf':
                orders = [1]  # RBF不需要阶数
            else:  # fourier
                orders = range(2, min(8, max_order + 1))

            for order in orders:
                try:
                    if basis_type == 'rbf':
                        model = KoopmanOperator(basis_type=basis_type)
                    else:
                        model = KoopmanOperator(basis_type=basis_type, basis_order=order)

                    # 训练模型
                    model.fit(states, next_states)

                    # 评估重构误差
                    recon_error = model.reconstruction_error(states[:100], next_states[:100])

                    # 评估预测性能
                    test_states = states[-50:]
                    predictions = []
                    for state in test_states:
                        pred = model.predict(state, 1)
                        predictions.append(pred[1])
                    predictions = np.array(predictions)

                    pred_error = np.mean(np.linalg.norm(
                        predictions - next_states[-50:], axis=1))

                    # 综合评分 (重构误差 + 预测误差)
                    total_score = recon_error + pred_error

                    result = {
                        'basis_type': basis_type,
                        'order': order,
                        'reconstruction_error': recon_error,
                        'prediction_error': pred_error,
                        'total_score': total_score,
                        'n_basis': model.n_basis_functions
                    }
                    results.append(result)

                    print(f"  Order {order}: Recon={recon_error:.4f}, "
                          f"Pred={pred_error:.4f}, Total={total_score:.4f}")

                    if total_score < best_score:
                        best_score = total_score
                        best_model = model
                        best_config = result

                except Exception as e:
                    print(f"  Order {order}: Failed - {e}")

        assert (
            best_config is not None
        ), "No valid model configuration found."
        print(f"\nBest configuration:")
        print(f"  Basis type: {best_config['basis_type']}")
        print(f"  Order: {best_config['order']}")
        print(f"  Total score: {best_config['total_score']:.4f}")
        print(f"  Basis functions: {best_config['n_basis']}")

        assert best_model is not None, "No valid model found."
        self.best_model = best_model

        return best_config, results

    def incremental_learning(self, initial_trajectory, new_data_stream, 
                           forgetting_factor=0.95, update_frequency=10):
        """
        增量学习算法
        在线更新Koopman算子
        """
        # 初始训练
        states = initial_trajectory[:-1]
        next_states = initial_trajectory[1:]

        model = KoopmanOperator(regularization=1e-4)
        model.fit(states, next_states)

        self.models['incremental'] = model

        # 增量更新
        update_history = []
        data_buffer = []

        for i, new_point in enumerate(new_data_stream):
            data_buffer.append(new_point)

            if (i + 1) % update_frequency == 0 and len(data_buffer) > 1:
                # 准备增量数据
                buffer_array = np.array(data_buffer)
                new_states = buffer_array[:-1]
                new_next_states = buffer_array[1:]

                # 重新训练 (简化的增量学习)
                # 实际实现中可以使用更复杂的在线算法
                combined_states = np.vstack([
                    states * forgetting_factor,
                    new_states
                ])
                combined_next_states = np.vstack([
                    next_states * forgetting_factor,
                    new_next_states
                ])

                # 更新模型
                model.fit(combined_states, combined_next_states)

                # 评估性能
                test_error = model.reconstruction_error(new_states, new_next_states)
                update_history.append({
                    'step': i + 1,
                    'error': test_error,
                    'buffer_size': len(data_buffer)
                })

                print(f"Update {len(update_history)}: Error = {test_error:.4f}")

                # 更新训练数据 (保持固定大小的滑动窗口)
                states = combined_states[-1000:]  # 保持最近1000个样本
                next_states = combined_next_states[-1000:]

                # 清空缓冲区
                data_buffer = [new_point]  # 保留最后一个点作为下一段的起始

        return update_history

    def ensemble_learning(self, trajectory, n_models=5, diversity_method='bootstrap'):
        """
        集成学习
        训练多个Koopman模型并组合预测
        """
        states = trajectory[:-1]
        next_states = trajectory[1:]

        models = []

        for i in range(n_models):
            print(f"Training ensemble model {i+1}/{n_models}...")

            if diversity_method == 'bootstrap':
                # Bootstrap采样
                n_samples = len(states)
                indices = np.random.choice(n_samples, n_samples, replace=True)
                train_states = states[indices]
                train_next_states = next_states[indices]

                # 随机参数
                basis_types = ['polynomial', 'rbf', 'fourier']
                basis_type = np.random.choice(basis_types)

                if basis_type == 'polynomial':
                    basis_order = np.random.choice([2, 3, 4])
                    model = KoopmanOperator(basis_type=basis_type, basis_order=basis_order)
                else:
                    model = KoopmanOperator(basis_type=basis_type)

            elif diversity_method == 'random_features':
                # 随机特征子集
                train_states = states
                train_next_states = next_states

                # 随机正则化参数
                reg_param = 10**np.random.uniform(-6, -2)
                model = KoopmanOperator(regularization=reg_param)

            # 训练模型
            model.fit(train_states, train_next_states)
            models.append(model)

        self.models['ensemble'] = models

        return models

    def ensemble_predict(self, initial_state, n_steps, method='average'):
        """
        集成预测
        """
        if 'ensemble' not in self.models:
            raise ValueError("No ensemble models found. Run ensemble_learning first.")

        models = self.models['ensemble']
        predictions = []

        for model in models:
            pred = model.predict(initial_state, n_steps)
            predictions.append(pred)

        predictions = np.array(predictions)  # (n_models, n_steps+1, n_states)

        if method == 'average':
            ensemble_pred = np.mean(predictions, axis=0)
        elif method == 'weighted':
            # 基于模型性能的加权平均 (简化实现)
            weights = np.ones(len(models)) / len(models)
            ensemble_pred = np.average(predictions, axis=0, weights=weights)
        elif method == 'median':
            ensemble_pred = np.median(predictions, axis=0)

        return ensemble_pred, predictions

    def save_model(self, filename, model_type='best'):
        """保存模型"""
        if model_type == 'best' and self.best_model is not None:
            model = self.best_model
        elif model_type in self.models:
            model = self.models[model_type]
        else:
            raise ValueError(f"Model type '{model_type}' not found")

        with open(filename, 'wb') as f:
            pickle.dump(model, f)

        print(f"Model saved to {filename}")

    def load_model(self, filename):
        """加载模型"""
        with open(filename, 'rb') as f:
            model = pickle.load(f)

        self.best_model = model
        print(f"Model loaded from {filename}")

        return model

    def analyze_stability(self, model):
        """分析模型稳定性"""
        if model.eigenvalues is None:
            return None

        eigenvals = model.eigenvalues
        magnitudes = np.abs(eigenvals)

        # 稳定性分析
        stable_modes = np.sum(magnitudes <= 1.0)
        unstable_modes = np.sum(magnitudes > 1.0)

        # 主导模态
        dominant_indices = np.argsort(magnitudes)[-5:]  # 前5个主导模态
        dominant_eigenvals = eigenvals[dominant_indices]

        stability_info = {
            'total_modes': len(eigenvals),
            'stable_modes': stable_modes,
            'unstable_modes': unstable_modes,
            'max_magnitude': np.max(magnitudes),
            'dominant_eigenvalues': dominant_eigenvals,
            'stability_margin': 1.0 - np.max(magnitudes[magnitudes <= 1.0]) if stable_modes > 0 else None
        }

        return stability_info

def compare_models(trajectory, models_config):
    """
    比较不同Koopman模型的性能
    """
    states = trajectory[:-1]
    next_states = trajectory[1:]
    
    # 分割训练/测试数据
    train_states, test_states, train_next, test_next = train_test_split(
        states, next_states, test_size=0.2, random_state=42)
    
    results = {}
    
    for name, config in models_config.items():
        print(f"\nTesting {name}...")
        
        try:
            model = KoopmanOperator(**config)
            model.fit(train_states, train_next)
            
            # 重构误差
            recon_error = model.reconstruction_error(test_states, test_next)
            
            # 多步预测误差
            multistep_errors = []
            for horizon in [1, 5, 10, 20]:
                horizon_errors = []
                for i in range(min(50, len(test_states))):
                    try:
                        pred = model.predict(test_states[i], horizon)
                        if i + horizon < len(test_states):
                            true_traj = trajectory[-(len(test_states)-i):-(len(test_states)-i-horizon)]
                            error = np.mean(np.linalg.norm(pred[1:] - true_traj, axis=1))
                            horizon_errors.append(error)
                    except:
                        pass
                
                if horizon_errors:
                    multistep_errors.append(np.mean(horizon_errors))
                else:
                    multistep_errors.append(float('inf'))
            
            # 稳定性分析
            stability = KoopmanLearner().analyze_stability(model)
            
            results[name] = {
                'model': model,
                'reconstruction_error': recon_error,
                'multistep_errors': dict(zip([1, 5, 10, 20], multistep_errors)),
                'stability': stability,
                'n_basis_functions': model.n_basis_functions
            }
            
            print(f"  Reconstruction error: {recon_error:.4f}")
            print(f"  1-step error: {multistep_errors[0]:.4f}")
            print(f"  Stable modes: {stability['stable_modes'] if stability else 'N/A'}")
            
        except Exception as e:
            print(f"  Failed: {e}")
            results[name] = {'error': str(e)}
    
    return results

if __name__ == "__main__":
    # 测试学习算法
    from aircraft_dynamics import RigidBodyAircraft, sinusoidal_control
    
    # 生成数据
    aircraft = RigidBodyAircraft()
    trim_state, _ = aircraft.trim_condition()
    
    t, trajectory = aircraft.simulate(
        initial_state=trim_state,
        time_span=(0, 30),
        controls_func=sinusoidal_control,
        dt=0.1
    )
    
    print(f"Generated trajectory with {len(trajectory)} time steps")
    
    # 初始化学习器
    learner = KoopmanLearner()
    
    # 自适应基函数选择
    print("=== Adaptive Basis Selection ===")
    best_config, all_results = learner.adaptive_basis_selection(trajectory)
    
    # 网格搜索
    print("\n=== Grid Search ===")
    param_grid = {
        'basis_type': ['polynomial', 'rbf'],
        'basis_order': [2, 3],
        'regularization': [1e-6, 1e-4, 1e-2]
    }
    
    best_params, search_results = learner.grid_search(trajectory, param_grid, cv_folds=3)
    
    # 集成学习
    print("\n=== Ensemble Learning ===")
    ensemble_models = learner.ensemble_learning(trajectory, n_models=3)
    
    # 测试集成预测
    test_state = trajectory[200]
    ensemble_pred, individual_preds = learner.ensemble_predict(test_state, 20)
    
    print(f"Ensemble prediction shape: {ensemble_pred.shape}")
    print(f"Individual predictions shape: {individual_preds.shape}")
    
    # 模型比较
    print("\n=== Model Comparison ===")
    models_config = {
        'poly_order2': {'basis_type': 'polynomial', 'basis_order': 2},
        'poly_order3': {'basis_type': 'polynomial', 'basis_order': 3},
        'rbf': {'basis_type': 'rbf'},
        'fourier': {'basis_type': 'fourier', 'basis_order': 3}
    }
    
    comparison_results = compare_models(trajectory, models_config)
    
    # 打印比较结果
    print("\nComparison Summary:")
    for name, result in comparison_results.items():
        if 'error' not in result:
            print(f"{name}:")
            print(f"  Reconstruction error: {result['reconstruction_error']:.4f}")
            print(f"  Basis functions: {result['n_basis_functions']}")
            if result['stability']:
                print(f"  Stable modes: {result['stability']['stable_modes']}")
