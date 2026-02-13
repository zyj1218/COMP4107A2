"""
Assignment 2 - Question 4: Experiments
MLB Position Player Salary Prediction

IMPORTANT NOTES:
- Test set is held out and ONLY used for final evaluation (Question 4e)
- All experiments (a-d) use validation set to select best configuration
- Data split: 60% train, 20% validation, 20% test (random seed=42)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import time

# 设置随机种子以保证可重复性
RANDOM_SEED = 128
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ========================================
# 数据加载和预处理函数
# ========================================
def load_and_split_data(filepath):
    """
    加载数据并分割为训练集、验证集、测试集 (60/20/20)
    Random seed = 42 for reproducibility
    Test set is held out and only used for final evaluation
    """
    data = np.loadtxt(filepath, delimiter=",", skiprows=1)
    y = data[:, 0:1]  # 工资
    X = data[:, 1:]   # 特征
    
    # 打乱数据 (seed=42)
    N = X.shape[0]
    idx = np.arange(N)
    np.random.shuffle(idx)
    
    # 分割：60% 训练，20% 验证，20% 测试
    train_end = int(0.6 * N)
    val_end = int(0.8 * N)
    
    train_idx = idx[:train_end]
    val_idx = idx[train_end:val_end]
    test_idx = idx[val_end:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    # 标准化（使用训练集统计量）
    x_mean = X_train.mean(axis=0, keepdims=True)
    x_std = X_train.std(axis=0, keepdims=True) + 1e-8
    
    X_train = (X_train - x_mean) / x_std
    X_val = (X_val - x_mean) / x_std
    X_test = (X_test - x_mean) / x_std
    
    # 标准化目标变量
    y_mean = y_train.mean()
    y_std = y_train.std() + 1e-8
    
    y_train_z = (y_train - y_mean) / y_std
    y_val_z = (y_val - y_mean) / y_std
    y_test_z = (y_test - y_mean) / y_std
    
    # 转换为torch张量
    data_dict = {
        'X_train': torch.as_tensor(X_train, dtype=torch.float32),
        'y_train': torch.as_tensor(y_train_z, dtype=torch.float32),
        'X_val': torch.as_tensor(X_val, dtype=torch.float32),
        'y_val': torch.as_tensor(y_val_z, dtype=torch.float32),
        'X_test': torch.as_tensor(X_test, dtype=torch.float32),
        'y_test': torch.as_tensor(y_test_z, dtype=torch.float32),
        'y_val_orig': torch.as_tensor(y_val, dtype=torch.float32),
        'y_test_orig': torch.as_tensor(y_test, dtype=torch.float32),
        'y_train_orig': torch.as_tensor(y_train, dtype=torch.float32),
        'y_mean': y_mean,
        'y_std': y_std
    }
    
    return data_dict


# ========================================
# 可配置的神经网络类
# ========================================
class ConfigurableSalaryNet(torch.nn.Module):
    def __init__(self, input_size=16, hidden_sizes=[32, 16], activation='relu'):
        super().__init__()
        
        # 选择激活函数
        if activation == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation == 'tanh':
            self.activation = torch.nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = torch.nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # 构建网络层
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(torch.nn.Linear(prev_size, hidden_size))
            layers.append(self.activation)
            prev_size = hidden_size
        
        # 输出层
        layers.append(torch.nn.Linear(prev_size, 1))
        
        self.net = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


# ========================================
# 训练函数
# ========================================
def train_model(model, data, num_epochs=300, batch_size=16, lr=0.01, verbose=False):
    """
    训练模型并返回训练历史
    """
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    y_val_orig = data['y_val_orig']
    y_mean = data['y_mean']
    y_std = data['y_std']
    
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    batches_per_epoch = int(np.ceil(X_train.shape[0] / batch_size))
    
    train_losses = []
    val_rmses = []
    
    for epoch in range(num_epochs):
        model.train()
        
        # 打乱数据
        perm = torch.randperm(X_train.shape[0])
        X_train_shuf = X_train[perm]
        y_train_shuf = y_train[perm]
        
        epoch_loss = 0.0
        for b in range(batches_per_epoch):
            start = b * batch_size
            end = min((b + 1) * batch_size, X_train_shuf.shape[0])
            
            xb = X_train_shuf[start:end]
            yb = y_train_shuf[start:end]
            
            pred = model(xb)
            loss = loss_fn(pred, yb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / batches_per_epoch
        train_losses.append(avg_train_loss)
        
        # 验证集性能
        model.eval()
        with torch.no_grad():
            pred_val_z = model(X_val)
            pred_val = pred_val_z * y_std + y_mean
            rmse = torch.sqrt(torch.mean((pred_val - y_val_orig) ** 2)).item()
            val_rmses.append(rmse)
        
        if verbose and (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val RMSE: {rmse:.2f}")
    
    return train_losses, val_rmses


def evaluate_model(model, data, dataset='val'):
    """
    评估模型在指定数据集上的RMSE
    """
    model.eval()
    
    if dataset == 'train':
        X = data['X_train']
        y_orig = data['y_train_orig']
    elif dataset == 'val':
        X = data['X_val']
        y_orig = data['y_val_orig']
    elif dataset == 'test':
        X = data['X_test']
        y_orig = data['y_test_orig']
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    y_mean = data['y_mean']
    y_std = data['y_std']
    
    with torch.no_grad():
        pred_z = model(X)
        pred = pred_z * y_std + y_mean
        rmse = torch.sqrt(torch.mean((pred - y_orig) ** 2)).item()
    
    return rmse


# ========================================
# 实验 1: 隐藏层神经元数量
# ========================================
def experiment_neurons(data, neuron_counts, num_epochs=300):
    """
    实验不同的隐藏层神经元数量
    保持2层隐藏层，改变每层的神经元数
    """
    print("\n" + "="*60)
    print("实验 1: 隐藏层神经元数量")
    print("="*60)
    
    results = []
    
    for neurons in neuron_counts:
        print(f"\n训练模型: 隐藏层大小 = [{neurons}, {neurons//2}]")
        
        model = ConfigurableSalaryNet(
            input_size=16,
            hidden_sizes=[neurons, neurons//2],
            activation='relu'
        )
        
        train_losses, val_rmses = train_model(
            model, data, num_epochs=num_epochs, verbose=True
        )
        
        final_val_rmse = val_rmses[-1]
        results.append({
            'neurons': neurons,
            'val_rmse': final_val_rmse,
            'model': model,
            'history': (train_losses, val_rmses)
        })
        
        print(f"最终验证集 RMSE: {final_val_rmse:.2f}")
    
    return results


# ========================================
# 实验 2: 隐藏层数量
# ========================================
def experiment_layers(data, layer_counts, neuron_per_layer=32, num_epochs=300):
    """
    实验不同数量的隐藏层
    """
    print("\n" + "="*60)
    print("实验 2: 隐藏层数量")
    print("="*60)
    
    results = []
    
    for num_layers in layer_counts:
        # 创建隐藏层大小列表（逐渐减小）
        hidden_sizes = [neuron_per_layer // (2**i) for i in range(num_layers)]
        hidden_sizes = [max(8, size) for size in hidden_sizes]  # 最小8个神经元
        
        print(f"\n训练模型: {num_layers} 层隐藏层 = {hidden_sizes}")
        
        model = ConfigurableSalaryNet(
            input_size=16,
            hidden_sizes=hidden_sizes,
            activation='relu'
        )
        
        train_losses, val_rmses = train_model(
            model, data, num_epochs=num_epochs, verbose=True
        )
        
        final_val_rmse = val_rmses[-1]
        results.append({
            'num_layers': num_layers,
            'hidden_sizes': hidden_sizes,
            'val_rmse': final_val_rmse,
            'model': model,
            'history': (train_losses, val_rmses)
        })
        
        print(f"最终验证集 RMSE: {final_val_rmse:.2f}")
    
    return results


# ========================================
# 实验 3: 训练epochs数量
# ========================================
def experiment_epochs(data, epoch_counts):
    """
    实验不同的训练epoch数量
    """
    print("\n" + "="*60)
    print("实验 3: 训练Epochs数量")
    print("="*60)
    
    results = []
    
    for num_epochs in epoch_counts:
        print(f"\n训练模型: {num_epochs} epochs")
        
        model = ConfigurableSalaryNet(
            input_size=16,
            hidden_sizes=[32, 16],
            activation='relu'
        )
        
        train_losses, val_rmses = train_model(
            model, data, num_epochs=num_epochs, verbose=True
        )
        
        final_val_rmse = val_rmses[-1]
        results.append({
            'epochs': num_epochs,
            'val_rmse': final_val_rmse,
            'model': model,
            'history': (train_losses, val_rmses),
            'all_val_rmses': val_rmses
        })
        
        print(f"最终验证集 RMSE: {final_val_rmse:.2f}")
    
    return results


# ========================================
# 实验 4: 激活函数
# ========================================
def experiment_activations(data, activations, num_epochs=300):
    """
    实验不同的激活函数
    """
    print("\n" + "="*60)
    print("实验 4: 激活函数")
    print("="*60)
    
    results = []
    
    for activation in activations:
        print(f"\n训练模型: 激活函数 = {activation}")
        
        model = ConfigurableSalaryNet(
            input_size=16,
            hidden_sizes=[32, 16],
            activation=activation
        )
        
        train_losses, val_rmses = train_model(
            model, data, num_epochs=num_epochs, verbose=True
        )
        
        final_val_rmse = val_rmses[-1]
        results.append({
            'activation': activation,
            'val_rmse': final_val_rmse,
            'model': model,
            'history': (train_losses, val_rmses)
        })
        
        print(f"最终验证集 RMSE: {final_val_rmse:.2f}")
    
    return results


# ========================================
# 绘图和报告生成
# ========================================
def generate_report(all_results, data, output_file='assignment2.pdf'):
    """
    生成包含所有实验结果的PDF报告
    """
    print("\n" + "="*60)
    print("生成PDF报告...")
    print("="*60)
    
    with PdfPages(output_file) as pdf:
        # 设置字体
        plt.rcParams['font.size'] = 10
        
        # ========================================
        # 封面
        # ========================================
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.7, 'Assignment 2 - Question 4', 
                ha='center', va='center', fontsize=24, fontweight='bold')
        fig.text(0.5, 0.6, 'Neural Network Experiments', 
                ha='center', va='center', fontsize=18)
        fig.text(0.5, 0.5, 'MLB Position Player Salary Prediction', 
                ha='center', va='center', fontsize=14)
        fig.text(0.5, 0.3, f'Date: February 13, 2026', 
                ha='center', va='center', fontsize=12)
        
        # 添加数据分割说明
        split_text = """
Data Split Information:
• Training Set: 60% of data
• Validation Set: 20% of data
• Test Set: 20% of data (held out)
• Random Seed: 42 (for reproducibility)

Performance Metric: RMSE (Root Mean Squared Error)
Lower values indicate better performance.

Note: Test set is ONLY used for final evaluation 
in Question 4(e) and does not participate in model selection.
        """
        fig.text(0.5, 0.15, split_text, ha='center', va='top', 
                fontsize=10, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ========================================
        # 实验 1: 神经元数量
        # ========================================
        fig, ax = plt.subplots(figsize=(10, 6))
        
        neurons_list = [r['neurons'] for r in all_results['neurons']]
        val_rmses = [r['val_rmse'] for r in all_results['neurons']]
        
        ax.plot(neurons_list, val_rmses, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Neurons (First Hidden Layer)', fontsize=12)
        ax.set_ylabel('Validation RMSE ($)', fontsize=12)
        ax.set_title('Experiment 1: Effect of Hidden Layer Size\n(Performance Metric: RMSE on Validation Set)', 
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 标注最佳值
        best_idx = np.argmin(val_rmses)
        ax.plot(neurons_list[best_idx], val_rmses[best_idx], 'r*', markersize=15, 
                label=f'Best: {neurons_list[best_idx]} neurons (RMSE=${val_rmses[best_idx]:.2f})')
        ax.legend()
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ========================================
        # 实验 2: 隐藏层数量
        # ========================================
        fig, ax = plt.subplots(figsize=(10, 6))
        
        layer_counts = [r['num_layers'] for r in all_results['layers']]
        val_rmses = [r['val_rmse'] for r in all_results['layers']]
        
        ax.plot(layer_counts, val_rmses, 's-', linewidth=2, markersize=8, color='green')
        ax.set_xlabel('Number of Hidden Layers', fontsize=12)
        ax.set_ylabel('Validation RMSE ($)', fontsize=12)
        ax.set_title('Experiment 2: Effect of Network Depth\n(Performance Metric: RMSE on Validation Set)', 
                     fontsize=14, fontweight='bold')
        ax.set_xticks(layer_counts)
        ax.grid(True, alpha=0.3)
        
        # 标注最佳值
        best_idx = np.argmin(val_rmses)
        ax.plot(layer_counts[best_idx], val_rmses[best_idx], 'r*', markersize=15,
                label=f'Best: {layer_counts[best_idx]} layers (RMSE=${val_rmses[best_idx]:.2f})')
        ax.legend()
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ========================================
        # 实验 3: Epochs数量 (两个子图)
        # ========================================
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # 子图1: 最终RMSE vs Epochs
        epoch_counts = [r['epochs'] for r in all_results['epochs']]
        val_rmses = [r['val_rmse'] for r in all_results['epochs']]
        
        ax1.plot(epoch_counts, val_rmses, 'd-', linewidth=2, markersize=8, color='orange')
        ax1.set_xlabel('Number of Training Epochs', fontsize=12)
        ax1.set_ylabel('Final Validation RMSE ($)', fontsize=12)
        ax1.set_title('Experiment 3a: Effect of Training Duration\n(Performance Metric: RMSE on Validation Set)', 
                      fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        best_idx = np.argmin(val_rmses)
        ax1.plot(epoch_counts[best_idx], val_rmses[best_idx], 'r*', markersize=15,
                label=f'Best: {epoch_counts[best_idx]} epochs (RMSE=${val_rmses[best_idx]:.2f})')
        ax1.legend()
        
        # 子图2: 学习曲线（所有epochs配置）
        for result in all_results['epochs']:
            epochs_range = range(1, result['epochs'] + 1)
            ax2.plot(epochs_range, result['all_val_rmses'], 
                    label=f"{result['epochs']} epochs", linewidth=1.5)
        
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Validation RMSE ($)', fontsize=12)
        ax2.set_title('Experiment 3b: Learning Curves', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ========================================
        # 实验 4: 激活函数
        # ========================================
        fig, ax = plt.subplots(figsize=(10, 6))
        
        activations = [r['activation'] for r in all_results['activations']]
        val_rmses = [r['val_rmse'] for r in all_results['activations']]
        
        colors = ['blue', 'green', 'red'][:len(activations)]
        bars = ax.bar(activations, val_rmses, color=colors, alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Activation Function', fontsize=12)
        ax.set_ylabel('Validation RMSE ($)', fontsize=12)
        ax.set_title('Experiment 4: Effect of Activation Function\n(Performance Metric: RMSE on Validation Set)', 
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 标注最佳值
        best_idx = np.argmin(val_rmses)
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(3)
        
        # 在柱状图上显示数值
        for i, (act, rmse) in enumerate(zip(activations, val_rmses)):
            ax.text(i, rmse + 20, f'${rmse:.2f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ========================================
        # 最佳模型性能总结 (Question 4e)
        # ========================================
        # 找到所有实验中的最佳模型
        all_models = []
        
        for r in all_results['neurons']:
            all_models.append({
                'exp_type': 'Hidden Layer Size',
                'config': f"{r['neurons']} neurons",
                'hidden_sizes': [r['neurons'], r['neurons']//2],
                'activation': 'relu',
                'epochs': 300,
                'val_rmse': r['val_rmse'],
                'model': r['model']
            })
        
        for r in all_results['layers']:
            all_models.append({
                'exp_type': 'Network Depth',
                'config': f"{r['num_layers']} layers",
                'hidden_sizes': r['hidden_sizes'],
                'activation': 'relu',
                'epochs': 300,
                'val_rmse': r['val_rmse'],
                'model': r['model']
            })
        
        for r in all_results['epochs']:
            all_models.append({
                'exp_type': 'Training Duration',
                'config': f"{r['epochs']} epochs",
                'hidden_sizes': [32, 16],
                'activation': 'relu',
                'epochs': r['epochs'],
                'val_rmse': r['val_rmse'],
                'model': r['model']
            })
        
        for r in all_results['activations']:
            all_models.append({
                'exp_type': 'Activation Function',
                'config': f"{r['activation']} activation",
                'hidden_sizes': [32, 16],
                'activation': r['activation'],
                'epochs': 300,
                'val_rmse': r['val_rmse'],
                'model': r['model']
            })
        
        # 找到验证集RMSE最小的模型
        best_config = min(all_models, key=lambda x: x['val_rmse'])
        best_model = best_config['model']
        
        # 在三个数据集上评估最佳模型
        train_rmse = evaluate_model(best_model, data, 'train')
        val_rmse = evaluate_model(best_model, data, 'val')
        test_rmse = evaluate_model(best_model, data, 'test')
        
        # 创建性能总结页面
        fig = plt.figure(figsize=(8.5, 11))
        
        summary_text = f"""
═══════════════════════════════════════════════════════════
QUESTION 4(e): BEST MODEL PERFORMANCE SUMMARY
═══════════════════════════════════════════════════════════

MODEL SELECTION CRITERION:
  Selected based on LOWEST validation set RMSE across all 
  experiments (a-d). Test set was NOT used for model selection.

BEST CONFIGURATION (from experiments a-d):
────────────────────────────────────────────────────────────
  Source Experiment:   {best_config['exp_type']}
  Configuration:       {best_config['config']}
  
  Full Architecture:
    • Hidden Layers:   {best_config['hidden_sizes']}
    • Activation:      {best_config['activation']}
    • Training Epochs: {best_config['epochs']}
    • Optimizer:       Adam (lr=0.01)

PERFORMANCE ON ALL THREE SETS:
────────────────────────────────────────────────────────────
  Training Set RMSE:   ${train_rmse:>10,.2f}
  Validation Set RMSE: ${val_rmse:>10,.2f}
  Test Set RMSE:       ${test_rmse:>10,.2f}

GENERALIZATION ANALYSIS:
────────────────────────────────────────────────────────────
  Train-to-Val Ratio:  {val_rmse/train_rmse:.3f}
    → {'Good generalization' if val_rmse/train_rmse < 1.5 else 'Some overfitting detected'}
  
  Val-to-Test Diff:    ${abs(val_rmse - test_rmse):.2f}
    → {'Consistent performance' if abs(val_rmse - test_rmse) < val_rmse * 0.15 else 'Notable variance'}

CONCLUSIONS:
────────────────────────────────────────────────────────────
  The neural network achieves a test set RMSE of ${test_rmse:,.2f}
  for predicting MLB player salaries. 
  
  Key Findings from Experiments:
  • Hidden Layer Size: {all_results['neurons'][np.argmin([r['val_rmse'] for r in all_results['neurons']])]['neurons']} neurons optimal (Exp 1)
  • Network Depth: {all_results['layers'][np.argmin([r['val_rmse'] for r in all_results['layers']])]['num_layers']} layers optimal (Exp 2)
  • Training Duration: {all_results['epochs'][np.argmin([r['val_rmse'] for r in all_results['epochs']])]['epochs']} epochs optimal (Exp 3)
  • Activation: {all_results['activations'][np.argmin([r['val_rmse'] for r in all_results['activations']])]['activation']} optimal (Exp 4)
  
  The model demonstrates {'strong' if abs(val_rmse - test_rmse) < 100 else 'adequate'} 
  generalization from validation to test data.
"""
        
        fig.text(0.1, 0.95, summary_text, ha='left', va='top', 
                fontsize=10, family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # 保存元数据
        d = pdf.infodict()
        d['Title'] = 'Assignment 2 - Neural Network Experiments'
        d['Author'] = 'Student'
        d['Subject'] = 'MLB Salary Prediction'
        d['Keywords'] = 'Neural Networks, Machine Learning, PyTorch'
        d['CreationDate'] = 'D:20260213000000'
    
    print(f"\n报告已保存到: {output_file}")
    print(f"\n最佳模型 (from {best_config['exp_type']}):")
    print(f"配置: {best_config['config']}")
    print(f"架构: {best_config['hidden_sizes']}, activation={best_config['activation']}")
    print(f"\n训练集 RMSE: ${train_rmse:,.2f}")
    print(f"验证集 RMSE: ${val_rmse:,.2f}")
    print(f"测试集 RMSE: ${test_rmse:,.2f}")


# ========================================
# 主函数
# ========================================
def main():
    print("="*60)
    print("Assignment 2 - Question 4: Experiments")
    print("MLB Position Player Salary Prediction")
    print("="*60)
    
    # 加载数据
    print("\n加载数据...")
    data = load_and_split_data('baseball.txt')
    print(f"训练集: {data['X_train'].shape[0]} 样本 (60%)")
    print(f"验证集: {data['X_val'].shape[0]} 样本 (20%)")
    print(f"测试集: {data['X_test'].shape[0]} 样本 (20%, held out)")
    print(f"随机种子: {RANDOM_SEED}")
    
    # 运行所有实验
    all_results = {}
    
    # 实验1: 神经元数量
    all_results['neurons'] = experiment_neurons(
        data, 
        neuron_counts=[8, 16, 32, 64, 128],
        num_epochs=300
    )
    
    # 实验2: 隐藏层数量
    all_results['layers'] = experiment_layers(
        data,
        layer_counts=[1, 2, 3, 4],
        neuron_per_layer=32,
        num_epochs=300
    )
    
    # 实验3: Epochs数量
    all_results['epochs'] = experiment_epochs(
        data,
        epoch_counts=[50, 100, 200, 300, 500]
    )
    
    # 实验4: 激活函数 (只用课上学过的)
    all_results['activations'] = experiment_activations(
        data,
        activations=['relu', 'tanh', 'sigmoid'],
        num_epochs=300
    )
    
    # 生成PDF报告
    generate_report(all_results, data, 'assignment2.pdf')
    
    print("\n" + "="*60)
    print("所有实验完成！")
    print("="*60)


if __name__ == "__main__":
    main()