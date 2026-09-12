import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# 解析命令行参数
parser = argparse.ArgumentParser(description='Plot OptNet vs Baseline comparison')
parser.add_argument('--optnet-dir', type=str, default='work_cartpole_filtered', 
                    help='Directory containing OptNet training results')
parser.add_argument('--baseline-dir', type=str, default='work_cartpole_baseline', 
                    help='Directory containing Baseline MLP training results')
parser.add_argument('--output-dir', type=str, default='plots', 
                    help='Output directory for the comparison plot')
args = parser.parse_args()

# 确保输出目录存在
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

def load_loss_data(dir_path):
    """加载损失数据"""
    train_csv = os.path.join(dir_path, 'train.csv')
    val_csv = os.path.join(dir_path, 'val.csv')
    test_csv = os.path.join(dir_path, 'test.csv')
    
    # 如果文件不存在，尝试使用另一种命名格式
    if not os.path.exists(train_csv):
        train_csv = os.path.join(dir_path, 'train_loss.csv')
    if not os.path.exists(val_csv):
        val_csv = os.path.join(dir_path, 'val_loss.csv')
    if not os.path.exists(test_csv):
        test_csv = os.path.join(dir_path, 'test_loss.csv')
    
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    
    return train_df, val_df, test_df

def plot_comparison():
    """绘制 OptNet vs Baseline 对比图"""
    # 加载数据
    optnet_train, optnet_val, optnet_test = load_loss_data(args.optnet_dir)
    baseline_train, baseline_val, baseline_test = load_loss_data(args.baseline_dir)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 使用配色方案 - 类似示例中的蓝色和橙色
    colors = {
        'optnet_train': '#1f77b4',      # 深蓝色 - 实线
        'optnet_test': '#1f77b4',       # 深蓝色 - 虚线
        'baseline_train': '#ff7f0e',    # 橙色 - 实线
        'baseline_test': '#ff7f0e'      # 橙色 - 虚线
    }
    
    # 绘制 OptNet（线性尺度）
    ax.plot(optnet_train['epoch'], optnet_train['loss'], 
                label='OptNet Train', color=colors['optnet_train'],
                linewidth=2, linestyle='-')
    ax.plot(optnet_test['epoch'], optnet_test['loss'], 
                label='OptNet Test', color=colors['optnet_test'],
                linewidth=2, linestyle='--')
    
    # 绘制 Baseline MLP（线性尺度）
    ax.plot(baseline_train['epoch'], baseline_train['loss'], 
                label='MLP Baseline Train', color=colors['baseline_train'],
                linewidth=2, linestyle='-')
    ax.plot(baseline_test['epoch'], baseline_test['loss'], 
                label='MLP Baseline Test', color=colors['baseline_test'],
                linewidth=2, linestyle='--')
    
    # 配置图表样式
    ax.set_title('OptNet vs MLP Baseline', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.grid(True, which='major', linestyle='--', alpha=0.7, zorder=0)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # 保存图表
    output_file = os.path.join(args.output_dir, 'optnet_vs_baseline.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f'Comparison plot saved to: {output_file}')
    
    # 显示图表
    plt.show()

if __name__ == '__main__':
    # 检查目录是否存在
    if not os.path.exists(args.optnet_dir):
        print(f'Error: {args.optnet_dir} not found!')
    elif not os.path.exists(args.baseline_dir):
        print(f'Error: {args.baseline_dir} not found!')
    else:
        plot_comparison()
