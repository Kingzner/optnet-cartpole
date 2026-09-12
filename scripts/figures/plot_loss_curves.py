import os
import sys

# Allow this script to be run directly from the repository root, e.g.
#     python scripts/figures/plot_loss_curves.py
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd
import matplotlib.pyplot as plt
import argparse

# 解析命令行参数
parser = argparse.ArgumentParser(description='Plot training results')
parser.add_argument('workdir', type=str, help='Directory containing training results')
parser.add_argument('--output-dir', type=str, default='figures', 
                    help='Output directory for plots')
parser.add_argument('--filename', type=str, default='loss_curves', 
                    help='Output filename (without extension)')
args = parser.parse_args()

workdir = args.workdir

# 确保输出目录存在
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# 读取 CSV 文件，支持两种命名格式
train_csv = os.path.join(workdir, 'train.csv')
val_csv = os.path.join(workdir, 'val.csv')
test_csv = os.path.join(workdir, 'test.csv')

# 如果文件不存在，尝试使用另一种命名格式
if not os.path.exists(train_csv):
    train_csv = os.path.join(workdir, 'train_loss.csv')
if not os.path.exists(val_csv):
    val_csv = os.path.join(workdir, 'val_loss.csv')
if not os.path.exists(test_csv):
    test_csv = os.path.join(workdir, 'test_loss.csv')

def plot_loss_curves():
    """绘制损失曲线"""
    # 读取数据
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    
    # 创建图表
    plt.figure(figsize=(10, 6))
    
    # 绘制训练损失
    plt.plot(train_df['epoch'], train_df['loss'], label='Train Loss', color='blue')
    
    # 绘制验证损失
    plt.plot(val_df['epoch'], val_df['loss'], label='Validation Loss', color='green')
    
    # 绘制测试损失
    plt.plot(test_df['epoch'], test_df['loss'], label='Test Loss', color='red')
    
    # 添加标题和标签
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # 保存图表到 plots 目录
    output_file = os.path.join(args.output_dir, f'{args.filename}.png')
    plt.savefig(output_file)
    print(f'Loss curves saved to: {output_file}')
    
    # 显示图表
    plt.show()

def plot_metrics():
    """绘制指标图表（如果有）"""
    # 这里可以添加其他指标的绘制逻辑
    # 例如，如果有准确率、F1分数等指标
    pass

if __name__ == '__main__':
    # 检查文件是否存在
    if not os.path.exists(train_csv):
        print(f'Error: {train_csv} not found!')
    elif not os.path.exists(val_csv):
        print(f'Error: {val_csv} not found!')
    elif not os.path.exists(test_csv):
        print(f'Error: {test_csv} not found!')
    else:
        plot_loss_curves()
        plot_metrics()
