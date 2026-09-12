import os
import sys

# Allow this script to be run directly from the repository root, e.g.
#     python scripts/figures/plot_dataset_positions.py
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "data"
OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    X = torch.load(os.path.join(DATA_DIR, "features.pt"))
    y = torch.load(os.path.join(DATA_DIR, "labels.pt"))
    return X, y

def plot_all_trajectories(X, limit=1.0):
    """
    绘制所有轨迹，展示它们都在约束范围内
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    t = np.arange(X.shape[1]) * 0.01  # dt=0.01s, 500 steps = 5s
    
    # === Left: All trajectories overlay ===
    ax1 = axes[0]
    
    # 绘制约束边界
    ax1.axhline(y=limit, color='red', linestyle='--', linewidth=2.5, label=f'Constraint boundary (±{limit}m)')
    ax1.axhline(y=-limit, color='red', linestyle='--', linewidth=2.5)
    
    # 填充允许区域
    ax1.fill_between(t, -limit, limit, color='red', alpha=0.1, label='Allowed region')
    
    # 绘制所有轨迹
    for i in range(X.shape[0]):
        x = X[i, :, 0].numpy()
        ax1.plot(t, x, color='#1f77b4', linewidth=0.8, alpha=0.15)
    
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Cart Position (m)', fontsize=12)
    ax1.set_title(f'All {X.shape[0]} Trajectories - Position Constraints', fontsize=13)
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_xlim(0, 5)
    ax1.set_ylim(-1.2, 1.2)
    
    # === Right: Histogram of maximum positions ===
    ax2 = axes[1]
    
    # 计算每条轨迹的最大绝对位置
    max_positions = []
    for i in range(X.shape[0]):
        x = X[i, :, 0].numpy()
        max_pos = np.max(np.abs(x))
        max_positions.append(max_pos)
    
    max_positions = np.array(max_positions)
    
    # 统计超过约束的数量
    violations = np.sum(max_positions > limit)
    within_limit = np.sum(max_positions <= limit)
    
    # 绘制直方图
    n, bins, patches = ax2.hist(max_positions, bins=50, color='#1f77b4', alpha=0.7, edgecolor='black')
    
    # 标记约束边界
    ax2.axvline(x=limit, color='red', linestyle='--', linewidth=2.5, label=f'Constraint limit ({limit}m)')
    
    # 高亮超过约束的部分
    for i, patch in enumerate(patches):
        if bins[i] > limit:
            patch.set_facecolor('#ff7f0e')
    
    ax2.set_xlabel('Maximum Absolute Position (m)', fontsize=12)
    ax2.set_ylabel('Number of Trajectories', fontsize=12)
    ax2.set_title('Distribution of Maximum Trajectory Positions', fontsize=13)
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_xlim(0, 1.3)
    
    plt.suptitle('Verification: All Trajectories Respect Position Constraints', fontsize=14, y=0.98)
    
    # 添加统计信息
    stats_text = f'Total trajectories: {X.shape[0]}\nWithin limit: {within_limit} ({100*within_limit/X.shape[0]:.1f}%)\nExceeding limit: {violations} ({100*violations/X.shape[0]:.1f}%)\nMax violation: {np.max(max_positions)-limit:.4f}m'
    plt.figtext(0.92, 0.15, stats_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    save_path = os.path.join(OUTPUT_DIR, 'all_trajectories_constraints.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    plt.show()
    
    # 打印详细统计
    print("\n=== Constraint Compliance Statistics ===")
    print(f"Total trajectories: {X.shape[0]}")
    print(f"Within limit (≤{limit}m): {within_limit} ({100*within_limit/X.shape[0]:.1f}%)")
    print(f"Exceeding limit (> {limit}m): {violations} ({100*violations/X.shape[0]:.1f}%)")
    print(f"Maximum violation: {np.max(max_positions)-limit:.4f}m")
    print(f"Mean max position: {np.mean(max_positions):.4f}m")
    print(f"Median max position: {np.median(max_positions):.4f}m")

def main():
    print("Loading data...")
    X, y = load_data()
    print(f"Loaded {X.shape[0]} trajectories")
    
    print("\nGenerating visualization of all trajectories with constraints...")
    plot_all_trajectories(X, limit=1.0)
    
    print("\n[OK] Plot generated successfully!")

if __name__ == "__main__":
    main()
