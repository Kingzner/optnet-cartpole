import torch
import numpy as np
import matplotlib.pyplot as plt
import os

DATA_DIR = "cartpole_data"
OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    X = torch.load(os.path.join(DATA_DIR, "features.pt"))
    y = torch.load(os.path.join(DATA_DIR, "labels.pt"))
    return X, y

def plot_all_control_forces(y, limit=10.0):
    """
    绘制所有轨迹的控制力，展示它们都在约束范围内
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    t = np.arange(y.shape[1]) * 0.01  # dt=0.01s, 500 steps = 5s
    
    # === Left: All control force trajectories overlay ===
    ax1 = axes[0]
    
    # 绘制约束边界
    ax1.axhline(y=limit, color='red', linestyle='--', linewidth=2.5, label=f'Constraint boundary (±{limit}N)')
    ax1.axhline(y=-limit, color='red', linestyle='--', linewidth=2.5)
    
    # 填充允许区域
    ax1.fill_between(t, -limit, limit, color='red', alpha=0.1, label='Allowed region')
    
    # 绘制所有轨迹的控制力
    for i in range(y.shape[0]):
        u = y[i, :, 0].numpy()
        ax1.plot(t, u, color='#2ca02c', linewidth=0.8, alpha=0.15)
    
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Control Force (N)', fontsize=12)
    ax1.set_title(f'All {y.shape[0]} Trajectories - Control Force Constraints', fontsize=13)
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_xlim(0, 5)
    ax1.set_ylim(-12, 12)
    
    # === Right: Histogram of maximum control forces ===
    ax2 = axes[1]
    
    # 计算每条轨迹的最大绝对控制力
    max_forces = []
    for i in range(y.shape[0]):
        u = y[i, :, 0].numpy()
        max_force = np.max(np.abs(u))
        max_forces.append(max_force)
    
    max_forces = np.array(max_forces)
    
    # 统计超过约束的数量
    violations = np.sum(max_forces > limit)
    within_limit = np.sum(max_forces <= limit)
    
    # 绘制直方图
    n, bins, patches = ax2.hist(max_forces, bins=50, color='#2ca02c', alpha=0.7, edgecolor='black')
    
    # 标记约束边界
    ax2.axvline(x=limit, color='red', linestyle='--', linewidth=2.5, label=f'Constraint limit ({limit}N)')
    
    # 高亮超过约束的部分
    for i, patch in enumerate(patches):
        if bins[i] > limit:
            patch.set_facecolor('#ff7f0e')
    
    ax2.set_xlabel('Maximum Absolute Control Force (N)', fontsize=12)
    ax2.set_ylabel('Number of Trajectories', fontsize=12)
    ax2.set_title('Distribution of Maximum Control Forces', fontsize=13)
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_xlim(0, 13)
    
    plt.suptitle('Verification: All Trajectories Respect Control Force Constraints (±10N)', fontsize=14, y=0.98)
    
    # 添加统计信息
    stats_text = f'Total trajectories: {y.shape[0]}\nWithin limit: {within_limit} ({100*within_limit/y.shape[0]:.1f}%)\nExceeding limit: {violations} ({100*violations/y.shape[0]:.1f}%)\nMax violation: {np.max(max_forces)-limit:.4f}N'
    plt.figtext(0.92, 0.15, stats_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    save_path = os.path.join(OUTPUT_DIR, 'all_control_forces_constraints.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    plt.show()
    
    # 打印详细统计
    print("\n=== Control Force Constraint Compliance Statistics ===")
    print(f"Total trajectories: {y.shape[0]}")
    print(f"Within limit (≤{limit}N): {within_limit} ({100*within_limit/y.shape[0]:.1f}%)")
    print(f"Exceeding limit (> {limit}N): {violations} ({100*violations/y.shape[0]:.1f}%)")
    print(f"Maximum violation: {np.max(max_forces)-limit:.4f}N")
    print(f"Mean max force: {np.mean(max_forces):.4f}N")
    print(f"Median max force: {np.median(max_forces):.4f}N")

def main():
    print("Loading data...")
    X, y = load_data()
    print(f"Loaded {y.shape[0]} trajectories")
    
    print("\nGenerating visualization of all control forces with constraints...")
    plot_all_control_forces(y, limit=10.0)
    
    print("\n[OK] Plot generated successfully!")

if __name__ == "__main__":
    main()
