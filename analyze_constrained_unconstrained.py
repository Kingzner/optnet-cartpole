import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 读取有约束结果（N=10，cart position limit = ±1.0 m）
constrained_df_val = pd.read_csv("results/N10_x1.0_reg1.0/val.csv")
constrained_df_train = pd.read_csv("results/N10_x1.0_reg1.0/train.csv")
val_constrained = constrained_df_val['loss'].values
train_constrained = constrained_df_train['loss'].values

# 读取放宽位置约束的结果（x_max = 100.0 m，位置约束实际不生效）
val_unconstrained = np.loadtxt("results_unconstrained/N10_x100.0_reg1.0/val.csv", delimiter=',')
train_unconstrained = np.loadtxt("results_unconstrained/N10_x100.0_reg1.0/train.csv", delimiter=',')

# 读取baseline结果
baseline_df_val = pd.read_csv("work_cartpole_baseline/val_loss.csv")
baseline_df_train = pd.read_csv("work_cartpole_baseline/train_loss.csv")
val_baseline = baseline_df_val['loss'].values
train_baseline = baseline_df_train['loss'].values

# 统计信息
print("=" * 60)
print("Constrained vs Unconstrained vs Baseline - Comparison")
print("=" * 60)
print("Validation Loss:")
print(f"  Constrained Final: {val_constrained[-1]:.6f}")
print(f"  Unconstrained Final: {val_unconstrained[-1]:.6f}")
print(f"  Baseline Final: {val_baseline[-1]:.6f}")
print("\nTraining Loss:")
print(f"  Constrained Final: {train_constrained[-1]:.6f}")
print(f"  Unconstrained Final: {train_unconstrained[-1]:.6f}")
print(f"  Baseline Final: {train_baseline[-1]:.6f}")
print("=" * 60)

# 绘图1 - 验证集
plt.figure(figsize=(10, 6))
plt.plot(val_constrained, label='Constrained', linewidth=2, color='#1f77b4')
plt.plot(val_unconstrained, label='Unconstrained', linewidth=2, color='#ff7f0e')
plt.plot(val_baseline, label='Baseline', linewidth=2, color='#2ca02c')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Constrained vs Unconstrained vs Baseline - Validation')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/constrained_vs_unconstrained_vs_baseline_val.png", dpi=150)

# 绘图2 - 训练集
plt.figure(figsize=(10, 6))
plt.plot(train_constrained, label='Constrained', linewidth=2, color='#1f77b4')
plt.plot(train_unconstrained, label='Unconstrained', linewidth=2, color='#ff7f0e')
plt.plot(train_baseline, label='Baseline', linewidth=2, color='#2ca02c')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Constrained vs Unconstrained vs Baseline - Training')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/constrained_vs_unconstrained_vs_baseline_train.png", dpi=150)

print("\nPlots saved:")
print("  - plots/constrained_vs_unconstrained_vs_baseline_val.png (Validation)")
print("  - plots/constrained_vs_unconstrained_vs_baseline_train.png (Training)")
print("=" * 60)
print("\nAnalysis Conclusion:")
if val_constrained[-1] < val_unconstrained[-1]:
    improvement = (val_unconstrained[-1] - val_constrained[-1]) / val_unconstrained[-1] * 100
    print("Constrained model performs best!")
    print(f"  Improvement over Unconstrained: {improvement:.1f}%")
elif val_baseline[-1] < val_constrained[-1]:
    improvement_baseline = (val_constrained[-1] - val_baseline[-1]) / val_constrained[-1] * 100
    print(f"  Baseline performs {improvement_baseline:.1f}% better than Constrained")
