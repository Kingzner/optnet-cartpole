"""
测试条件模型的闭环鲁棒性（修正版：收敛基于目标状态）
支持自定义任务参数范围，输出收敛率、热图、失败案例
"""
import os
import sys

# Allow this script to be run directly from the repository root, e.g.
#     python extensions/test_conditional_robustness.py
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from cartpole.models import OptNetCartPoleH3_Conditional
import pinocchio as pin
import aligator
from aligator import manifolds, dynamics
from cartpole.utils import create_cartpole
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="qpth")

# ==================== 配置 ====================
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="extensions/weights/best_model_conditional.pth")
parser.add_argument("--norm_stats_path", type=str, default="extensions/weights/norm_stats_conditional.pt")
parser.add_argument("--num_tasks", type=int, default=20, help="测试的任务数量")
parser.add_argument("--num_init_per_task", type=int, default=5, help="每个任务测试的初始状态数")
parser.add_argument("--dt", type=float, default=0.02)
parser.add_argument("--max_steps", type=int, default=300)
parser.add_argument("--thresh_theta", type=float, default=0.2, help="角度收敛阈值 (rad)")
parser.add_argument("--thresh_theta_dot", type=float, default=1.0, help="角速度收敛阈值 (rad/s)")
parser.add_argument("--thresh_x", type=float, default=1.5, help="位置收敛阈值 (m)")
parser.add_argument("--seed", type=int, default=42)
# 可选：缩小测试范围以便快速验证
parser.add_argument("--easy", action="store_true", help="使用简单范围（小角度、小速度、对称约束）")
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 创建 aligator 动力学 ====================
def create_aligator_dynamics(dt=0.02):
    model = create_cartpole(1)[0]
    act_mat = np.zeros((model.nv, 1))
    act_mat[0, 0] = 1.0
    space = manifolds.MultibodyPhaseSpace(model)
    cont_dyn = dynamics.MultibodyFreeFwdDynamics(space, act_mat)
    disc_dyn = dynamics.IntegratorSemiImplEuler(cont_dyn, dt)
    return disc_dyn, space

disc_dyn, space = create_aligator_dynamics(dt=args.dt)
data_dyn = disc_dyn.createData()

# ==================== 加载模型 ====================
norm_stats = torch.load(args.norm_stats_path, map_location=device)
model = OptNetCartPoleH3_Conditional().to(device)
model.set_norm_stats(norm_stats, device=device)
model.set_task_norm_stats(norm_stats["task_mean"], norm_stats["task_std"], device=device)
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.eval()

# 提取归一化统计量
x_mean = norm_stats["x_mean"].cpu().numpy().flatten()
x_std = norm_stats["x_std"].cpu().numpy().flatten()
y_mean = norm_stats["y_mean"].cpu().item()
y_std = norm_stats["y_std"].cpu().item()
task_mean = norm_stats["task_mean"].cpu().numpy().flatten()
task_std = norm_stats["task_std"].cpu().numpy().flatten()

# ==================== 辅助函数 ====================
def denormalize_u(u_norm, y_mean, y_std):
    return u_norm * y_std + y_mean

def normalize_state(state, x_mean, x_std):
    return (state - x_mean) / x_std

def normalize_task(task_phys, task_mean, task_std):
    return (task_phys - task_mean) / task_std

def is_converged(state, target_state, thresh):
    """判断当前状态是否收敛到目标状态（模型顺序）"""
    x, x_dot, theta, theta_dot = state
    target_x, target_x_dot, target_theta, target_theta_dot = target_state
    theta_max, theta_dot_max, x_max = thresh
    return (abs(theta - target_theta) < theta_max) and \
           (abs(theta_dot - target_theta_dot) < theta_dot_max) and \
           (abs(x - target_x) < x_max)

def simulate_step(state, action, disc_dyn, data):
    # 状态顺序转换：模型顺序 [x, x_dot, theta, theta_dot] -> aligator顺序 [x, theta, x_dot, theta_dot]
    x, x_dot, theta, theta_dot = state
    state_aligator = np.array([x, theta, x_dot, theta_dot], dtype=np.float64)
    u = np.array([action], dtype=np.float64)
    disc_dyn.forward(state_aligator, u, data)
    x_next_aligator = data.xnext
    x_next, theta_next, x_dot_next, theta_dot_next = x_next_aligator
    return np.array([x_next, x_dot_next, theta_next, theta_dot_next])

# ==================== 定义测试范围 ====================
if args.easy:
    # 简单模式：小范围、对称约束
    task_ranges = {
        "target_x": (-0.3, 0.3),
        "target_theta": (-0.3, 0.3),
        "x_min": (-1.0, -0.8),   # 固定窄范围
        "x_max": (0.8, 1.0)
    }
    x0_ranges = {
        "x": (-0.2, 0.2),
        "x_dot": (-0.5, 0.5),
        "theta": (-0.3, 0.3),
        "theta_dot": (-0.8, 0.8)
    }
else:
    # 原始范围（较大）
    task_ranges = {
        "target_x": (-0.8, 0.8),
        "target_theta": (-0.8, 0.8),
        "x_min": (-1.5, -0.5),
        "x_max": (0.5, 1.5)
    }
    x0_ranges = {
        "x": (-0.5, 0.5),
        "x_dot": (-1.0, 1.0),
        "theta": (-0.8, 0.8),
        "theta_dot": (-1.5, 1.5)
    }

# ==================== 测试循环 ====================
results = []  # (task_params, success_rate, failed_states)
for task_idx in range(args.num_tasks):
    # 随机生成任务参数
    target_x = np.random.uniform(*task_ranges["target_x"])
    target_theta = np.random.uniform(*task_ranges["target_theta"])
    x_min = np.random.uniform(*task_ranges["x_min"])
    x_max = np.random.uniform(*task_ranges["x_max"])
    task_phys = np.array([target_x, target_theta, x_min, x_max], dtype=np.float32)
    task_norm = normalize_task(task_phys, task_mean, task_std)
    
    # 目标状态（模型顺序：[x, x_dot, theta, theta_dot]）
    target_state = np.array([target_x, 0.0, target_theta, 0.0])
    
    successes = 0
    failed_states = []
    for init_i in range(args.num_init_per_task):
        x0_phys = np.array([
            np.random.uniform(*x0_ranges["x"]),
            np.random.uniform(*x0_ranges["x_dot"]),
            np.random.uniform(*x0_ranges["theta"]),
            np.random.uniform(*x0_ranges["theta_dot"])
        ])
        state = x0_phys.copy()
        converged = False
        for step in range(args.max_steps):
            # 归一化
            state_norm = normalize_state(state, x_mean, x_std)
            state_tensor = torch.tensor(state_norm, dtype=torch.float32, device=device).view(1,1,4,1)
            task_tensor = torch.tensor(task_norm, dtype=torch.float32, device=device).view(1,1,4,1)
            with torch.no_grad():
                u_norm = model(state_tensor, task_tensor).cpu().item()
            u_phys = denormalize_u(u_norm, y_mean, y_std)
            state = simulate_step(state, u_phys, disc_dyn, data_dyn)
            if is_converged(state, target_state, (args.thresh_theta, args.thresh_theta_dot, args.thresh_x)):
                converged = True
                break
        if converged:
            successes += 1
        else:
            failed_states.append(x0_phys.tolist())
    
    success_rate = successes / args.num_init_per_task
    results.append((task_phys, success_rate, failed_states))
    print(f"Task {task_idx+1}/{args.num_tasks}: target=({target_x:.2f},{target_theta:.2f}), bounds=({x_min:.2f},{x_max:.2f}), rate={success_rate:.2f}")

# ==================== 可视化 ====================
task_array = np.array([r[0] for r in results])
rates = np.array([r[1] for r in results])

# 1. 目标角度 vs 成功率
plt.figure(figsize=(6,4))
plt.scatter(task_array[:,1], rates, alpha=0.6)
plt.xlabel("Target theta (rad)")
plt.ylabel("Success rate")
plt.title("Convergence vs Target Angle")
plt.grid(True)
plt.savefig("conv_vs_target_theta.png", dpi=150)
plt.show()

# 2. 约束宽度 vs 成功率
widths = task_array[:,3] - task_array[:,2]
plt.figure(figsize=(6,4))
plt.scatter(widths, rates, alpha=0.6)
plt.xlabel("Constraint width (m)")
plt.ylabel("Success rate")
plt.title("Convergence vs Constraint Width")
plt.grid(True)
plt.savefig("conv_vs_constraint_width.png", dpi=150)
plt.show()

# 3. 热图
from scipy.interpolate import griddata
if len(task_array) >= 4:
    theta_vals = np.linspace(task_array[:,1].min(), task_array[:,1].max(), 20)
    width_vals = np.linspace(widths.min(), widths.max(), 20)
    theta_grid, width_grid = np.meshgrid(theta_vals, width_vals)
    grid_points = np.column_stack([theta_grid.ravel(), width_grid.ravel()])
    rate_grid = griddata((task_array[:,1], widths), rates, grid_points, method='linear')
    if not np.all(np.isnan(rate_grid)):
        rate_grid = rate_grid.reshape(theta_grid.shape)
        plt.figure(figsize=(8,6))
        plt.contourf(theta_grid, width_grid, rate_grid, levels=20, cmap='RdYlGn')
        plt.colorbar(label='Success rate')
        plt.xlabel("Target theta (rad)")
        plt.ylabel("Constraint width (m)")
        plt.title("Convergence Rate Heatmap")
        plt.savefig("conv_heatmap.png", dpi=150)
        plt.show()
    else:
        print("无法生成热图：插值结果全为NaN")
else:
    print("点数不足，跳过热图绘制")

# 保存结果
np.savez("extensions/data/test_results.npz", tasks=task_array, rates=rates)
print("测试完成，结果已保存至 test_results.npz")