#!/usr/bin/env python3
"""
鲁棒性测试脚本 - 使用 aligator 动力学验证 OptNetCartPoleH3 模型
用法：
    python robustness_aligator.py --mode random --num 100
    python robustness_aligator.py --mode train   # 需提供 features.pt
"""
import os
import sys

# Allow this script to be run directly from the repository root, e.g.
#     python extensions/robustness.py
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from qpth.qp import QPFunction
import pinocchio as pin
import aligator
from aligator import manifolds, dynamics

# ==================== 模型类定义 ====================
# 从用户提供的代码复制，精简导入
class OptNetCartPoleH3(nn.Module):
    """
    2B 模型：QP 在物理单位里解，但训练输出仍是归一化单位。
    输入 x: (B,T,4,1)  —— 归一化 (normalized)
    输出 u0: (B,T,1,1) —— 归一化 (normalized)
    """
    def __init__(
        self,
        u_min: float = -10.0,
        u_max: float = 10.0,
        x_min: float = -2.4,
        x_max: float = 2.4,
        dt: float = 0.02,
        pos_idx: int = 0,
        q_reg_eps: float = 1e-1,
        qp_max_iter: int = 2000,
        qp_eps: float = 1e-3,
        p_clip: float = 50.0,
    ):
        super().__init__()
        self.u_min = float(u_min)
        self.u_max = float(u_max)
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.pos_idx = int(pos_idx)
        self.nx = 4
        self.nu = 1
        self.N = 3
        self.n_var = self.N * self.nx + self.N * self.nu  # 15
        self.n_eq = self.N * self.nx                      # 12
        self.n_ineq = 4 * self.N                           # 12
        self.q_reg_eps = float(q_reg_eps)
        self.qp_max_iter = int(qp_max_iter)
        self.qp_eps = float(qp_eps)
        self.p_clip = float(p_clip)
        self.qp = QPFunction(maxIter=self.qp_max_iter, eps=self.qp_eps, verbose=False)

        # feature net
        self.phi = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        # Q = L L^T
        self.L_raw = nn.Parameter(torch.eye(self.n_var, dtype=torch.double))
        self.W_p = nn.Linear(64, self.n_var, bias=False).double()

        # linear dynamics (物理单位)
        A_d = torch.tensor([
            [1.0, dt, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, dt],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=torch.double)
        B_d = torch.tensor([
            [0.0],
            [dt],
            [0.0],
            [dt],
        ], dtype=torch.double)
        self.register_buffer("A_d", A_d)
        self.register_buffer("B_d", B_d)

        # norm stats buffers
        self.register_buffer("x_mean_buf", torch.zeros(self.nx, dtype=torch.double))
        self.register_buffer("x_std_buf", torch.ones(self.nx, dtype=torch.double))
        self.register_buffer("y_mean_buf", torch.zeros(1, dtype=torch.double))
        self.register_buffer("y_std_buf", torch.ones(1, dtype=torch.double))
        self.has_norm = False

        # indices
        def x_block(i): return (i - 1) * self.nx
        def u_index(i): return self.N * self.nx + i
        self.u0_index = u_index(0)

        # Aeq_base
        Aeq = torch.zeros(self.n_eq, self.n_var, dtype=torch.double)
        # x1 - B u0 = A x0
        rows = slice(0, self.nx)
        cols_x1 = slice(x_block(1), x_block(1) + self.nx)
        idx_u0 = u_index(0)
        Aeq[rows, cols_x1] = torch.eye(self.nx, dtype=torch.double)
        Aeq[rows, idx_u0] = -self.B_d.view(-1)
        # x2 - A x1 - B u1 = 0
        rows = slice(self.nx, 2*self.nx)
        cols_x2 = slice(x_block(2), x_block(2) + self.nx)
        cols_x1 = slice(x_block(1), x_block(1) + self.nx)
        idx_u1 = u_index(1)
        Aeq[rows, cols_x2] = torch.eye(self.nx, dtype=torch.double)
        Aeq[rows, cols_x1] = -self.A_d
        Aeq[rows, idx_u1] = -self.B_d.view(-1)
        # x3 - A x2 - B u2 = 0
        rows = slice(2*self.nx, 3*self.nx)
        cols_x3 = slice(x_block(3), x_block(3) + self.nx)
        cols_x2 = slice(x_block(2), x_block(2) + self.nx)
        idx_u2 = u_index(2)
        Aeq[rows, cols_x3] = torch.eye(self.nx, dtype=torch.double)
        Aeq[rows, cols_x2] = -self.A_d
        Aeq[rows, idx_u2] = -self.B_d.view(-1)
        self.register_buffer("Aeq_base", Aeq)

        # G_base, h_base (物理 bounds)
        G = torch.zeros(self.n_ineq, self.n_var, dtype=torch.double)
        h = torch.zeros(self.n_ineq, dtype=torch.double)
        for i in range(self.N):
            ui = u_index(i)
            # u_i <= u_max
            G[2*i, ui] = 1.0
            h[2*i] = self.u_max
            # -u_i <= -u_min
            G[2*i+1, ui] = -1.0
            h[2*i+1] = -self.u_min
            # position bound on x_{i+1}[pos_idx]
            xi0 = x_block(i + 1)
            xi_pos = xi0 + self.pos_idx
            # x_pos <= x_max
            G[2*self.N + 2*i, xi_pos] = 1.0
            h[2*self.N + 2*i] = self.x_max
            # -x_pos <= -x_min
            G[2*self.N + 2*i+1, xi_pos] = -1.0
            h[2*self.N + 2*i+1] = -self.x_min
        self.register_buffer("G_base", G)
        self.register_buffer("h_base", h)

    def set_norm_stats(self, norm_stats, device=None):
        if norm_stats is None:
            self.has_norm = False
            return
        if device is None:
            device = self.x_mean_buf.device
        x_mean = norm_stats["x_mean"].view(self.nx).double().to(device)
        x_std = norm_stats["x_std"].view(self.nx).double().to(device)
        y_mean = norm_stats["y_mean"].view(1).double().to(device)
        y_std = norm_stats["y_std"].view(1).double().to(device)
        self.x_mean_buf.copy_(x_mean)
        self.x_std_buf.copy_(x_std)
        self.y_mean_buf.copy_(y_mean)
        self.y_std_buf.copy_(y_std)
        self.has_norm = True

    def forward(self, x, return_qp=False):
        B, T, _, _ = x.shape
        device = x.device
        x_flat = x.view(B*T, self.nx).float()
        x0_norm = x_flat.double()
        if not self.has_norm:
            raise RuntimeError("需要先调用 set_norm_stats 设置归一化统计量。")
        x0_phys = x0_norm * self.x_std_buf.unsqueeze(0) + self.x_mean_buf.unsqueeze(0)
        f = self.phi(x_flat).double()
        L = torch.tril(self.L_raw)
        Q_big = (L @ L.T).unsqueeze(0).expand(B*T, -1, -1)
        I = torch.eye(self.n_var, dtype=torch.double, device=device).unsqueeze(0).expand(B*T, -1, -1)
        Q_big = Q_big + self.q_reg_eps * I
        Q_big = 0.5 * (Q_big + Q_big.transpose(1, 2))
        p_big = self.W_p(f)
        if self.p_clip is not None and self.p_clip > 0:
            p_big = torch.clamp(p_big, -self.p_clip, self.p_clip)
        G = self.G_base.unsqueeze(0).expand(B*T, -1, -1).to(device)
        h = self.h_base.unsqueeze(0).expand(B*T, -1).to(device)
        Aeq = self.Aeq_base.unsqueeze(0).expand(B*T, -1, -1).to(device)
        beq = torch.zeros(B*T, self.n_eq, dtype=torch.double, device=device)
        beq[:, 0:self.nx] = (self.A_d @ x0_phys.unsqueeze(-1)).squeeze(-1)
        z = self.qp(Q_big, p_big, G, h, Aeq, beq)
        u0_phys = z[:, self.u0_index].view(B, T, 1, 1).double()
        y_mean = self.y_mean_buf.view(1, 1, 1, 1)
        y_std = self.y_std_buf.view(1, 1, 1, 1)
        u0_norm = ((u0_phys - y_mean) / (y_std + 1e-12)).float()
        if return_qp:
            return u0_norm, (Q_big, p_big, G, h, Aeq, beq, z)
        else:
            return u0_norm


# ==================== 创建 aligator 倒立摆动力学 ====================
def create_cartpole_model(mass_cart=1.0, mass_pole=0.1, length_pole=0.5):
    """使用 pinocchio 构建简单的倒立摆模型（新版API）"""
    model = pin.Model()

    # 1. 添加小车关节（平移 X 轴）
    joint_cart = pin.JointModelPX()
    cart_placement = pin.SE3.Identity()  # 关节相对于父体的位姿
    cart_id = model.addJoint(0, joint_cart, cart_placement, "cart")
    # 设置小车的惯量
    inertia_cart = pin.Inertia(mass_cart, np.zeros(3), np.eye(3) * 1e-6)
    model.appendBodyToJoint(cart_id, inertia_cart, pin.SE3.Identity())

    # 2. 添加摆杆关节（绕 Y 轴旋转）
    joint_pole = pin.JointModelRY()
    pole_placement = pin.SE3.Identity()  # 摆杆相对于小车的位置（默认在小车原点）
    pole_id = model.addJoint(cart_id, joint_pole, pole_placement, "pole")
    # 设置摆杆的惯量（质心在杆的中间偏下位置）
    inertia_pole = pin.Inertia(mass_pole, np.array([0, 0, -length_pole/2]), np.eye(3) * 1e-6)
    model.appendBodyToJoint(pole_id, inertia_pole, pin.SE3.Identity())

    # 3. 添加末端框架（可选，用于可视化）
    frame_placement = pin.SE3(np.eye(3), np.array([0, 0, -length_pole]))
    model.addFrame(pin.Frame("end_effector", pole_id, 0, frame_placement, pin.FrameType.OP_FRAME))

    # 4. 设置关节限位（可根据需要调整）
    model.lowerPositionLimit = np.array([-10.0, -np.pi])
    model.upperPositionLimit = np.array([10.0, np.pi])
    model.velocityLimit = np.array([10.0, 10.0])

    return model

def create_aligator_dynamics(dt=0.02):
    """创建 aligator 离散动力学对象"""
    model = create_cartpole_model()
    # 驱动矩阵：力作用在小车关节（第一个关节）
    act_mat = np.zeros((model.nv, 1))
    act_mat[0, 0] = 1.0
    space = manifolds.MultibodyPhaseSpace(model)
    cont_dyn = dynamics.MultibodyFreeFwdDynamics(space, act_mat)
    disc_dyn = dynamics.IntegratorSemiImplEuler(cont_dyn, dt)
    return disc_dyn, space

# ==================== 辅助函数 ====================
def denormalize_u(u_norm, y_mean, y_std):
    return u_norm * y_std + y_mean

def normalize_x(x_phys, x_mean, x_std):
    return (x_phys - x_mean) / x_std

def is_converged(state, thresh=(0.1, 0.5, 1.0)):
    """判断是否收敛到直立
    state: [x, x_dot, theta, theta_dot]
    thresh: (theta_max, theta_dot_max, x_max)
    """
    x, x_dot, theta, theta_dot = state
    theta_max, theta_dot_max, x_max = thresh
    return (abs(theta) < theta_max) and (abs(theta_dot) < theta_dot_max) and (abs(x) < x_max)

def simulate_step_aligator(state, action, disc_dyn, data):
    """使用 aligator 动力学仿真一步"""
    # 状态顺序转换
    x, x_dot, theta, theta_dot = state
    state_aligator = np.array([x, theta, x_dot, theta_dot], dtype=np.float64)
    u = np.array([action], dtype=np.float64)
    # 调用 forward，结果存入 data
    disc_dyn.forward(state_aligator, u, data)
    # 获取下一个状态（假设属性名为 xnext）
    x_next_aligator = data.xnext
    # 转换回原始顺序
    x_next, theta_next, x_dot_next, theta_dot_next = x_next_aligator
    return np.array([x_next, x_dot_next, theta_next, theta_dot_next])

# ==================== 数据加载（训练集初始状态） ====================
def load_train_init_states(data_dir="data", train_ratio=0.7, val_ratio=0.1):
    """从 features.pt 中提取训练集和验证集的初始状态（物理单位）"""
    import torch
    X = torch.load(os.path.join(data_dir, "features.pt"))  # (N, T, 4)
    Y = torch.load(os.path.join(data_dir, "labels.pt"))
    N = X.shape[0]
    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)
    X_train = X[:n_train]
    X_val = X[n_train:n_train + n_val]
    init_train = X_train[:, 0, :].numpy()
    init_val = X_val[:, 0, :].numpy()
    return np.vstack([init_train, init_val])

# ==================== 主测试函数 ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "random"], default="random",
                        help="测试模式：train 使用训练集初始状态，random 随机生成")
    parser.add_argument("--num", type=int, default=100, help="随机模式下的样本数")
    parser.add_argument("--model_path", type=str, default="extensions/weights/cartpole_optnet_best_2B.pth",
                        help="模型权重文件路径")
    parser.add_argument("--norm_stats_path", type=str, default="extensions/weights/norm_stats.pt",
                        help="标准化统计量文件路径")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="原始数据目录（train模式需要）")
    parser.add_argument("--dt", type=float, default=0.02, help="仿真步长（应与模型一致）")
    parser.add_argument("--max_steps", type=int, default=300, help="最大仿真步数")
    parser.add_argument("--thresh_theta", type=float, default=0.1, help="收敛角度阈值 (rad)")
    parser.add_argument("--thresh_theta_dot", type=float, default=0.5, help="收敛角速度阈值 (rad/s)")
    parser.add_argument("--thresh_x", type=float, default=1.0, help="收敛位置阈值 (m)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    # 1. 创建 aligator 动力学
    disc_dyn, space = create_aligator_dynamics(dt=args.dt)
    data_dyn = disc_dyn.createData()  # 新增
    print("aligator 动力学创建成功，步长 =", args.dt)

    # 2. 加载模型
    model = OptNetCartPoleH3().to(device)
    model.eval()
    norm_stats = torch.load(args.norm_stats_path, map_location=device)
    for k in norm_stats:
        norm_stats[k] = norm_stats[k].to(device)
    model.set_norm_stats(norm_stats, device=device)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print("模型权重加载成功")
    else:
        print("警告：模型文件不存在，使用随机初始化")

    # 提取标准化统计量的数值
    x_mean = norm_stats['x_mean'].cpu().numpy().flatten()
    x_std = norm_stats['x_std'].cpu().numpy().flatten()
    y_mean = norm_stats['y_mean'].cpu().item()
    y_std = norm_stats['y_std'].cpu().item()

    # 3. 准备初始状态
    if args.mode == "train":
        init_states = load_train_init_states(args.data_dir)
        print(f"从训练/验证集加载了 {len(init_states)} 个初始状态")
    else:
        # 随机生成初始状态：在合理范围内采样
        x_range = (-0.3, 0.3)
        x_dot_range = (-1.0, 1.0)
        theta_range = (-0.5, 0.5)      # 大角度
        theta_dot_range = (-1.0, 1.0)
        init_states = []
        for _ in range(args.num):
            x = np.random.uniform(*x_range)
            x_dot = np.random.uniform(*x_dot_range)
            theta = np.random.uniform(*theta_range)
            theta_dot = np.random.uniform(*theta_dot_range)
            init_states.append([x, x_dot, theta, theta_dot])
        init_states = np.array(init_states)
        print(f"随机生成了 {len(init_states)} 个初始状态")

    # 4. 测试循环
    results = []  # (init_state, converged, steps)
    thresh = (args.thresh_theta, args.thresh_theta_dot, args.thresh_x)

    for i, x0 in enumerate(init_states):
        state = x0.copy()
        converged = False
        steps = 0
        for step in range(args.max_steps):
            # 归一化当前状态
            state_norm = (state - x_mean) / x_std
            state_tensor = torch.tensor(state_norm, dtype=torch.float32, device=device).view(1, 1, 4, 1)
            with torch.no_grad():
                u_norm = model(state_tensor).cpu().item()
            u_phys = denormalize_u(u_norm, y_mean, y_std)
            # 仿真一步
            state = simulate_step_aligator(state, u_phys, disc_dyn, data_dyn)
            if is_converged(state, thresh):
                converged = True
                steps = step + 1
                break
        results.append((x0, converged, steps))
        if (i+1) % 20 == 0:
            print(f"已完成 {i+1}/{len(init_states)}")

    # 5. 统计
    converged_flags = [r[1] for r in results]
    conv_rate = np.mean(converged_flags) * 100
    print(f"\n收敛率: {sum(converged_flags)}/{len(results)} = {conv_rate:.2f}%")

    # 6. 分析未收敛样本
    failed = [r for r in results if not r[1]]
    if failed:
        failed_states = np.array([r[0] for r in failed])
        print("\n未收敛样本统计:")
        print("  位置 x: 均值 {:.3f}, 标准差 {:.3f}".format(failed_states[:,0].mean(), failed_states[:,0].std()))
        print("  速度 x_dot: 均值 {:.3f}, 标准差 {:.3f}".format(failed_states[:,1].mean(), failed_states[:,1].std()))
        print("  角度 theta: 均值 {:.3f}, 标准差 {:.3f}".format(failed_states[:,2].mean(), failed_states[:,2].std()))
        print("  角速度 theta_dot: 均值 {:.3f}, 标准差 {:.3f}".format(failed_states[:,3].mean(), failed_states[:,3].std()))

        # 绘制散点图
        all_states = np.array([r[0] for r in results])
        plt.figure(figsize=(8,6))
        sc = plt.scatter(all_states[:,0], all_states[:,2], c=converged_flags, cmap='coolwarm', alpha=0.6, edgecolors='k')
        plt.xlabel('Cart Position x')
        plt.ylabel('Pole Angle theta')
        plt.title('Initial States (Red: converged, Blue: not converged)')
        plt.colorbar(sc, label='Converged')
        plt.grid(True)
        plt.savefig('robustness_aligator.png', dpi=150)
        plt.show()
    else:
        print("所有样本均收敛，无需绘图。")

    # 保存结果
    np.savez('extensions/data/robustness_aligator_results.npz',
             init_states=init_states,
             converged=converged_flags,
             steps=[r[2] for r in results])
    print("结果已保存至 robustness_aligator_results.npz")

if __name__ == "__main__":
    main()