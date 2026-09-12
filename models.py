import numpy as np

from itertools import product

import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn.parameter import Parameter

from qpth.qp import SpQPFunction, QPFunction


class OptNetCartPoleMPC(nn.Module):
    """
    完整的 MPC 模型，支持可配置的预测步数 N
    
    输入 x: (B,T,4,1)  —— 归一化 (normalized)
    QP 内部：把 x_norm -> x_phys，构造 beq (物理) 并用物理 bounds 解 QP
    输出 u0: (B,T,1,1) —— 归一化 (normalized) 便于继续用 MSE vs labels
    
    关键参数:
    - N: 预测步数（推荐 10-20）
    - dt: 时间步长
    - u_min/u_max: 控制输入约束
    - x_min/x_max: 小车位置约束
    """

    def __init__(
        self,
        N: int = 10,           # 预测步数（完整MPC推荐10-20）
        u_min: float = -10.0,
        u_max: float = 10.0,
        x_min: float = -1.0,
        x_max: float =  1.0,
        dt: float    = 0.01,
        dynamics_file: str = None,
        pos_idx: int = 0,           # 哪一维是 cart position（默认第0维）
        q_reg_eps: float = 1e-1,    # Q += eps I，先用大一点救稳定
        qp_max_iter: int = 500,     # 训练先别太大（不然慢），raw check 再调大
        qp_eps: float = 1e-10,
        p_clip: float = 50.0,
    ):
        super().__init__()

        # ----- bounds (物理单位) -----
        self.u_min = float(u_min)
        self.u_max = float(u_max)
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.pos_idx = int(pos_idx)

        # dims
        self.nx = 4
        self.nu = 1
        self.N  = N

        self.n_var  = self.N * self.nx + self.N * self.nu   # 状态变量 + 控制变量
        self.n_eq   = self.N * self.nx                      # 等式约束数（每个时间步一个动力学约束）
        self.n_ineq = 4 * self.N                            # 不等式约束数（每个时间步：u上下界 + x位置上下界）

        # solver knobs
        self.q_reg_eps = float(q_reg_eps)
        self.qp_max_iter = int(qp_max_iter)
        self.qp_eps = float(qp_eps)
        self.p_clip = float(p_clip)
        
        self.qp = QPFunction(maxIter=self.qp_max_iter, eps=self.qp_eps, verbose=0)
        
        # ===== 1) feature net φ(x_norm) =====
        self.phi = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # ===== 2) Q = L L^T (learnable PSD) =====
        self.L_raw = nn.Parameter(torch.eye(self.n_var, dtype=torch.double))

        # p(x_norm) = W_p φ(x_norm)
        self.W_p = nn.Linear(64, self.n_var, bias=False).double()

        # ===== 3) linear dynamics (物理单位) =====
        if dynamics_file:
            dyn = np.load(dynamics_file)
            A_np = dyn["A_d"]
            B_np = dyn["B_d"]
            if A_np.shape != (4, 4) or B_np.shape != (4, 1):
                raise ValueError(
                    f"Invalid dynamics shapes from {dynamics_file}: "
                    f"A_d={A_np.shape}, B_d={B_np.shape}"
                )
            A_d = torch.tensor(A_np, dtype=torch.double)
            B_d = torch.tensor(B_np, dtype=torch.double)
        else:
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

        # ===== norm stats buffers =====
        self.register_buffer("x_mean_buf", torch.zeros(self.nx, dtype=torch.double))
        self.register_buffer("x_std_buf",  torch.ones(self.nx, dtype=torch.double))
        self.register_buffer("y_mean_buf", torch.zeros(1, dtype=torch.double))
        self.register_buffer("y_std_buf",  torch.ones(1, dtype=torch.double))
        self.has_norm = False

        nx, N = self.nx, self.N

        def x_block(i: int):
            return (i - 1) * nx

        def u_index(i: int):
            return N * nx + i

        self.u0_index = u_index(0)

        # ===== 4) Aeq_base - 动态约束矩阵 (支持任意N) =====
        Aeq = torch.zeros(self.n_eq, self.n_var, dtype=torch.double)

        # x1 - B u0 = A x0
        rows = slice(0, nx)
        cols_x1 = slice(x_block(1), x_block(1) + nx)
        idx_u0 = u_index(0)
        Aeq[rows, cols_x1] = torch.eye(nx, dtype=torch.double)
        Aeq[rows, idx_u0]  = -self.B_d.view(-1)

        # xi - A x_{i-1} - B u_{i-1} = 0, for i = 2..N
        for i in range(2, N+1):
            rows = slice((i-1)*nx, i*nx)
            cols_xi = slice(x_block(i), x_block(i) + nx)
            cols_xim1 = slice(x_block(i-1), x_block(i-1) + nx)
            idx_uim1 = u_index(i-1)
            Aeq[rows, cols_xi] = torch.eye(nx, dtype=torch.double)
            Aeq[rows, cols_xim1] = -self.A_d
            Aeq[rows, idx_uim1]  = -self.B_d.view(-1)

        self.register_buffer("Aeq_base", Aeq)

        # ===== 5) G_base, h_base (物理 bounds) =====
        G = torch.zeros(self.n_ineq, self.n_var, dtype=torch.double)
        h = torch.zeros(self.n_ineq, dtype=torch.double)

        for i in range(N):
            ui = u_index(i)

            # 控制约束: u_i <= u_max 和 u_i >= u_min
            r_u_upper = 2*i
            r_u_lower = 2*i + 1

            G[r_u_upper, ui] = 1.0
            h[r_u_upper] = self.u_max

            G[r_u_lower, ui] = -1.0
            h[r_u_lower] = -self.u_min

            # 位置约束: x_{i+1}[pos_idx] <= x_max 和 >= x_min
            xi0 = x_block(i + 1)
            xi_pos = xi0 + self.pos_idx

            r_x_upper = 2*N + 2*i
            r_x_lower = 2*N + 2*i + 1

            G[r_x_upper, xi_pos] = 1.0
            h[r_x_upper] = self.x_max

            G[r_x_lower, xi_pos] = -1.0
            h[r_x_lower] = -self.x_min

        self.register_buffer("G_base", G)
        self.register_buffer("h_base", h)
        
    # ====== 设置归一化统计量 ======
    def set_norm_stats(self, norm_stats, device=None):
        if norm_stats is None:
            self.has_norm = False
            return

        if device is None:
            device = self.x_mean_buf.device

        x_mean = norm_stats["x_mean"].view(self.nx).double().to(device)
        x_std  = norm_stats["x_std"].view(self.nx).double().to(device)
        y_mean = norm_stats["y_mean"].view(1).double().to(device)
        y_std  = norm_stats["y_std"].view(1).double().to(device)

        self.x_mean_buf.copy_(x_mean)
        self.x_std_buf.copy_(x_std)
        self.y_mean_buf.copy_(y_mean)
        self.y_std_buf.copy_(y_std)

        self.has_norm = True

    def forward(self, x, return_qp=False):
        """
        x: (B,T,4,1)  —— 归一化输入
        return:
          u0_norm: (B,T,1,1)
          if return_qp: (u0_norm, (Q_big,p_big,G,h,Aeq,beq,z_phys))
        """
        B, T, _, _ = x.shape
        device = x.device

        x_flat = x.view(B*T, self.nx).float()
        x0_norm = x_flat.double()  # (B*T,4)

        if not self.has_norm:
            raise RuntimeError("需要 set_norm_stats(norm_stats)")

        # x_norm -> x_phys
        x0_phys = x0_norm * self.x_std_buf.unsqueeze(0) + self.x_mean_buf.unsqueeze(0)  # (B*T,4)

        # features from normalized x
        f = self.phi(x_flat).double()  # (B*T,64)

        # Q = L L^T + eps I, and symmetrize
        L = torch.tril(self.L_raw).to(device)
        Q_big = (L @ L.T).unsqueeze(0).expand(B*T, -1, -1).to(device)

        I = torch.eye(self.n_var, dtype=torch.double, device=device).unsqueeze(0).expand(B*T, -1, -1)
        Q_big = Q_big + self.q_reg_eps * I
        Q_big = 0.5 * (Q_big + Q_big.transpose(1, 2))

        # p(x_norm) and clamp
        p_big = self.W_p(f).to(device)
        if self.p_clip is not None and self.p_clip > 0:
            p_big = torch.clamp(p_big, -self.p_clip, self.p_clip)

        # constraints (物理)
        G = self.G_base.unsqueeze(0).expand(B*T, -1, -1).to(device)
        h = self.h_base.unsqueeze(0).expand(B*T, -1).to(device)

        Aeq = self.Aeq_base.unsqueeze(0).expand(B*T, -1, -1).to(device)

        # beq: 第一段 = A_d x0_phys，其余为 0
        beq = torch.zeros(B*T, self.n_eq, dtype=torch.double, device=device)
        beq[:, 0:self.nx] = (self.A_d.to(device) @ x0_phys.unsqueeze(-1).to(device)).squeeze(-1)

        # solve QP -> z_phys
        z = self.qp(Q_big, p_big, G, h, Aeq, beq)  # (B*T, n_var) 物理单位

        # u0_phys -> u0_norm (用于训练loss)
        u0_phys = z[:, self.u0_index].view(B, T, 1, 1).double().to(device)
        y_mean = self.y_mean_buf.view(1, 1, 1, 1).to(device)
        y_std  = self.y_std_buf.view(1, 1, 1, 1).to(device)
        u0_norm = ((u0_phys - y_mean) / (y_std + 1e-12)).float()

        if return_qp:
            return u0_norm, (Q_big, p_big, G, h, Aeq, beq, z)
        else:
            return u0_norm


class OptNetCartPoleMPC_Conditional(OptNetCartPoleMPC):
    """
    条件化MPC模型：接收状态和任务参数，输出控制量。
    任务参数包括：[target_x, target_theta, x_min, x_max]（已归一化）
    """
    def __init__(self, task_dim=4, **kwargs):
        super().__init__(**kwargs)
        self.task_dim = task_dim

        # 重新定义 phi 层，输入维度 = 4 (状态) + task_dim
        self.phi = nn.Sequential(
            nn.Linear(4 + task_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # 任务参数统计量 buffers
        self.register_buffer("task_mean_buf", torch.zeros(task_dim, dtype=torch.double))
        self.register_buffer("task_std_buf", torch.ones(task_dim, dtype=torch.double))
        self.has_task_norm = False

    def set_task_norm_stats(self, task_mean, task_std, device=None):
        """设置任务参数的归一化统计量"""
        if device is None:
            device = self.task_mean_buf.device
        self.task_mean_buf.copy_(task_mean.view(self.task_dim).double().to(device))
        self.task_std_buf.copy_(task_std.view(self.task_dim).double().to(device))
        self.has_task_norm = True

    def forward(self, x, task_params, return_qp=False):
        """
        x: (B, T, 4, 1) 归一化状态
        task_params: (B, T, task_dim, 1) 或 (B, T, task_dim) 归一化任务参数
        """
        B, T, _, _ = x.shape
        device = x.device

        # 展平状态和任务参数
        x_flat = x.view(B * T, self.nx).float()
        if task_params.dim() == 4:
            task_flat = task_params.view(B * T, self.task_dim).float()
        else:
            task_flat = task_params.view(B * T, self.task_dim).float()

        # 拼接状态和任务参数
        combined = torch.cat([x_flat, task_flat], dim=1)

        # 特征提取
        f = self.phi(combined).double()

        # 构造 QP
        L = torch.tril(self.L_raw).to(device)
        Q_big = (L @ L.T).unsqueeze(0).expand(B * T, -1, -1).to(device)
        I = torch.eye(self.n_var, dtype=torch.double, device=device).unsqueeze(0).expand(B * T, -1, -1)
        Q_big = Q_big + self.q_reg_eps * I
        Q_big = 0.5 * (Q_big + Q_big.transpose(1, 2))

        p_big = self.W_p(f).to(device)
        if self.p_clip is not None and self.p_clip > 0:
            p_big = torch.clamp(p_big, -self.p_clip, self.p_clip)

        # 约束
        G = self.G_base.unsqueeze(0).expand(B * T, -1, -1).to(device)
        h = self.h_base.unsqueeze(0).expand(B * T, -1).to(device)
        Aeq = self.Aeq_base.unsqueeze(0).expand(B * T, -1, -1).to(device)

        # 等式右边
        x0_norm = x_flat.double().to(device)
        if not self.has_norm:
            raise RuntimeError("请先调用 set_norm_stats 设置状态归一化统计量")
        x0_phys = x0_norm * self.x_std_buf.unsqueeze(0).to(device) + self.x_mean_buf.unsqueeze(0).to(device)

        beq = torch.zeros(B * T, self.n_eq, dtype=torch.double, device=device)
        beq[:, 0:self.nx] = (self.A_d.to(device) @ x0_phys.unsqueeze(-1)).squeeze(-1)

        # 求解 QP
        z = self.qp(Q_big, p_big, G, h, Aeq, beq)

        # 提取控制量并反标准化
        u0_phys = z[:, self.u0_index].view(B, T, 1, 1).double().to(device)
        y_mean = self.y_mean_buf.view(1, 1, 1, 1).to(device)
        y_std = self.y_std_buf.view(1, 1, 1, 1).to(device)
        u0_norm = ((u0_phys - y_mean) / (y_std + 1e-12)).float()

        if return_qp:
            return u0_norm, (Q_big, p_big, G, h, Aeq, beq, z)
        else:
            return u0_norm
