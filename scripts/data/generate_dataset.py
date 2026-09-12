import os
import sys

# Allow this script to be run directly from the repository root, e.g.
#     python scripts/data/generate_dataset.py
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pinocchio as pin 
import numpy as np 
import aligator 
import torch 
from aligator import constraints, manifolds 
from cartpole.utils import create_cartpole  # 请确保 utils.py 在同一目录或 PYTHONPATH 中 
 
# =============================== 
# 1️⃣ 基本参数配置 
# =============================== 
dt = 0.01                # 时间步长 (s) 
nsteps = 500             # 每个轨迹的步数（对应 5 秒） 
nu = 1                   # 控制维度 
N_SAMPLES = 1000         # 总共生成 1000 条轨迹 
SAVE_DIR = "data"  # 保存数据的文件夹 
os.makedirs(SAVE_DIR, exist_ok=True)  # 如果文件夹不存在则创建 
 
# 位置约束范围（放宽到 ±1.0 米） 
x_cart_min = -1.0 
x_cart_max = 1.0 
 
# 噪声参数（保持小幅度，不破坏物理意义） 
STATE_NOISE_STD = 0.01    # 状态噪声标准差（传感器噪声级别） 
CONTROL_NOISE_STD = 0.1   # 控制噪声标准差（执行器噪声级别） 
 
# =============================== 
# 2️⃣ 创建倒立摆模型与动力学 
# =============================== 
model, geom_model, data, geom_data, ddl = create_cartpole(1) 
 
# 驱动矩阵：力作用在小车关节（第一个关节） 
act_mat = np.zeros((2, nu)) 
act_mat[0, 0] = 1.0 
 
# 状态空间（位置+速度） 
space = manifolds.MultibodyPhaseSpace(model) 
ndx = space.ndx  # 状态切空间维度（等于状态维度） 
 
# 连续动力学 + 离散化（半隐式欧拉） 
cont_dyn = aligator.dynamics.MultibodyFreeFwdDynamics(space, act_mat) 
disc_dyn = aligator.dynamics.IntegratorSemiImplEuler(cont_dyn, dt) 
 
# =============================== 
# 3️⃣ 定义成本函数 
# =============================== 
# 运行成本（每步） 
rcost = aligator.CostStack(space, nu) 
wu = np.ones(nu) * 1e-3           # 控制正则化权重 
wx_reg = np.eye(ndx) * 1e-3       # 状态正则化权重 
rcost.addCost( 
    "ureg", 
    aligator.QuadraticControlCost(space, np.zeros(nu), np.diag(wu) * dt) 
) 
rcost.addCost( 
    "xreg", 
    aligator.QuadraticStateCost(space, nu, space.neutral(), wx_reg * dt) 
) 
 
# 终端成本（引导到零状态） 
term_cost = aligator.CostStack(space, nu) 
term_cost.addCost( 
    "xreg", 
    aligator.QuadraticStateCost(space, nu, space.neutral(), wx_reg) 
) 
 
# =============================== 
# 4️⃣ 小车位置约束函数 
# =============================== 
def get_cart_pos_cstr(): 
    """返回位置约束的残差和边界""" 
    state_err = aligator.StateErrorResidual(space, nu, space.neutral()) 
    # 选择矩阵：提取状态中的第一个元素（小车位置 x） 
    select_mat = np.zeros((1, ndx)) 
    select_mat[0, 0] = 1.0 
    cart_pos_residual = aligator.LinearFunctionComposition( 
        state_err, 
        select_mat, 
        np.zeros(1) 
    ) 
    # 盒式约束 [x_min, x_max] 
    cart_box = constraints.BoxConstraint( 
        np.array([x_cart_min]), 
        np.array([x_cart_max]) 
    ) 
    return cart_pos_residual, cart_box 
 
# =============================== 
# 5️⃣ 单条轨迹求解函数 
# =============================== 
def solve_one_traj(x0): 
    """ 
    求解给定初始状态 x0 的最优轨迹。 
    返回 (xs_opt, us_opt)，若求解失败则返回 (None, None)。 
    """ 
    problem = aligator.TrajOptProblem(x0, nu, space, term_cost) 
 
    # 构建多阶段问题（每步添加运行成本和位置约束） 
    for i in range(nsteps): 
        stage = aligator.StageModel(rcost, disc_dyn) 
        stage.addConstraint(*get_cart_pos_cstr()) 
        problem.addStage(stage) 
 
    # 配置求解器（ProxDDP） 
    solver = aligator.SolverProxDDP( 
        tol=1e-6,          # 收敛容忍度 
        mu_init=1e-3,      # 初始正则化参数 
        max_iters=500,     # 最大迭代次数（可适当增加） 
        verbose=aligator.VerboseLevel.QUIET  # 静默模式 
    ) 
 
    # 初始猜测：零控制输入，向前滚动得到初始状态轨迹 
    u0 = np.zeros(nu) 
    us_init = [u0] * nsteps 
    xs_init = aligator.rollout(disc_dyn, x0, us_init) 
 
    try: 
        solver.setup(problem) 
        solver.run(problem, xs_init, us_init) 
        return solver.results.xs, solver.results.us 
    except Exception as e: 
        # 如果求解失败（例如不收敛），打印警告并跳过 
        print(f"  求解失败 for x0={x0}, 错误: {e}") 
        return None, None 
 
# =============================== 
# 6️⃣ 添加噪声函数 
# =============================== 
def add_noise_to_trajectory(xs, us, state_noise_std=0.01, control_noise_std=0.1): 
    """ 
    对轨迹添加高斯噪声 
    - xs: 状态轨迹 (nsteps, 4) 
    - us: 控制轨迹 (nsteps, 1) 
    - 返回带噪声的轨迹 
    """ 
    # 添加状态噪声（保持初始状态不变，从第2步开始） 
    xs_noisy = xs.copy() 
    for t in range(1, len(xs)): 
        xs_noisy[t] += np.random.normal(0, state_noise_std, xs.shape[1]) 
     
    # 添加控制噪声 
    us_noisy = us.copy() 
    us_noisy += np.random.normal(0, control_noise_std, us.shape) 
     
    # 确保控制信号在合理范围内（物理约束） 
    us_noisy = np.clip(us_noisy, -10.0, 10.0) 
     
    return xs_noisy, us_noisy 
 
# =============================== 
# 7️⃣ 批量生成数据（带噪声） 
# =============================== 
all_states = []   # 存放所有轨迹的状态（转换顺序后） 
all_controls = [] # 存放所有轨迹的控制 
 
print("开始生成数据集（带噪声）...") 
print(f"噪声参数: 状态std={STATE_NOISE_STD}, 控制std={CONTROL_NOISE_STD}") 
success_count = 0 
 
for k in range(N_SAMPLES): 
    # 随机生成初始状态（物理单位）- 保持原始分布不变 
    x0 = space.neutral()  # 中性状态 [0,0,0,0] 
    x0[0] = np.random.uniform(-0.3, 0.3)   # 小车位置 (m) 
    x0[1] = np.random.uniform(-0.5, 0.5)   # 小车速度 (m/s) 
    x0[2] = np.random.uniform(-0.5, 0.5)   # 摆杆角度 (rad) 
    x0[3] = np.random.uniform(-1.0, 1.0)   # 摆杆角速度 (rad/s) 
 
    if (k + 1) % 50 == 0: 
        print(f"求解轨迹 {k+1}/{N_SAMPLES}") 
 
    xs_opt, us_opt = solve_one_traj(x0) 
 
    if xs_opt is None: 
        continue   # 求解失败，跳过该样本 
 
    success_count += 1 
 
    # ===== 状态顺序转换 ===== 
    # aligator 输出的 xs_opt 顺序为 [x, theta, x_dot, theta_dot] 
    # OptNet 模型期望的顺序为 [x, x_dot, theta, theta_dot] 
    xs_opt = np.asarray(xs_opt)  # shape (nsteps+1, 4) 
    xs_converted = np.stack([ 
        xs_opt[:, 0],   # x 
        xs_opt[:, 2],   # x_dot 
        xs_opt[:, 1],   # theta 
        xs_opt[:, 3]    # theta_dot 
    ], axis=1)          # 最终形状 (nsteps+1, 4) 
 
    us_opt = np.asarray(us_opt)  # shape (nsteps, 1) 
 
    # ===== 添加噪声 ===== 
    xs_noisy, us_noisy = add_noise_to_trajectory( 
        xs_converted[:-1],  # 去掉最后一个状态 (nsteps, 4) 
        us_opt,              # (nsteps, 1) 
        state_noise_std=STATE_NOISE_STD, 
        control_noise_std=CONTROL_NOISE_STD 
    ) 
 
    # 保存带噪声的轨迹 
    all_states.append(xs_noisy)   # (nsteps, 4) 
    all_controls.append(us_noisy) # (nsteps, 1) 
 
print(f"成功生成 {success_count} 条轨迹（目标 {N_SAMPLES}）") 
 
# 如果没有成功样本，提前退出 
if success_count == 0: 
    print("没有成功生成的轨迹，请检查求解器设置或初始状态范围。") 
    exit(1) 
 
# =============================== 
# 8️⃣ 将轨迹整理为 (N, T, 4) 和 (N, T, 1) 格式并保存 
# =============================== 
# 将所有轨迹堆叠为三维张量： (成功条数, 步数, 特征维度) 
X = np.stack(all_states, axis=0)      # shape (success_count, nsteps, 4) 
U = np.stack(all_controls, axis=0)    # shape (success_count, nsteps, 1) 
 
print("最终数据形状:") 
print("X:", X.shape)   # 应为 (N, 500, 4) 
print("U:", U.shape)   # 应为 (N, 500, 1) 
 
# 打印数据统计 
print("\n数据统计:") 
print(f"  X - min: {X.min():.4f}, max: {X.max():.4f}, mean: {X.mean():.4f}, std: {X.std():.4f}") 
print(f"  U - min: {U.min():.4f}, max: {U.max():.4f}, mean: {U.mean():.4f}, std: {U.std():.4f}") 
 
# 转换为 torch 张量（float32） 
X_tensor = torch.tensor(X, dtype=torch.float32) 
U_tensor = torch.tensor(U, dtype=torch.float32) 
 
# 保存为 .pt 文件（原始轨迹，供 filter_data.py 过滤）
torch.save(X_tensor, os.path.join(SAVE_DIR, "features_raw.pt"))
torch.save(U_tensor, os.path.join(SAVE_DIR, "labels_raw.pt"))

print(f"\n数据已保存到 {SAVE_DIR}/features_raw.pt 和 labels_raw.pt")
print(f"总轨迹数: {X.shape[0]}") 
print(f"每条轨迹步数: {X.shape[1]}")
