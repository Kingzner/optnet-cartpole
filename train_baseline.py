"""
倒立摆基线模型：MLP（无 QP 层）
输入：状态 (4维)
输出：控制力 (1维)
损失：MSE
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# ==================== 模型定义 ====================
class BaselineMLP(nn.Module):
    def __init__(self, input_dim=4, hidden_dims=[128, 128, 64]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.ReLU())
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, T, 4, 1) -> (B*T, 4)
        B, T, _, _ = x.shape
        x_flat = x.view(B*T, 4)
        u = self.net(x_flat)          # (B*T, 1)
        return u.view(B, T, 1, 1)

# ==================== 超参数配置 ====================
BATCH_SIZE = 32
NUM_EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
WORKDIR = "work_cartpole_baseline"
os.makedirs(WORKDIR, exist_ok=True)

# 数据路径（与原始 OptNet 训练相同）
DATA_DIR = "cartpole_data"
FEATURES_PATH = os.path.join(DATA_DIR, "features.pt")
LABELS_PATH = os.path.join(DATA_DIR, "labels.pt")

# ==================== 加载数据 ====================
print("加载数据...")
X = torch.load(FEATURES_PATH)        # (N, T, 4)
U = torch.load(LABELS_PATH)          # (N, T, 1)

N, T, _ = X.shape
print(f"总样本数: {N}, 轨迹长度: {T}")

# 添加最后一维 (B,T,4,1) 和 (B,T,1,1)
X = X.unsqueeze(-1).float()
U = U.unsqueeze(-1).float()

# 划分训练/验证/测试（与 OptNet 一致：70/10/20）
np.random.seed(42)
indices = np.random.permutation(N)
n_train = int(0.7 * N)
n_val = int(0.1 * N)
train_idx = indices[:n_train]
val_idx = indices[n_train:n_train+n_val]
test_idx = indices[n_train+n_val:]

X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
U_train, U_val, U_test = U[train_idx], U[val_idx], U[test_idx]

print(f"训练集: {len(train_idx)} 条轨迹")
print(f"验证集: {len(val_idx)} 条轨迹")
print(f"测试集: {len(test_idx)} 条轨迹")

# 归一化（使用训练集统计量）
x_mean = X_train.mean(dim=(0,1,3), keepdim=True)
x_std = X_train.std(dim=(0,1,3), keepdim=True) + 1e-8
y_mean = U_train.mean(dim=(0,1,3), keepdim=True)
y_std = U_train.std(dim=(0,1,3), keepdim=True) + 1e-8

X_train = (X_train - x_mean) / x_std
X_val   = (X_val   - x_mean) / x_std
X_test  = (X_test  - x_mean) / x_std
U_train = (U_train - y_mean) / y_std
U_val   = (U_val   - y_mean) / y_std
U_test  = (U_test  - y_mean) / y_std

# 保存统计量（可选，用于测试时反标准化）
norm_stats = {
    "x_mean": x_mean.cpu(),
    "x_std": x_std.cpu(),
    "y_mean": y_mean.cpu(),
    "y_std": y_std.cpu(),
}
torch.save(norm_stats, os.path.join(WORKDIR, "norm_stats_baseline.pt"))

# 创建 DataLoader
train_ds = TensorDataset(X_train, U_train)
val_ds   = TensorDataset(X_val, U_val)
test_ds  = TensorDataset(X_test, U_test)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# ==================== 模型、优化器、损失 ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

model = BaselineMLP().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
criterion = nn.MSELoss()

# 损失记录
train_csv = os.path.join(WORKDIR, "train_loss.csv")
val_csv   = os.path.join(WORKDIR, "val_loss.csv")
test_csv  = os.path.join(WORKDIR, "test_loss.csv")
for f, header in zip([train_csv, val_csv, test_csv], ['epoch,loss\n', 'epoch,loss\n', 'epoch,loss\n']):
    with open(f, 'w') as fp:
        fp.write(header)

# ==================== 训练循环 ====================
print("开始训练...")
best_val_loss = float('inf')
patience = 10
counter = 0

for epoch in range(1, NUM_EPOCHS + 1):
    # 训练
    model.train()
    train_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # 验证
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    # 测试
    test_loss = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            test_loss += loss.item()
    test_loss /= len(test_loader)

    # 记录
    with open(train_csv, 'a') as f:
        f.write(f"{epoch},{train_loss}\n")
    with open(val_csv, 'a') as f:
        f.write(f"{epoch},{val_loss}\n")
    with open(test_csv, 'a') as f:
        f.write(f"{epoch},{test_loss}\n")

    print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Test Loss: {test_loss:.6f}")

    # 学习率调整
    scheduler.step(val_loss)

    # 早停
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), os.path.join(WORKDIR, "best_baseline.pth"))
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

print(f"训练完成！最佳模型保存至 {WORKDIR}/best_baseline.pth")
print(f"损失曲线已保存至 {WORKDIR}/train_loss.csv, val_loss.csv, test_loss.csv")