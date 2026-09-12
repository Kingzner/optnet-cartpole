import os
import sys

# Allow this script to be run directly from the repository root, e.g.
#     python scripts/train/train_unconstrained.py
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import argparse

from cartpole.models import OptNetCartPoleMPC

###############################
# 命令行参数解析 - 无约束版本
###############################
parser = argparse.ArgumentParser(description='OptNet CartPole MPC Training (UNCONSTRAINED)')
parser.add_argument('--N', type=int, default=10, help='MPC prediction horizon (default: 10)')
parser.add_argument('--u-max', type=float, default=10.0, help='Control input max magnitude (default: 10.0)')
parser.add_argument('--u-min', type=float, default=-10.0, help='Control input min magnitude (default: -10.0)')
parser.add_argument('--x-max', type=float, default=100.0, help='Position constraint upper bound (default: 100.0)')
parser.add_argument('--x-min', type=float, default=-100.0, help='Position constraint lower bound (default: -100.0)')
parser.add_argument('--reg', type=float, default=1.0, help='QP regularization epsilon (default: 1.0)')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs (default: 50)')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
parser.add_argument('--output-dir', type=str, default='outputs/relaxed', help='Output directory')
parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
args = parser.parse_args()

BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.epochs
LR = args.lr
WORKDIR = args.output_dir
os.makedirs(WORKDIR, exist_ok=True)

def load_data(data_dir="data", normalize=True, val_ratio=0.1, test_ratio=0.2):
    X = torch.load(os.path.join(data_dir, "features.pt"))
    Y = torch.load(os.path.join(data_dir, "labels.pt"))

    X = X.unsqueeze(-1).float()
    Y = Y.unsqueeze(-1).float()

    N = X.shape[0]

    perm = torch.randperm(N)
    X = X[perm]
    Y = Y[perm]

    n_train = int(N * (1.0 - val_ratio - test_ratio))
    n_val = int(N * val_ratio)

    X_train = X[:n_train]
    Y_train = Y[:n_train]
    X_val = X[n_train:n_train + n_val]
    Y_val = Y[n_train:n_train + n_val]
    X_test = X[n_train + n_val:]
    Y_test = Y[n_train + n_val:]

    if normalize:
        x_mean = X_train.mean(dim=(0, 1), keepdim=True)
        x_std = X_train.std(dim=(0, 1), keepdim=True) + 1e-8
        y_mean = Y_train.mean(dim=(0, 1), keepdim=True)
        y_std = Y_train.std(dim=(0, 1), keepdim=True) + 1e-8
        norm_stats = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}
    else:
        norm_stats = None

    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, norm_stats

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    n_samples = 0
    for X, Y in loader:
        X = X.to(device)
        Y = Y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = nn.MSELoss()(pred, Y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.shape[0]
        n_samples += X.shape[0]
    return total_loss / n_samples

def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0
    n_samples = 0
    with torch.no_grad():
        for X, Y in loader:
            X = X.to(device)
            Y = Y.to(device)
            pred = model(X)
            loss = nn.MSELoss()(pred, Y)
            total_loss += loss.item() * X.shape[0]
            n_samples += X.shape[0]
    return total_loss / n_samples

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, norm_stats = load_data(data_dir=args.data_dir)
    
    model = OptNetCartPoleMPC(
        N=args.N,
        u_min=args.u_min,
        u_max=args.u_max,
        x_min=args.x_min,
        x_max=args.x_max,
        q_reg_eps=args.reg
    ).to(device)

    if norm_stats is not None:
        model.set_norm_stats(norm_stats, device=device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    best_loss = float('inf')

    train_losses = []
    val_losses = []

    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = eval_epoch(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(WORKDIR, "cartpole_optnet_best_2B.pth"))

    torch.save(model.state_dict(), os.path.join(WORKDIR, "cartpole_optnet_final_2B.pth"))
    if norm_stats is not None:
        torch.save(norm_stats, os.path.join(WORKDIR, "norm_stats.pt"))

    np.savetxt(os.path.join(WORKDIR, "train.csv"), np.array(train_losses), delimiter=',')
    np.savetxt(os.path.join(WORKDIR, "val.csv"), np.array(val_losses), delimiter=',')

    print(f"Final Test Loss: {eval_epoch(model, test_loader, device):.6f}")

if __name__ == "__main__":
    main()
