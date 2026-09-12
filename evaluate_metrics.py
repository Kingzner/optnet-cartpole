import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models import OptNetCartPoleMPC

WORKDIR = 'work_cartpole_filtered'
DATA_DIR = 'cartpole_data'

# Load data
X = torch.load(DATA_DIR + '/features.pt', weights_only=True).unsqueeze(-1).float()
Y = torch.load(DATA_DIR + '/labels.pt', weights_only=True).unsqueeze(-1).float()

N = X.shape[0]
np.random.seed(42)
indices = np.random.permutation(N)
n_train = int(0.7 * N)
n_val = int(0.1 * N)
train_idx = indices[:n_train]
val_idx = indices[n_train:n_train+n_val]
test_idx = indices[n_train+n_val:]

X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
Y_train, Y_val, Y_test = Y[train_idx], Y[val_idx], Y[test_idx]

# Load norm stats
norm_stats = torch.load(WORKDIR + '/norm_stats.pt', weights_only=True)
x_mean = norm_stats['x_mean']
x_std = norm_stats['x_std']
y_mean = norm_stats['y_mean']
y_std = norm_stats['y_std']

# Normalize
X_train = (X_train - x_mean) / x_std
X_val = (X_val - x_mean) / x_std
X_test = (X_test - x_mean) / x_std
Y_train = (Y_train - y_mean) / y_std
Y_val = (Y_val - y_mean) / y_std
Y_test = (Y_test - y_mean) / y_std

# Create dataloaders
train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=32, shuffle=False)
val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=32, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=32, shuffle=False)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = OptNetCartPoleMPC(N=10).to(device)
model.load_state_dict(torch.load(WORKDIR + '/cartpole_optnet_best_2B.pth', map_location=device, weights_only=True))
model.set_norm_stats(norm_stats, device=device)
model.eval()

def evaluate_metrics(loader, device, norm_stats, split_name=''):
    y_mean = norm_stats['y_mean'].to(device)
    y_std = norm_stats['y_std'].to(device)
    
    all_pred = []
    all_true = []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            
            y_denorm = y * y_std + y_mean
            pred_denorm = pred * y_std + y_mean
            
            all_true.append(y_denorm.view(-1))
            all_pred.append(pred_denorm.view(-1))
    
    all_true = torch.cat(all_true)
    all_pred = torch.cat(all_pred)
    
    mse = torch.mean((all_pred - all_true) ** 2).item()
    rmse = mse ** 0.5
    mae = torch.mean(torch.abs(all_pred - all_true)).item()
    
    ss_res = torch.sum((all_true - all_pred) ** 2)
    ss_tot = torch.sum((all_true - torch.mean(all_true)) ** 2)
    r2 = (1 - ss_res / (ss_tot + 1e-8)).item()
    
    vx = all_true - torch.mean(all_true)
    vy = all_pred - torch.mean(all_pred)
    corr = (torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2) + 1e-8) * torch.sqrt(torch.sum(vy ** 2) + 1e-8))).item()
    
    print(f'=== {split_name} metrics (denormalized) ===')
    print(f'MSE : {mse:.6f}')
    print(f'RMSE: {rmse:.6f}')
    print(f'MAE : {mae:.6f}')
    print(f'R^2 : {r2:.6f}')
    print(f'Corr: {corr:.6f}')
    print()

evaluate_metrics(train_loader, device, norm_stats, split_name='Train')
evaluate_metrics(val_loader, device, norm_stats, split_name='Val')
evaluate_metrics(test_loader, device, norm_stats, split_name='Test')