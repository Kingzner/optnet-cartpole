import torch
import os

# 输入：generate_dataset.py 生成的原始轨迹
# 输出：过滤后用于训练/评估的轨迹（文件名与训练脚本期望的一致）
data_dir = "cartpole_data"
X = torch.load(os.path.join(data_dir, "features_raw.pt"))
Y = torch.load(os.path.join(data_dir, "labels_raw.pt"))

print("=" * 70)
print("检查并过滤极端轨迹")
print("=" * 70)

print(f"\n原始数据: {X.shape[0]} 条轨迹")

# 找出哪些轨迹有极端值
good_trajectories = []
bad_trajectories = []

for i in range(X.shape[0]):
    max_val = X[i].abs().max()
    if max_val < 20:  # 阈值：小于20
        good_trajectories.append(i)
    else:
        bad_trajectories.append((i, max_val))

print(f"\n好的轨迹（|X| < 20）: {len(good_trajectories)} 条")
print(f"坏的轨迹（|X| >= 20）: {len(bad_trajectories)} 条")

if bad_trajectories:
    print(f"\n坏轨迹详情（前10条）:")
    for idx, max_val in bad_trajectories[:10]:
        print(f"  轨迹 {idx}: max |X| = {max_val:.2f}")

# 过滤出好的轨迹
X_good = X[good_trajectories]
Y_good = Y[good_trajectories]

print(f"\n过滤后数据:")
print(f"  X shape: {X_good.shape}")
print(f"  Y shape: {Y_good.shape}")
print(f"  X range: {X_good.min().round(2)} ~ {X_good.max().round(2)}")
print(f"  Y range: {Y_good.min().round(2)} ~ {Y_good.max().round(2)}")

# 保存过滤后的数据
save_path_X = os.path.join(data_dir, "features.pt")
save_path_Y = os.path.join(data_dir, "labels.pt")
torch.save(X_good, save_path_X)
torch.save(Y_good, save_path_Y)
print(f"\n已保存过滤后数据:")
print(f"  {save_path_X}")
print(f"  {save_path_Y}")

print("\n" + "=" * 70)
print("完成！")
print("=" * 70)
