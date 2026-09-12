import os

import numpy as np
import torch


FEATURE_CANDIDATES = [
    "features.pt",
    os.path.join("cartpole_data", "features.pt"),
    "features (7).pt",
]


def find_features_path():
    for path in FEATURE_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Could not find features.pt. Checked: "
        + ", ".join(FEATURE_CANDIDATES)
    )


def check_success(traj, pos_limit=1.0, angle_threshold=0.1, stable_steps=50):
    # Expected state order: [x, x_dot, theta, theta_dot]
    x = traj[:, 0]
    theta = traj[:, 2]

    stable_count = 0
    for t in range(len(traj)):
        if abs(x[t]) < pos_limit and abs(theta[t]) < angle_threshold:
            stable_count += 1
            if stable_count >= stable_steps:
                return True, t - stable_steps + 1
        else:
            stable_count = 0

    return False, -1


def evaluate(X, angle_threshold, stable_steps):
    success_count = 0
    success_times = []

    for i in range(X.shape[0]):
        success, t = check_success(
            X[i],
            pos_limit=1.0,
            angle_threshold=angle_threshold,
            stable_steps=stable_steps,
        )
        if success:
            success_count += 1
            success_times.append(t)

    rate = success_count / X.shape[0]
    avg_t = float(np.mean(success_times)) if success_times else -1.0

    print("=" * 60)
    print(f"angle_threshold = {angle_threshold}")
    print(f"stable_steps    = {stable_steps}")
    print(f"success         = {success_count}/{X.shape[0]} ({rate:.2%})")
    print(f"avg start step  = {avg_t:.1f}")


def main():
    features_path = find_features_path()
    X = torch.load(features_path, map_location="cpu").float().numpy()

    print("Expert trajectory success check")
    print(f"features path: {features_path}")
    print(f"X shape: {X.shape}")
    print(f"x range:     {X[:, :, 0].min():.4f} to {X[:, :, 0].max():.4f}")
    print(f"theta range: {X[:, :, 2].min():.4f} to {X[:, :, 2].max():.4f}")

    evaluate(X, angle_threshold=0.1, stable_steps=50)
    evaluate(X, angle_threshold=0.2, stable_steps=50)
    evaluate(X, angle_threshold=0.3, stable_steps=25)


if __name__ == "__main__":
    main()
