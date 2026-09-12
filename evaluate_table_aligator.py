import os

import numpy as np
import torch

from models import OptNetCartPoleMPC

try:
    import aligator
    from aligator import manifolds
    from utils import create_cartpole
except ImportError as exc:
    raise ImportError(
        "evaluate_table_aligator.py needs aligator and utils.py. "
        "Upload the same utils.py used by your data-generation script, "
        "and run in an environment where aligator/pinocchio are installed."
    ) from exc


DT = 0.01
NUM_EPISODES = 50
NUM_STEPS = 500
POS_LIMIT = 1.0
ANGLE_THRESHOLD = 0.2
STABLE_STEPS = 25
N_VALUES = [1, 2, 5, 8, 10, 12, 15, 18]


def torch_load(path, **kwargs):
    try:
        return torch.load(path, weights_only=True, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def load_features():
    for path in ["features.pt", os.path.join("cartpole_data", "features.pt")]:
        if os.path.exists(path):
            return torch_load(path, map_location="cpu").float()
    raise FileNotFoundError("Could not find features.pt or cartpole_data/features.pt")


def load_loss_data(model_dir):
    val_path = os.path.join(model_dir, "val.csv")
    val_losses = []

    if os.path.exists(val_path):
        with open(val_path, "r") as f:
            for line in f.readlines()[1:]:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    val_losses.append(float(parts[1]))

    return val_losses


def make_dynamics():
    pin_model, _, _, _, _ = create_cartpole(1)
    nu = 1

    act_mat = np.zeros((2, nu))
    act_mat[0, 0] = 1.0

    space = manifolds.MultibodyPhaseSpace(pin_model)
    cont_dyn = aligator.dynamics.MultibodyFreeFwdDynamics(space, act_mat)
    disc_dyn = aligator.dynamics.IntegratorSemiImplEuler(cont_dyn, DT)
    return disc_dyn


def model_to_aligator_state(x_model_order):
    # Model/data order: [x, x_dot, theta, theta_dot]
    # Aligator order:   [x, theta, x_dot, theta_dot]
    return np.array(
        [
            x_model_order[0],
            x_model_order[2],
            x_model_order[1],
            x_model_order[3],
        ],
        dtype=float,
    )


def aligator_to_model_state(x_aligator_order):
    return np.array(
        [
            x_aligator_order[0],
            x_aligator_order[2],
            x_aligator_order[1],
            x_aligator_order[3],
        ],
        dtype=float,
    )


def step_aligator(disc_dyn, x_model_order, u_phys):
    x_aligator = model_to_aligator_state(x_model_order)
    xs = aligator.rollout(disc_dyn, x_aligator, [np.array([u_phys], dtype=float)])
    return aligator_to_model_state(np.asarray(xs[-1]))


def is_success(positions, angles):
    stable_count = 0
    for x, theta in zip(positions, angles):
        if abs(x) < POS_LIMIT and abs(theta) < ANGLE_THRESHOLD:
            stable_count += 1
            if stable_count >= STABLE_STEPS:
                return True
        else:
            stable_count = 0
    return False


def load_model(model_dir, device):
    model_path = os.path.join(model_dir, "cartpole_optnet_best_2B.pth")
    norm_path = os.path.join(model_dir, "norm_stats.pt")

    if not os.path.exists(model_path) or not os.path.exists(norm_path):
        return None, None

    N = int(os.path.basename(model_dir).split("N")[1].split("_")[0])
    model = OptNetCartPoleMPC(N=N, x_min=-1.0, x_max=1.0, dt=DT).to(device)

    state = torch_load(model_path, map_location=device)
    for key in list(state.keys()):
        if key.endswith("G_base") or key.endswith("h_base"):
            del state[key]
    model.load_state_dict(state, strict=False)

    norm = torch_load(norm_path, map_location=device)
    model.set_norm_stats(norm, device=device)
    model.eval()
    return model, norm


def evaluate_model(model_dir, X, disc_dyn, num_episodes=NUM_EPISODES):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, norm = load_model(model_dir, device)
    if model is None:
        return None

    x_mean = norm["x_mean"].cpu().numpy().reshape(-1)[:4]
    x_std = norm["x_std"].cpu().numpy().reshape(-1)[:4]
    y_mean = norm["y_mean"].cpu().numpy().reshape(-1)[0]
    y_std = norm["y_std"].cpu().numpy().reshape(-1)[0]

    rng = np.random.default_rng(42)
    indices = rng.choice(X.shape[0], num_episodes, replace=False)

    success_count = 0
    violated_episodes = 0
    max_violation = 0.0
    state_rmse_values = []

    for idx in indices:
        x = X[idx, 0, :].cpu().numpy().astype(float)
        positions = []
        angles = []
        violated = False

        for _ in range(NUM_STEPS):
            positions.append(x[0])
            angles.append(x[2])

            violation = abs(x[0]) - POS_LIMIT
            if violation > 0:
                violated = True
                max_violation = max(max_violation, float(violation))

            x_norm = (x - x_mean) / x_std
            x_tensor = torch.tensor(x_norm, dtype=torch.float32).view(1, 1, 4, 1).to(device)

            with torch.no_grad():
                u_norm = model(x_tensor)

            u_phys = float(u_norm.cpu().numpy().reshape(-1)[0] * y_std + y_mean)
            u_phys = float(np.clip(u_phys, -10.0, 10.0))
            x = step_aligator(disc_dyn, x, u_phys)

            if abs(x[0]) > 3.0 or abs(x[2]) > 2.0 * np.pi:
                break

        positions = np.asarray(positions)
        angles = np.asarray(angles)

        if is_success(positions, angles):
            success_count += 1
        if violated:
            violated_episodes += 1

        state_rmse_values.append(np.sqrt(np.mean(positions**2 + angles**2)))

    return {
        "success_rate": success_count / num_episodes,
        "violation_rate": violated_episodes / num_episodes,
        "max_violation": max_violation,
        "test_rmse": float(np.mean(state_rmse_values)),
    }


def main():
    X = load_features()
    disc_dyn = make_dynamics()
    results = []

    print("=" * 95)
    print(
        f"{'Horizon N':^12} {'Validation loss':^18} {'Rollout RMSE':^12} "
        f"{'Violation rate':^16} {'Max violation':^16} {'Success rate':^14}"
    )
    print("-" * 95)

    for N in N_VALUES:
        model_dir = f"results/N{N}_x1.0_reg1.0"
        if not os.path.exists(model_dir):
            print(f"{N:^12} {'-' * 18} {'-' * 12} {'-' * 16} {'-' * 16} {'-' * 14}")
            continue

        val_losses = load_loss_data(model_dir)
        val_loss = np.mean(val_losses[-5:]) if val_losses else np.nan
        eval_results = evaluate_model(model_dir, X, disc_dyn)

        if eval_results is None:
            rollout_rmse = np.nan
            violation_rate = np.nan
            max_violation = np.nan
            success_rate = np.nan
        else:
            rollout_rmse = eval_results["test_rmse"]
            violation_rate = eval_results["violation_rate"]
            max_violation = eval_results["max_violation"]
            success_rate = eval_results["success_rate"]

        print(
            f"{N:^12} {val_loss:^18.4f} {rollout_rmse:^12.4f} "
            f"{violation_rate:^16.1%} {max_violation:^16.3f} {success_rate:^14.1%}"
        )

        results.append(
            {
                "N": N,
                "val_loss": val_loss,
                "rollout_rmse": rollout_rmse,
                "violation_rate": violation_rate,
                "max_violation": max_violation,
                "success_rate": success_rate,
            }
        )

    print("=" * 95)

    with open("table_results_aligator.csv", "w") as f:
        f.write("N,Validation loss,Rollout RMSE,Violation rate,Max violation,Success rate\n")
        for r in results:
            f.write(
                f"{r['N']},{r['val_loss']:.4f},{r['rollout_rmse']:.4f},"
                f"{r['violation_rate']:.4f},{r['max_violation']:.4f},"
                f"{r['success_rate']:.4f}\n"
            )

    print("\nResults saved to: table_results_aligator.csv")


if __name__ == "__main__":
    main()
