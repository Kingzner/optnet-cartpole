import os
import sys

# Allow this script to be run directly from the repository root, e.g.
#     python scripts/figures/plot_rollout_comparison.py
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class BaselineMLP(nn.Module):
    def __init__(self, input_dim=4, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128, 64]

        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.ReLU())
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        batch_size, horizon, _, _ = x.shape
        x_flat = x.view(batch_size * horizon, 4)
        u = self.net(x_flat)
        return u.view(batch_size, horizon, 1, 1)


def load_episodes(data_dir, num_episodes, limit, feasibility_margin):
    x = torch.load(os.path.join(data_dir, "features.pt"), weights_only=True).float()
    x_next_open_loop = x[:, :, 0] + 0.02 * x[:, :, 1]
    feasible = x_next_open_loop.abs().amax(dim=1) <= (limit - feasibility_margin)
    feasible_indices = torch.nonzero(feasible).flatten()
    if feasible_indices.numel() == 0:
        raise RuntimeError("No feasible episodes found for the requested position limit.")

    # Pick episodes across the whole dataset so the figure looks like a family of rollouts.
    if num_episodes > feasible_indices.numel():
        num_episodes = int(feasible_indices.numel())
    pick = torch.linspace(0, feasible_indices.numel() - 1, num_episodes).round().long()
    indices = feasible_indices[pick]
    return x[indices].unsqueeze(-1), indices.cpu().numpy()


def normalize_input(x_phys, norm_stats):
    x_mean = norm_stats["x_mean"].float().view(1, 1, 4, 1)
    x_std = norm_stats["x_std"].float().view(1, 1, 4, 1)
    return (x_phys - x_mean) / (x_std + 1e-8)


def load_models(limit, device):
    from cartpole.models import OptNetCartPoleMPC

    optnet_workdir = "outputs/main"
    baseline_workdir = "outputs/baseline"

    optnet_norm = torch.load(
        os.path.join(optnet_workdir, "norm_stats.pt"),
        weights_only=True,
        map_location=device,
    )
    baseline_norm = torch.load(
        os.path.join(baseline_workdir, "norm_stats_baseline.pt"),
        weights_only=True,
        map_location=device,
    )

    optnet_model = OptNetCartPoleMPC(
        N=10,
        u_min=-10.0,
        u_max=10.0,
        x_min=-limit,
        x_max=limit,
        dt=0.02,
    ).to(device)
    state = torch.load(
        os.path.join(optnet_workdir, "cartpole_optnet_best_2B.pth"),
        weights_only=True,
        map_location=device,
    )
    for key in list(state.keys()):
        if key.endswith("G_base") or key.endswith("h_base"):
            del state[key]

    optnet_model.load_state_dict(state, strict=False)
    optnet_model.set_norm_stats(optnet_norm, device=device)
    optnet_model.eval()

    baseline_model = BaselineMLP().to(device)
    baseline_model.load_state_dict(
        torch.load(
            os.path.join(baseline_workdir, "best_baseline.pth"),
            weights_only=True,
            map_location=device,
        )
    )
    baseline_model.eval()

    return optnet_model, baseline_model, optnet_norm, baseline_norm


def predict_optnet_states(model, x_phys, norm_stats, device, chunk_size):
    states = []
    x_norm = normalize_input(x_phys, norm_stats).to(device)
    nx = model.nx

    with torch.no_grad():
        for start in range(0, x_norm.shape[0], chunk_size):
            batch = x_norm[start : start + chunk_size]
            _, (_, _, _, _, _, _, z_phys) = model(batch, return_qp=True)
            z = z_phys.view(batch.shape[0], batch.shape[1], -1)
            states.append(z[:, :, :nx].cpu().numpy())

    return np.concatenate(states, axis=0)


def rollout_baseline_states(model, x_phys, norm_stats):
    baseline_norm = {
        key: value.detach().cpu().numpy().reshape(-1)
        for key, value in norm_stats.items()
    }
    x_mean = baseline_norm["x_mean"][:4]
    x_std = baseline_norm["x_std"][:4]
    y_mean = baseline_norm["y_mean"][0]
    y_std = baseline_norm["y_std"][0]

    a_d = np.array(
        [
            [1.0, 0.02, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.02],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    b_d = np.array([[0.0], [0.02], [0.0], [0.02]])

    x0 = x_phys[:, 0, :, 0].cpu().numpy()
    all_states = np.zeros((x_phys.shape[0], x_phys.shape[1], 4), dtype=np.float64)
    all_states[:, 0, :] = x0

    for episode in range(x_phys.shape[0]):
        x = x0[episode].copy()
        for t in range(1, x_phys.shape[1]):
            x_norm = (x - x_mean) / (x_std + 1e-8)
            x_tensor = torch.tensor(x_norm, dtype=torch.float32).view(1, 1, 4, 1)
            with torch.no_grad():
                u_norm = model(x_tensor)
            u_phys = u_norm.cpu().numpy().reshape(-1)[0] * y_std + y_mean
            x = a_d @ x + (b_d @ np.array([[u_phys]])).reshape(-1)
            all_states[episode, t, :] = x

    return all_states


def count_violations(positions, limit):
    mask = np.logical_or(positions > limit, positions < -limit)
    return int(mask.sum()), int(mask.shape[0] * mask.shape[1])


def plot_position_constraints(optnet_states, mlp_states, limit, output_dir, dt):
    t = (np.arange(optnet_states.shape[1]) + 1) * dt
    optnet_pos = optnet_states[:, :, 0]
    mlp_pos = mlp_states[:, :, 0]
    optnet_violations, total = count_violations(optnet_pos, limit)
    mlp_violations, _ = count_violations(mlp_pos, limit)

    all_pos = np.concatenate([optnet_pos.reshape(-1), mlp_pos.reshape(-1)])
    y_min = min(-1.25 * limit, float(np.nanmin(all_pos)) - 0.15)
    y_max = max(1.25 * limit, float(np.nanmax(all_pos)) + 0.15)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4), sharex=True, sharey=True)
    panels = [
        (axes[0], optnet_pos, "OptNet", "#1f77b4", optnet_violations, "#1d4ed8"),
        (axes[1], mlp_pos, "MLP baseline", "#ff9f0a", mlp_violations, "#ea580c"),
    ]

    for ax, positions, label, color, violations, strong_color in panels:
        ax.set_facecolor("#fbfdff")
        ax.fill_between(t, -limit, limit, color="#dbeafe", alpha=0.46, label="allowed region")
        ax.axhspan(limit, y_max, color="#fee2e2", alpha=0.42, zorder=0)
        ax.axhspan(y_min, -limit, color="#fee2e2", alpha=0.42, zorder=0)
        ax.axhline(limit, color="black", linestyle="--", linewidth=1.9, label="constraint boundary")
        ax.axhline(-limit, color="black", linestyle="--", linewidth=1.8)
        for i in range(positions.shape[0]):
            trajectory = positions[i]
            ax.plot(t, trajectory, color=color, alpha=0.18, linewidth=0.9)
            violation_mask = np.logical_or(trajectory > limit, trajectory < -limit)
            if violation_mask.any():
                violation_trace = np.ma.masked_where(~violation_mask, trajectory)
                ax.plot(t, violation_trace, color="#dc2626", alpha=0.64, linewidth=1.1)
        ax.plot([], [], color=color, alpha=0.75, linewidth=1.3, label="trajectories")
        if violations > 0:
            ax.plot([], [], color="#dc2626", alpha=0.85, linewidth=1.4, label="violation segments")
        ax.set_title(f"{label} ({positions.shape[0]} trajectories)", fontsize=12, pad=10)
        ax.set_xlabel("Time (s)", fontsize=11)
        ax.set_xlim(dt, optnet_states.shape[1] * dt)
        ax.set_xticks(np.arange(1, int(np.ceil(optnet_states.shape[1] * dt)) + 1))
        ax.grid(True, linestyle="--", alpha=0.45)
        violation_rate = 100.0 * violations / total
        ax.text(
            0.02,
            0.95,
            f"violations: {violations}/{total}\nrate: {violation_rate:.1f}%",
            transform=ax.transAxes,
            va="top",
            fontsize=10,
            bbox=dict(facecolor="white", edgecolor=strong_color, linewidth=1.4, alpha=0.95),
        )

    axes[0].set_ylabel("Cart position (m)", fontsize=11)
    axes[0].legend(loc="lower left", fontsize=9)
    axes[1].legend(loc="lower left", fontsize=9)
    axes[0].set_ylim(y_min, y_max)
    fig.suptitle("Cart-Position Constraint Satisfaction: OptNet vs MLP", fontsize=15, y=0.985)
    fig.text(0.5, 0.925, f"Constraint boundary: +/-{limit:.1f} m", ha="center", fontsize=11)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.895])

    png_path = os.path.join(output_dir, "optnet_vs_mlp_many_trajectories_limit_1.0.png")
    pdf_path = os.path.join(output_dir, "optnet_vs_mlp_many_trajectories_limit_1.0.pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path, optnet_violations, mlp_violations, total


def draw_cartpole_family(ax, states, title, color_map_name, limit, max_trajs=10, poses_per_traj=34):
    cmap = plt.get_cmap(color_map_name)
    traj_count = min(max_trajs, states.shape[0])
    traj_indices = np.linspace(0, states.shape[0] - 1, traj_count).round().astype(int)
    time_indices = np.linspace(0, states.shape[1] - 1, poses_per_traj).round().astype(int)
    pole_length = 0.45

    ax.axvline(limit, color="black", linestyle="--", linewidth=1.3)
    ax.axvline(-limit, color="black", linestyle="--", linewidth=1.3)
    ax.axhline(0.0, color="#555555", linewidth=1.0, alpha=0.55)

    for j, episode_idx in enumerate(traj_indices):
        color = cmap(j / max(traj_count - 1, 1))
        x = states[episode_idx, :, 0]
        theta = states[episode_idx, :, 2]
        ax.plot(x, np.zeros_like(x), color=color, alpha=0.23, linewidth=1.0)

        for time_idx in time_indices:
            cart_x = x[time_idx]
            pole_theta = theta[time_idx]
            tip_x = cart_x + pole_length * np.sin(pole_theta)
            tip_y = pole_length * np.cos(pole_theta)
            ax.plot([cart_x, tip_x], [0.0, tip_y], color=color, alpha=0.62, linewidth=1.0)
            ax.scatter(cart_x, 0.0, s=9, color=color, alpha=0.72, linewidths=0)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Cart position (m)", fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.65, 1.65)
    ax.set_ylim(-0.12, 0.55)
    ax.grid(True, linestyle="--", alpha=0.28)


def plot_cartpole_pose_traces(optnet_states, mlp_states, limit, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    draw_cartpole_family(axes[0], optnet_states, "OptNet", "viridis", limit)
    draw_cartpole_family(axes[1], mlp_states, "MLP baseline", "plasma", limit)
    axes[0].set_ylabel("Pole height (m)", fontsize=11)
    fig.suptitle("Cart-pole pose traces under cart-position constraints", fontsize=15)
    fig.tight_layout()

    png_path = os.path.join(output_dir, "optnet_vs_mlp_cartpole_pose_traces_limit_1.0.png")
    pdf_path = os.path.join(output_dir, "optnet_vs_mlp_cartpole_pose_traces_limit_1.0.pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=float, default=1.0)
    parser.add_argument("--num-episodes", type=int, default=60)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="figures")
    parser.add_argument("--feasibility-margin", type=float, default=0.02)
    parser.add_argument("--dt", type=float, default=0.01)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cpu")

    x_phys, indices = load_episodes(
        args.data_dir,
        args.num_episodes,
        args.limit,
        args.feasibility_margin,
    )
    optnet_model, baseline_model, optnet_norm, baseline_norm = load_models(args.limit, device)

    print(f"Selected dataset episodes: {indices.tolist()}")
    print("Running OptNet QP predictions...")
    optnet_states = predict_optnet_states(
        optnet_model,
        x_phys,
        optnet_norm,
        device,
        chunk_size=args.chunk_size,
    )

    print("Rolling out MLP baseline...")
    mlp_states = rollout_baseline_states(baseline_model, x_phys, baseline_norm)

    pos_png, pos_pdf, opt_vio, mlp_vio, total = plot_position_constraints(
        optnet_states,
        mlp_states,
        args.limit,
        args.output_dir,
        args.dt,
    )
    pose_png, pose_pdf = plot_cartpole_pose_traces(
        optnet_states,
        mlp_states,
        args.limit,
        args.output_dir,
    )

    print(f"OptNet violations: {opt_vio}/{total}")
    print(f"MLP baseline violations: {mlp_vio}/{total}")
    print(f"Saved: {pos_png}")
    print(f"Saved: {pos_pdf}")
    print(f"Saved: {pose_png}")
    print(f"Saved: {pose_pdf}")


if __name__ == "__main__":
    main()
