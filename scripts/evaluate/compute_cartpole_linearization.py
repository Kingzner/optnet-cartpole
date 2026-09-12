import os
import sys

# Allow this script to be run directly from the repository root, e.g.
#     python scripts/evaluate/compute_cartpole_linearization.py
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse

import aligator
import numpy as np
from aligator import manifolds

from cartpole.utils import create_cartpole


def model_to_aligator_state(x_model_order):
    # Model/data order: [x, x_dot, theta, theta_dot]
    # Aligator order:   [x, theta, x_dot, theta_dot]
    return np.array(
        [x_model_order[0], x_model_order[2], x_model_order[1], x_model_order[3]],
        dtype=float,
    )


def aligator_to_model_state(x_aligator_order):
    return np.array(
        [x_aligator_order[0], x_aligator_order[2], x_aligator_order[1], x_aligator_order[3]],
        dtype=float,
    )


def rollout_one_step(disc_dyn, x_model_order, u):
    x_aligator = model_to_aligator_state(x_model_order)
    xs = aligator.rollout(disc_dyn, x_aligator, [np.array([u], dtype=float)])
    return aligator_to_model_state(np.asarray(xs[-1]))


def finite_difference_linearization(disc_dyn, x_ref, u_ref, eps_x, eps_u):
    nx = x_ref.shape[0]
    A = np.zeros((nx, nx))
    B = np.zeros((nx, 1))

    f0 = rollout_one_step(disc_dyn, x_ref, u_ref)

    for i in range(nx):
        dx = np.zeros(nx)
        dx[i] = eps_x
        fp = rollout_one_step(disc_dyn, x_ref + dx, u_ref)
        fm = rollout_one_step(disc_dyn, x_ref - dx, u_ref)
        A[:, i] = (fp - fm) / (2.0 * eps_x)

    fp = rollout_one_step(disc_dyn, x_ref, u_ref + eps_u)
    fm = rollout_one_step(disc_dyn, x_ref, u_ref - eps_u)
    B[:, 0] = (fp - fm) / (2.0 * eps_u)

    return A, B, f0


def main():
    parser = argparse.ArgumentParser(
        description="Compute linearized cartpole dynamics from aligator rollout."
    )
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--eps-x", type=float, default=1e-5)
    parser.add_argument("--eps-u", type=float, default=1e-5)
    parser.add_argument("--output", type=str, default="cartpole_linear_dt001.npz")
    args = parser.parse_args()

    pin_model, _, _, _, _ = create_cartpole(1)
    act_mat = np.zeros((2, 1))
    act_mat[0, 0] = 1.0

    space = manifolds.MultibodyPhaseSpace(pin_model)
    cont_dyn = aligator.dynamics.MultibodyFreeFwdDynamics(space, act_mat)
    disc_dyn = aligator.dynamics.IntegratorSemiImplEuler(cont_dyn, args.dt)

    x_ref = np.zeros(4)
    u_ref = 0.0
    A, B, f0 = finite_difference_linearization(
        disc_dyn, x_ref=x_ref, u_ref=u_ref, eps_x=args.eps_x, eps_u=args.eps_u
    )

    np.savez(
        args.output,
        A_d=A,
        B_d=B,
        x_ref=x_ref,
        u_ref=np.array([u_ref]),
        f0=f0,
        dt=np.array([args.dt]),
    )

    print(f"Saved linearized dynamics to: {os.path.abspath(args.output)}")
    print("A_d:")
    print(A)
    print("B_d:")
    print(B)
    print("f(x_ref, u_ref):")
    print(f0)


if __name__ == "__main__":
    main()
