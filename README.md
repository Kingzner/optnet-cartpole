# OptNet-Based Constrained Imitation Learning for Cart-Pole Control

Code and data for my final year project (SAT301): a cart-pole controller that learns from
expert demonstrations and produces its control output through a **differentiable quadratic
programming layer** (OptNet), so that the physical constraints — control force
`u ∈ [-10, 10] N` and cart position `x ∈ [-1.0, 1.0] m` — are part of the model structure
instead of being learned only from data.

The controller is compared against a plain MLP baseline trained on the same demonstrations.
The main finding is that the MLP reaches a slightly lower one-step prediction error, but the
OptNet-QP controller is the one that respects the cart-position bound during closed-loop
rollout.

## Key results

| Result | OptNet-QP | MLP baseline |
|---|---|---|
| Test MSE / RMSE (physical units) | 0.2932 / 0.5415 | — |
| Test R² / Pearson correlation | 0.8534 / 0.9244 | — |
| Cart-position violations over 30 000 rollout steps (±1.0 m) | **0** | 9386 (31.3 %) |

Prediction horizon sweep (validation loss vs. closed-loop success rate): the lowest
validation loss is obtained at `N = 10`, but the highest success rate (40 %, with a 0 %
cart-position violation rate) is obtained at `N = 12`. One-step supervised loss is therefore
not a sufficient criterion for choosing a controller.

## Repository layout

```
.
├── models.py                              # OptNetCartPoleMPC: feature network + differentiable QP layer
├── utils.py                               # cart-pole model builder (Pinocchio)
├── generate_dataset.py                    # expert trajectories via constrained trajectory optimization (aligator/ProxDDP)
├── filter_data.py                         # remove extreme trajectories (threshold 20) -> 994 kept
├── train_cartpole.py                      # train the OptNet-QP controller
├── train_baseline.py                      # train the MLP baseline
├── train_unconstrained.py                 # same QP structure, position bound relaxed (x_max = 100 m)
├── compute_cartpole_linearization.py      # optional: linearised dynamics from aligator rollouts
├── evaluate_metrics.py                    # prediction metrics in physical units (Table 3)
├── evaluate_table_aligator.py             # closed-loop horizon comparison (Table 5)
├── check_expert_success.py                # sanity check on the expert demonstrations
├── plot.py                                # loss curves (Figure 4)
├── plot_comparison.py                     # OptNet vs. MLP losses (Figure 5)
├── analyze_constrained_unconstrained.py   # constrained / relaxed-bound / MLP (Figure 6)
├── plot_optnet_vs_mlp_many_trajectories.py# rollout comparison (Figure 7) + violation counts (Table 4)
├── plot_qp_constraints_standalone.py      # sparse structure of G and vector h (Figures 8 and 9)
├── plot_all_trajectories_constraints.py   # dataset visualisation (Figure 1)
├── plot_all_control_forces_constraints.py # dataset visualisation (Figure 1)
├── cartpole_data/                         # dataset (see below)
├── work_cartpole_filtered/                # main OptNet-QP model, N = 10
├── work_cartpole_baseline/                # MLP baseline
├── results/                               # horizon sweep N = 1, 2, 5, 8, 10, 12, 15, 18 (Table 5)
├── results_unconstrained/                 # relaxed-bound model used in Figure 6
├── plots/                                 # figures as included in the thesis
└── extensions/                            # follow-up work (conditional / robustness experiments)
```

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` covers everything needed to train, evaluate and plot. Data generation and
the closed-loop rollout in `evaluate_table_aligator.py` additionally need Pinocchio and
aligator:

```bash
pip install -r requirements-aligator.txt
```

All commands below are meant to be run from the repository root.

## Reproducing the results

### Data (already included — this step is optional)

```bash
python generate_dataset.py     # 1000 trajectories -> cartpole_data/features_raw.pt, labels_raw.pt
python filter_data.py          # remove extreme trajectories -> cartpole_data/features.pt, labels.pt (994)
```

`generate_dataset.py` solves a constrained swing-up problem with ProxDDP for each randomly
sampled initial state, reorders the state from the simulator convention
`[x, θ, ẋ, θ̇]` to the learning convention `[x, ẋ, θ, θ̇]`, adds sensor/actuator noise and saves
the trajectories. `filter_data.py` then drops trajectories whose maximum absolute state value
exceeds 20.

| File | Shape | Meaning |
|---|---|---|
| `cartpole_data/features_raw.pt` / `labels_raw.pt` | (1000, 500, 4) / (1000, 500, 1) | raw generated trajectories |
| `cartpole_data/features.pt` / `labels.pt` | (994, 500, 4) / (994, 500, 1) | dataset used for training |

### Training

```bash
python train_cartpole.py                  # main model -> work_cartpole_filtered/
python train_baseline.py                  # MLP baseline -> work_cartpole_baseline/

# prediction-horizon sweep (Table 5)
for N in 1 2 5 8 10 12 15 18; do
  python train_cartpole.py --N $N --output-dir results/N${N}_x1.0_reg1.0
done

# relaxed position bound used in Figure 6
python train_unconstrained.py --x-max 100.0 --output-dir results_unconstrained/N10_x100.0_reg1.0
```

### Evaluation

```bash
python evaluate_metrics.py            # Table 3: MSE / RMSE / MAE / R² / correlation
python evaluate_table_aligator.py     # Table 5: horizon comparison in closed loop
python check_expert_success.py        # sanity check on the expert demonstrations
```

### Figures

```bash
python plot.py work_cartpole_filtered                                     # Figure 4
python plot_comparison.py                                                 # Figure 5
python analyze_constrained_unconstrained.py                               # Figure 6
python plot_optnet_vs_mlp_many_trajectories.py --limit 1.0 --num-episodes 60   # Figure 7, Table 4
python plot_qp_constraints_standalone.py                                  # Figures 8 and 9
python plot_all_trajectories_constraints.py                               # Figure 1
python plot_all_control_forces_constraints.py                             # Figure 1
```

Figures 2 and 3 (model diagrams) were drawn outside of this code base.

## The model in one paragraph

The normalised state is pushed through a feature network `4 → 128 → 128 → 64` that produces
the state-dependent linear cost `p(x)` of a quadratic program. The same state is mapped back to
physical units so that the equality constraints (a simplified linear prediction model
`x_{k+1} = A_d x_k + B_d u_k`) and the inequality constraints (force bounds and cart-position
bounds) can be built in physical units. The QP is solved in the forward pass with a
differentiable QP solver, and only the first control input `u_0` of the horizon is used,
following the receding-horizon idea. `G` and `h` are fixed buffers, not learned parameters: the
constraint satisfaction comes from the QP formulation, and the network only shapes the
objective.

## Notes and known issues

- **Two different `N = 10` models exist.** `work_cartpole_filtered/` is the main model used for
  Figure 4, Figure 7, Table 3 and Table 4 (final validation loss 0.1957, normalised).
  `results/N10_x1.0_reg1.0/` comes from a later training run performed as part of the horizon
  sweep and is the `N = 10` row of Table 5 / the constrained curve in Figure 6 (last-5-epoch mean
  validation loss 0.2802). They are separate runs, not two evaluations of one checkpoint.
- **QP regularisation.** The thesis describes the regularisation term as `ε = 10^-1`, while the
  saved runs use `--reg 1.0` (`results/*/training_params.txt`). The checkpoint in this repository
  was produced with `--reg 1.0`, i.e. the default of `train_cartpole.py`.
- **Figure 1 composition.** `plot_all_trajectories_constraints.py` and
  `plot_all_control_forces_constraints.py` each produce a two-panel figure (trajectories plus a
  histogram). The two-panel position/force layout printed in the thesis was composed from these
  outputs.
- `evaluate_table_aligator.py` performs the closed-loop rollout with the true aligator dynamics,
  so it is the only script that requires the optional dependencies.
- The QP layer is a convex layer with a linear prediction model; it is not a nonlinear MPC
  controller. Constraint satisfaction and closed-loop swing-up success are therefore different
  objectives — see Chapters 4 and 5 of the thesis.

## Extensions

`extensions/` contains follow-up work on the same cart-pole setup: conditional (task-parameterised)
models and a robustness evaluation against the aligator dynamics. See
[`extensions/README.md`](extensions/README.md) for how to run it. It is self-contained and is not
required for the thesis results above.

## Citation

```bibtex
@InProceedings{amos2017optnet,
  title     = {{O}pt{N}et: Differentiable Optimization as a Layer in Neural Networks},
  author    = {Brandon Amos and J. Zico Kolter},
  booktitle = {Proceedings of the 34th International Conference on Machine Learning},
  pages     = {136--145},
  year      = {2017},
  volume    = {70},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
}
```

If you use the trajectory-optimisation part of this repository, please also cite aligator and
ProxDDP (see `THIRD_PARTY_NOTICES.md`).

## License

Released under the Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
Third-party components and their licenses are listed in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
