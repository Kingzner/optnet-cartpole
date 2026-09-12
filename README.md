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
├── cartpole/                               # shared code used by every script
│   ├── models.py                           # OptNetCartPoleMPC: feature network + differentiable QP layer
│   └── utils.py                            # cart-pole model builder (Pinocchio)
│
├── scripts/
│   ├── data/
│   │   ├── generate_dataset.py             # expert trajectories via constrained trajectory optimisation (aligator/ProxDDP)
│   │   └── filter_data.py                  # drop extreme trajectories (threshold 20) -> 994 kept
│   ├── train/
│   │   ├── train_cartpole.py               # train the OptNet-QP controller
│   │   ├── train_baseline.py               # train the MLP baseline
│   │   └── train_unconstrained.py          # same QP structure, cart-position bound relaxed (x_max = 100 m)
│   ├── evaluate/
│   │   ├── evaluate_metrics.py             # prediction metrics in physical units (Table 3)
│   │   ├── evaluate_table_aligator.py      # closed-loop horizon comparison (Table 5)
│   │   ├── check_expert_success.py         # sanity check on the expert demonstrations
│   │   └── compute_cartpole_linearization.py
│   └── figures/
│       ├── plot_loss_curves.py             # Figure 4
│       ├── plot_loss_comparison.py         # Figure 5
│       ├── plot_constrained_vs_relaxed.py  # Figure 6
│       ├── plot_rollout_comparison.py      # Figure 7 and Table 4
│       ├── plot_qp_constraints.py          # Figures 8 and 9
│       ├── plot_dataset_positions.py       # Figure 1
│       └── plot_dataset_control_forces.py  # Figure 1
│
├── data/                                   # dataset
│   ├── features.pt / labels.pt             # 994 trajectories, used for training
│   └── features_raw.pt / labels_raw.pt     # 1000 generated trajectories, before filtering
│
├── outputs/                                # training results (checkpoints, loss curves, logs)
│   ├── main/                               # main OptNet-QP model, N = 10
│   ├── baseline/                           # MLP baseline
│   ├── horizon/N{1,2,5,8,10,12,15,18}_x1.0_reg1.0    # prediction-horizon sweep (Table 5)
│   └── relaxed/N10_x100.0_reg1.0           # relaxed cart-position bound (Figure 6)
│
├── figures/                                # the figures as included in the thesis
├── extensions/                             # follow-up work: conditional / robustness experiments
├── requirements.txt
└── requirements-aligator.txt
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

Every script can be run directly from the repository root — each one adds the repository root
to `sys.path` itself, so `python scripts/train/train_cartpole.py` works without installing the
code as a package. All data and output paths are relative to the repository root.

## Reproducing the results

| Thesis item | Command |
|---|---|
| Figure 1 (dataset) | `python scripts/figures/plot_dataset_positions.py` and `...control_forces.py` |
| Table 3 (prediction accuracy) | `python scripts/evaluate/evaluate_metrics.py` |
| Figure 4 (loss curves) | `python scripts/figures/plot_loss_curves.py outputs/main` |
| Figure 5 (OptNet vs. MLP) | `python scripts/figures/plot_loss_comparison.py` |
| Figure 6 (constrained vs. relaxed) | `python scripts/figures/plot_constrained_vs_relaxed.py` |
| Figure 7 + Table 4 (rollout) | `python scripts/figures/plot_rollout_comparison.py --limit 1.0 --num-episodes 60` |
| Figures 8 and 9 (G and h) | `python scripts/figures/plot_qp_constraints.py` |
| Table 5 (horizon sweep) | `python scripts/evaluate/evaluate_table_aligator.py` |

### Data (already included — this step is optional)

```bash
python scripts/data/generate_dataset.py  # 1000 trajectories -> data/features_raw.pt, labels_raw.pt
python scripts/data/filter_data.py       # remove extreme trajectories -> data/features.pt, labels.pt (994)
```

`generate_dataset.py` solves a constrained swing-up problem with ProxDDP for each randomly
sampled initial state, reorders the state from the simulator convention
`[x, θ, ẋ, θ̇]` to the learning convention `[x, ẋ, θ, θ̇]`, adds sensor/actuator noise and saves
the trajectories. `filter_data.py` then drops trajectories whose maximum absolute state value
exceeds 20.

| File | Shape | Meaning |
|---|---|---|
| `data/features_raw.pt` / `labels_raw.pt` | (1000, 500, 4) / (1000, 500, 1) | raw generated trajectories |
| `data/features.pt` / `labels.pt` | (994, 500, 4) / (994, 500, 1) | dataset used for training |

### Training

```bash
python scripts/train/train_cartpole.py        # main model -> outputs/main/
python scripts/train/train_baseline.py        # MLP baseline -> outputs/baseline/

# prediction-horizon sweep (Table 5)
for N in 1 2 5 8 10 12 15 18; do
  python scripts/train/train_cartpole.py --N $N --output-dir outputs/horizon/N${N}_x1.0_reg1.0
done

# relaxed position bound used in Figure 6
python scripts/train/train_unconstrained.py --x-max 100.0 --output-dir outputs/relaxed/N10_x100.0_reg1.0
```

### Evaluation

```bash
python scripts/evaluate/evaluate_metrics.py         # Table 3: MSE / RMSE / MAE / R² / correlation
python scripts/evaluate/evaluate_table_aligator.py  # Table 5: horizon comparison in closed loop
python scripts/evaluate/check_expert_success.py     # sanity check on the expert demonstrations
```

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

- **Two different `N = 10` models exist.** `outputs/main/` is the main model used for
  Figure 4, Figure 7, Table 3 and Table 4 (final validation loss 0.1957, normalised).
  `outputs/horizon/N10_x1.0_reg1.0/` comes from a later training run performed as part of the
  horizon sweep and is the `N = 10` row of Table 5 / the constrained curve in Figure 6
  (last-5-epoch mean validation loss 0.2802). They are separate runs, not two evaluations of
  one checkpoint.
- **QP regularisation.** The thesis describes the regularisation term as `ε = 10^-1`, while the
  saved runs use `--reg 1.0` (`outputs/horizon/*/training_params.txt`). The checkpoints in this
  repository were produced with `--reg 1.0`, i.e. the default of `train_cartpole.py`.
- **Figure 1 composition.** `plot_dataset_positions.py` and `plot_dataset_control_forces.py`
  each produce a two-panel figure (trajectories plus a histogram). The two-panel position/force
  layout printed in the thesis was composed from these outputs.
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
