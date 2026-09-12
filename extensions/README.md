# Extensions: conditional models and robustness evaluation

Follow-up experiments on the same cart-pole setup, kept separate from the thesis pipeline. The
idea is to make the QP cost depend not only on the state but also on task parameters
(`target_x`, `target_theta`, `x_min`, `x_max`), so that one model covers a family of
swing-up / tracking tasks, and then to test how that model behaves under perturbations using
the true aligator dynamics.

## Contents

```
extensions/
├── robustness.py                  # closed-loop robustness test with aligator dynamics
├── test_conditional_robustness.py # success-rate evaluation of the conditional model
├── weights/
│   ├── cartpole_optnet_best_2B.pth   # robustness.py default model (OptNetCartPoleH3)
│   ├── norm_stats.pt                 # normalisation statistics for that model
│   ├── best_model_conditional.pth    # conditional model
│   └── norm_stats_conditional.pt     # normalisation statistics for the conditional model
└── data/
    ├── cartpole_data_multi/          # task-parameterised dataset (features, labels, task_params)
    ├── cartpole_dataset.npz
    ├── cartpole_dataset_pos_only.npz
    ├── robustness_aligator_results.npz
    └── test_results.npz
```

`robustness_aligator.png`, `conv_heatmap.png`, `conv_vs_constraint_width.png` and
`conv_vs_target_theta.png` are the figures produced by these scripts.

## Running

`robustness.py` is self-contained (it defines its own `OptNetCartPoleH3` module) and needs
Pinocchio and aligator:

```bash
pip install -r ../requirements-aligator.txt

# random initial states
python extensions/robustness.py --mode random --num 100 \
    --model_path extensions/weights/cartpole_optnet_best_2B.pth \
    --norm_stats_path extensions/weights/norm_stats.pt \
    --data_dir cartpole_data

# initial states taken from the training set
python extensions/robustness.py --mode train --num 100 \
    --model_path extensions/weights/cartpole_optnet_best_2B.pth \
    --norm_stats_path extensions/weights/norm_stats.pt \
    --data_dir cartpole_data
```

## Known issue

`test_conditional_robustness.py` imports `OptNetCartPoleH3_Conditional` from `models`, but no
version of `models.py` in this repository defines that class — the conditional variant that
ships here is `OptNetCartPoleMPC_Conditional`, which has a different architecture (it keeps the
MPC structure instead of the three-layer `H3` head). The script is therefore **not runnable
as-is**: it needs either the original `models.py` that defined the `H3` variant, or a rewrite of
the conditional model to match. It is kept here for reference together with the trained
conditional checkpoint and its data.
