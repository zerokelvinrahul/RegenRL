# RegenRL: Optimizing Electric Vehicle Braking Torque Split via Soft Actor-Critic

**Author:** Rahul Choudhary (B23EE1060), IIT Jodhpur
**Course:** AI Mini Project, Assignment 2: Real-Life Use Cases of Reinforcement Learning

RegenRL trains a Soft Actor-Critic (SAC) agent to decide, at 10 Hz, how much of a driver's braking
demand an electric vehicle should satisfy with **regenerative braking** (energy recovered into the
battery) versus **friction brakes**, while meeting driver deceleration and respecting physical limits.
The agent is compared against a fixed-ratio controller, a rule-based automotive heuristic, and a
physics-aware greedy reference (empirical upper bound).

The SAC agent, replay buffer, environment and baselines are implemented from scratch in this repository
(SAC follows Haarnoja et al., 2018). The 2-page report is in `report/`.

## Problem Formulation

| Component | Definition |
|---|---|
| **State** (4-D, normalized) | `[v/35, SoC, slope/10°, T_dem/2500]` |
| **Action** (1-D, continuous) | `alpha in [0, 1]`: fraction of demanded torque requested from the motor. Friction gets `(1 - alpha)` |
| **Reward** (dense) | `r = 1.5 * E_kJ - 14 * (d_viol + 0.05 * T_deficit/1000) - 0.15 * (alpha_t - alpha_{t-1})^2` |
| **Episode** | 4 braking segments (urban / highway mix), randomized speed, slope, torque demand (600-2400 Nm) and initial SoC (0.50-0.95) |
| **Algorithm** | SAC with twin critics, tanh-squashed Gaussian actor rescaled to [0, 1], automatic entropy tuning |

- `E_kJ`: net electrical energy stored in the battery this step (after motor efficiency and I²R losses).
- `d_viol`: stopping-distance violation caused by deceleration shortfall when requested regen exceeds hardware limits.
- Hardware limits modeled: low-speed back-EMF cutoff (zero below 1.2 m/s, full above 4 m/s), motor constant-torque / constant-power curve (1850 Nm, 60 kW), slope stability reduction, and BMS charge-power taper above 82% SoC.

## Repository Structure

```
RegenRL/
├── agents/
│   └── sac.py              # SAC agent, replay buffer, actor/critic networks with [0, 1] action bounds
├── baselines/
│   └── controllers.py      # Fixed-ratio, rule-based heuristic, and greedy reference controller
├── env/
│   ├── __init__.py
│   ├── drive_cycles.py     # Synthetic braking-segment generator (urban & highway mix)
│   └── ev_braking_env.py   # Gymnasium environment (vehicle, motor, battery model)
├── report/
│   ├── main.tex            # 2-page two-column LaTeX report (Overleaf-ready)
│   ├── RegenRL.pdf         # Compiled report
│   └── *.png               # Copies of the figures used by main.tex (auto-synced, see below)
├── results/plots/          # Primary output of evaluate.py / plot_training.py
├── saved_models/
│   ├── sac_regen.pt        # Primary actor weights (seed 42)
│   └── sac_seed_{42,43,44}.pt
├── train.py                # Train SAC across deterministic seeds
├── plot_training.py        # Multi-seed learning curve
├── evaluate.py             # Evaluates all controllers with paired t-tests & win rates
└── requirements.txt
```

### Why are the plots in two folders?

- `results/plots/` is where the scripts **write** the figures. It is the source of truth.
- `report/*.png` are **copies** of the figures that `main.tex` includes. Overleaf only sees the files you
  upload, so the `report/` folder must be self-contained (tex + PNGs) to compile on its own.
  `evaluate.py` and `plot_training.py` copy the figures into `report/` automatically, so the two folders
  never drift apart. `paired_differences.png` is only in `results/plots/`, not in the 2-page report.

## Installation

Requires Python 3.9+.

```bash
git clone https://github.com/zerokelvinrahul/RegenRL
cd RegenRL

python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

## Usage (reproduction order)

Run all commands from the repository root:

```bash
# 1. Sanity-check the environment (random policy)
python -m env.ev_braking_env

# 2. Sanity-check the SAC agent networks, actor squashing & replay buffer
python -m agents.sac

# 3. Train SAC on the 3 reported seeds (writes saved_models/*.pt and training_log.txt)
python train.py --seeds 42 43 44

# 4. Baseline sanity check and fixed-alpha validation sweep (best alpha = 0.5)
python -m baselines.controllers

# 5. Plot the multi-seed training curve (reads training_log.txt)
python plot_training.py

# 6. Evaluate all controllers on 20 test cycles, run paired t-tests, regenerate all figures
#    (figures are also copied to report/)
python evaluate.py
```

Notes:
- `evaluate.py` averages **every** `saved_models/sac_seed_*.pt` file, so keep only seeds 42, 43, 44 there.
- Optional reward-weight ablation: `python train.py --ablation` (w_viol in {5, 14, 30}).
- Exact numbers can vary slightly with hardware and PyTorch version when retraining. The values below are from the run submitted here.

## Baselines

- **Fixed-Ratio (50%):** best static split from a validation sweep over alpha in [0.1, 1.0] on seeds 500-519 (mean validation reward peaks at alpha = 0.5).
- **Rule-Based Heuristic:** alpha = 0.9, linearly tapered at low speed (< 8 m/s) and high SoC (> 80%), mimicking standard automotive production logic.
- **Greedy reference:** privileged controller setting alpha_t = min(1, T_cap / T_dem) from exact instantaneous hardware limits. It has zero torque deficit and acts as an empirical upper bound (it ignores smoothness and efficiency-map effects).

## Evaluation Results

Evaluated across **20 unseen test drive cycles** (seeds 200..219). SAC numbers are averaged over the 3 trained seeds per cycle.

| Controller | Episode Reward | Energy Recovered (kJ) | Stopping Violation (m) |
|---|:---:|:---:|:---:|
| **Fixed (50%)** | 325.46 ± 267.18 | 289.17 ± 114.01 | 6.920 ± 7.450 |
| **Rule-Based Heuristic** | 392.43 ± 215.39 | 340.15 ± 211.58 | 7.596 ± 8.728 |
| **SAC (Ours, 3 seeds)** | **479.21 ± 292.53** | **358.20 ± 167.44** | **3.681 ± 3.021** |
| **Greedy reference (upper bound)** | 575.84 ± 269.72 | 383.99 ± 179.83 | 0.000 ± 0.000 |

### Paired hypothesis tests

All controllers use the same 20 test cycles, and tests are paired per cycle.

- **SAC vs. Rule-Based (reward):** t = 3.472, p = 0.0026, win rate 80% (16/20), mean paired diff +86.78 (95% CI 34.47 to 139.09), i.e. +22.1%.
  SAC also recovers about 5% more energy (358.2 vs 340.2 kJ).
  Mean stopping violation drops 52% (7.60 m to 3.68 m, SAC lower in 11/20 cycles), but this difference alone is **not** statistically significant (p = 0.129).
- **SAC vs. Fixed (50%):** t = 10.429, p < 1e-4, win rate 100% (20/20), mean paired diff +153.75 (95% CI 122.89 to 184.61), i.e. +47.2%.
- **Gap closed:** SAC closes 47.3% of the reward gap between the rule-based controller and the greedy reference.
- **Seed stability:** across seeds 42-44, mean test reward is 479.21 ± 5.40, energy 358.20 ± 3.01 kJ, violation 3.681 ± 0.566 m.

## Generated Visualizations

- [`trajectory.png`](results/plots/trajectory.png): speed, alpha and SoC over seed 101 (seed-42 agent; seed 101 is not one of the 20 test cycles).
- [`bar_metrics.png`](results/plots/bar_metrics.png): mean energy vs. stopping violation for all controllers.
- [`heatmap.png`](results/plots/heatmap.png): learned alpha(v, SoC) of the seed-42 agent, flat road, 50% brake demand.
- [`paired_differences.png`](results/plots/paired_differences.png): per-cycle SAC minus Rule-Based reward.
- [`training_curve.png`](results/plots/training_curve.png): multi-seed training curve with ±1σ band.

## Hyperparameters

| Parameter | Value |
|---|---|
| Learning rate (actor, critic, alpha) | 3e-4 |
| Discount γ | 0.99 |
| Target smoothing τ | 0.005 |
| Batch size | 256 |
| Replay buffer capacity | 100,000 |
| Network architecture | 2 × 256 hidden units, ReLU |
| Target entropy | −1 (−dim(A)) |
| Training steps / warm-up | 15,000 / 1,000 per seed |

## Dependencies

`numpy`, `gymnasium`, `torch`, `matplotlib` (see `requirements.txt`).