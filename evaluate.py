import os
import glob
import math
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from env.ev_braking_env import EVBrakingEnv
from agents.sac import SACAgent
from baselines.controllers import FixedRatioController, RuleBasedController, GreedyCapController


def sync_plots_to_report(names=("bar_metrics.png", "heatmap.png", "trajectory.png",
                                "paired_differences.png", "training_curve.png")):
    os.makedirs("report", exist_ok=True)
    for n in names:
        src = os.path.join("results", "plots", n)
        if os.path.exists(src):
            shutil.copy(src, os.path.join("report", n))
    print("Synced figures from results/plots/ to report/")


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Evaluate the continued fraction used by the regularized beta function."""
    max_iterations = 200
    epsilon = 3e-14
    tiny = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    result = d

    for m in range(1, max_iterations + 1):
        m2 = 2 * m
        term = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + term * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + term / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        result *= d * c

        term = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + term * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + term / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < epsilon:
            break

    return result


def _regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    factor = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return factor * _beta_continued_fraction(a, b, x) / a
    return 1.0 - factor * _beta_continued_fraction(b, a, 1.0 - x) / b


def _student_t_two_sided_p(statistic: float, degrees_of_freedom: int) -> float:
    x = degrees_of_freedom / (degrees_of_freedom + statistic * statistic)
    return _regularized_beta(x, degrees_of_freedom / 2.0, 0.5)


def _student_t_critical(probability: float, degrees_of_freedom: int) -> float:
    low, high = 0.0, 1.0
    while 1.0 - 0.5 * _regularized_beta(
        degrees_of_freedom / (degrees_of_freedom + high * high),
        degrees_of_freedom / 2.0,
        0.5,
    ) < probability:
        high *= 2.0

    for _ in range(80):
        midpoint = (low + high) / 2.0
        cdf = 1.0 - 0.5 * _regularized_beta(
            degrees_of_freedom / (degrees_of_freedom + midpoint * midpoint),
            degrees_of_freedom / 2.0,
            0.5,
        )
        if cdf < probability:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def run_evaluation_episode(env: EVBrakingEnv, controller, is_rl: bool = False, seed: int = 101):
    obs, _ = env.reset(seed=seed)
    done = False

    history = {"v": [], "soc": [], "alpha": [], "reward": 0.0, "energy": 0.0, "violation": 0.0}

    while not done:
        if is_rl:
            action = controller.select_action(obs, evaluate=True)
            action_array = np.array([action], dtype=np.float32).reshape(-1)
        else:
            action_array = controller.predict(obs).reshape(-1)

        history["v"].append(obs[0] * env.v_max)
        history["soc"].append(obs[1])
        history["alpha"].append(float(action_array[0]))

        obs, reward, terminated, truncated, info = env.step(action_array)
        history["reward"] += reward
        history["energy"] += info["energy_recovered_kj"]
        history["violation"] += info["dist_violation_m"]

        done = terminated or truncated

    return history


def compute_paired_stats(x: np.ndarray, y: np.ndarray) -> dict:
    """Computes paired difference statistics (x - y) including t-test and 95% CI."""
    d = x - y
    n = len(d)
    mean_d = float(np.mean(d))
    std_d = float(np.std(d, ddof=1)) if n > 1 else 0.0
    se_d = std_d / math.sqrt(n) if n > 0 else 0.0
    t_crit = _student_t_critical(0.975, n - 1) if n > 1 else 1.96
    ci_low = mean_d - t_crit * se_d
    ci_high = mean_d + t_crit * se_d

    if n > 1:
        t_stat = mean_d / se_d if se_d > 0.0 else (0.0 if mean_d == 0.0 else math.copysign(math.inf, mean_d))
        p_value = _student_t_two_sided_p(t_stat, n - 1)
    else:
        t_stat, p_value = 0.0, 1.0
    win_rate = float(np.mean(d > 0.0)) * 100.0

    return {
        "mean_diff": mean_d,
        "std_diff": std_d,
        "ci_95": (ci_low, ci_high),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "win_rate_pct": win_rate,
        "wins": int(np.sum(d > 0.0)),
        "total": n,
    }


def evaluate_all(env: EVBrakingEnv, num_episodes: int = 20, seed_start: int = 200):
    os.makedirs("results/plots", exist_ok=True)

    observation_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    if observation_shape is None or action_shape is None:
        raise ValueError("Environment spaces must have defined shapes")

    # Find available SAC checkpoints
    sac_seed_files = sorted(glob.glob("saved_models/sac_seed_*.pt"))
    if not sac_seed_files and os.path.exists("saved_models/sac_regen.pt"):
        sac_seed_files = ["saved_models/sac_regen.pt"]

    print(f"Loaded SAC model checkpoints: {sac_seed_files}")

    # Build SAC agents
    sac_agents = []
    for f in sac_seed_files:
        agent = SACAgent(observation_shape[0], action_shape[0])
        agent.actor.load_state_dict(torch.load(f, weights_only=True, map_location=torch.device("cpu")))
        sac_agents.append((f, agent))

    # Baselines
    fixed_ctrl = FixedRatioController(0.5)
    rule_ctrl = RuleBasedController()
    oracle_ctrl = GreedyCapController(env)

    controllers = {
        "Fixed (50%)": (fixed_ctrl, False),
        "Rule-Based": (rule_ctrl, False),
        "Greedy Oracle": (oracle_ctrl, False),
    }

    # Evaluate baselines across test cycles
    metrics = {k: {"reward": [], "energy": [], "viol": []} for k in controllers}
    test_seeds = [seed_start + i for i in range(num_episodes)]

    for s in test_seeds:
        for name, (ctrl, is_rl) in controllers.items():
            h = run_evaluation_episode(env, ctrl, is_rl=is_rl, seed=s)
            metrics[name]["reward"].append(h["reward"])
            metrics[name]["energy"].append(h["energy"])
            metrics[name]["viol"].append(h["violation"])

    # Evaluate SAC agents (across all seeds)
    sac_per_seed = []
    for path, agent in sac_agents:
        seed_m = {"reward": [], "energy": [], "viol": []}
        for s in test_seeds:
            h = run_evaluation_episode(env, agent, is_rl=True, seed=s)
            seed_m["reward"].append(h["reward"])
            seed_m["energy"].append(h["energy"])
            seed_m["viol"].append(h["violation"])
        sac_per_seed.append(seed_m)

    # Average SAC performance across seeds
    sac_agg = {
        "reward": np.mean([s["reward"] for s in sac_per_seed], axis=0),
        "energy": np.mean([s["energy"] for s in sac_per_seed], axis=0),
        "viol": np.mean([s["viol"] for s in sac_per_seed], axis=0),
    }
    metrics["SAC (Ours)"] = sac_agg

    # Print summary table
    print("\n" + "=" * 80)
    print(f"EVALUATION RESULTS OVER {num_episodes} TEST CYCLES (Seeds {seed_start}..{seed_start+num_episodes-1})")
    print("=" * 80)
    print(f"{'Method':<16} | {'Reward':<18} | {'Energy (kJ)':<18} | {'Violation (m)':<18}")
    print("-" * 80)

    order = ["Fixed (50%)", "Rule-Based", "SAC (Ours)", "Greedy Oracle"]
    for name in order:
        m = metrics[name]
        r_m, r_s = np.mean(m["reward"]), np.std(m["reward"])
        e_m, e_s = np.mean(m["energy"]), np.std(m["energy"])
        v_m, v_s = np.mean(m["viol"]), np.std(m["viol"])
        print(f"{name:<16} | {r_m:7.2f} +- {r_s:6.2f}    | {e_m:7.2f} +- {e_s:6.2f}    | {v_m:6.3f} +- {v_s:6.3f}")
    print("-" * 80)

    # Multi-seed SAC breakdown
    if len(sac_per_seed) > 1:
        print("\nMulti-Seed SAC Variance across", len(sac_per_seed), "seeds:")
        seed_rewards = [np.mean(s["reward"]) for s in sac_per_seed]
        seed_energies = [np.mean(s["energy"]) for s in sac_per_seed]
        seed_viols = [np.mean(s["viol"]) for s in sac_per_seed]
        print(f"  Reward across seeds:     {np.mean(seed_rewards):.2f} +- {np.std(seed_rewards):.2f}")
        print(f"  Energy (kJ) across seeds: {np.mean(seed_energies):.2f} +- {np.std(seed_energies):.2f}")
        print(f"  Violation across seeds:   {np.mean(seed_viols):.3f} +- {np.std(seed_viols):.3f}")
        print("-" * 80)

    # Paired Statistics: SAC vs Baselines
    sac_r = np.array(metrics["SAC (Ours)"]["reward"])
    rule_r = np.array(metrics["Rule-Based"]["reward"])
    fixed_r = np.array(metrics["Fixed (50%)"]["reward"])
    oracle_r = np.array(metrics["Greedy Oracle"]["reward"])

    sac_v = np.array(metrics["SAC (Ours)"]["viol"])
    rule_v = np.array(metrics["Rule-Based"]["viol"])
    fixed_v = np.array(metrics["Fixed (50%)"]["viol"])

    print("\nPAIRED STATISTICAL COMPARISONS (Per-Cycle Difference):")
    print("-" * 80)

    p_rule = compute_paired_stats(sac_r, rule_r)
    print(f"SAC vs Rule-Based (Reward):")
    print(f"  Win Rate: {p_rule['wins']}/{p_rule['total']} ({p_rule['win_rate_pct']:.1f}%)")
    print(f"  Mean Paired Diff: +{p_rule['mean_diff']:.2f} (95% CI: [{p_rule['ci_95'][0]:.2f}, {p_rule['ci_95'][1]:.2f}])")
    print(f"  Paired t-test: t = {p_rule['t_stat']:.3f}, p-value = {p_rule['p_value']:.4f}")

    p_v_rule = compute_paired_stats(rule_v, sac_v)
    print(f"  Violation Reduction: {p_v_rule['mean_diff']:.3f} m (SAC lower in {p_v_rule['wins']}/{p_v_rule['total']} cycles, p = {p_v_rule['p_value']:.4f})")

    p_fixed = compute_paired_stats(sac_r, fixed_r)
    print(f"\nSAC vs Fixed 50% (Reward):")
    print(f"  Win Rate: {p_fixed['wins']}/{p_fixed['total']} ({p_fixed['win_rate_pct']:.1f}%)")
    print(f"  Mean Paired Diff: +{p_fixed['mean_diff']:.2f} (95% CI: [{p_fixed['ci_95'][0]:.2f}, {p_fixed['ci_95'][1]:.2f}])")
    print(f"  Paired t-test: t = {p_fixed['t_stat']:.3f}, p-value = {p_fixed['p_value']:.4f}")

    oracle_gap = float(np.mean(oracle_r) - np.mean(rule_r))
    sac_closed = float(np.mean(sac_r) - np.mean(rule_r))
    pct_gap_closed = (sac_closed / max(oracle_gap, 1e-3)) * 100.0
    print(f"\nOracle Gap Analysis:")
    print(f"  Rule-Based Mean: {np.mean(rule_r):.2f} | SAC Mean: {np.mean(sac_r):.2f} | Oracle Upper Bound: {np.mean(oracle_r):.2f}")
    print(f"  SAC closes {pct_gap_closed:.1f}% of the optimality gap toward the Oracle without privileged models.")
    print("=" * 80)

    # Plot 1: Trajectory Comparison on Seed 101
    primary_sac = sac_agents[0][1]
    plot_trajectory_comparison(env, primary_sac, fixed_ctrl, rule_ctrl, oracle_ctrl)

    # Plot 2: Bar Metrics
    plot_bar_metrics(metrics)

    # Plot 3: Policy Heatmap
    plot_policy_heatmap(env, primary_sac)

    # Plot 4: Paired Differences Waterfall
    plot_paired_differences(sac_r - rule_r, test_seeds)

    sync_plots_to_report()

    return metrics, p_rule, p_fixed


def plot_trajectory_comparison(env, sac, fixed, rule, oracle):
    h_sac = run_evaluation_episode(env, sac, is_rl=True, seed=101)
    h_fixed = run_evaluation_episode(env, fixed, is_rl=False, seed=101)
    h_rule = run_evaluation_episode(env, rule, is_rl=False, seed=101)
    h_oracle = run_evaluation_episode(env, oracle, is_rl=False, seed=101)

    t = np.arange(len(h_sac["v"])) * env.dt

    fig, axs = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)

    axs[0].plot(t, h_sac["v"], label="Speed", color="black", linewidth=1.5)
    axs[0].set_ylabel("Speed (m/s)")
    axs[0].legend(loc="upper right")
    axs[0].grid(True, linestyle="--", alpha=0.6)

    axs[1].plot(t, h_sac["alpha"], label="SAC (Ours)", color="blue", linewidth=1.8)
    axs[1].plot(t, h_fixed["alpha"], label="Fixed (50%)", color="gray", linestyle="--")
    axs[1].plot(t, h_rule["alpha"], label="Rule-Based", color="orange", linestyle="-.")
    axs[1].plot(t, h_oracle["alpha"], label="Oracle Upper Bound", color="green", linestyle=":", linewidth=1.5)
    axs[1].set_ylabel(r"Regen Split ($\alpha$)")
    axs[1].legend(loc="lower right")
    axs[1].grid(True, linestyle="--", alpha=0.6)

    axs[2].plot(t, h_sac["soc"], label="SAC (Ours)", color="blue", linewidth=1.8)
    axs[2].plot(t, h_rule["soc"], label="Rule-Based", color="orange", linestyle="-.")
    axs[2].plot(t, h_oracle["soc"], label="Oracle Upper Bound", color="green", linestyle=":", linewidth=1.5)
    axs[2].set_ylabel("Battery SoC")
    axs[2].set_xlabel("Time (s)")
    axs[2].legend(loc="lower right")
    axs[2].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig("results/plots/trajectory.png", dpi=300)
    plt.close()
    print("Saved trajectory comparison to results/plots/trajectory.png")


def plot_bar_metrics(metrics):
    order = ["Fixed (50%)", "Rule-Based", "SAC (Ours)", "Greedy Oracle"]
    energy_means = [np.mean(metrics[k]["energy"]) for k in order]
    viol_means = [np.mean(metrics[k]["viol"]) for k in order]

    x = np.arange(len(order))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    ax2 = ax1.twinx()

    b1 = ax1.bar(x - width / 2, energy_means, width, label="Energy Recovered (kJ)", color="#2ca02c", alpha=0.85)
    b2 = ax2.bar(x + width / 2, viol_means, width, label="Stopping Violation (m)", color="#d62728", alpha=0.85)

    ax1.set_ylabel("Mean Energy Recovered (kJ)", color="#2ca02c", fontweight="bold")
    ax2.set_ylabel("Mean Stopping Violation (m)", color="#d62728", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(order, fontweight="bold")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")

    plt.title("Performance over 20 Unseen Test Drive Cycles", fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/plots/bar_metrics.png", dpi=300)
    plt.close()
    print("Saved bar metrics to results/plots/bar_metrics.png")


def plot_policy_heatmap(env, sac):
    v_range = np.linspace(0, 1, 50)
    soc_range = np.linspace(0.1, 1.0, 50)

    V, SOC = np.meshgrid(v_range, soc_range)
    Alpha = np.zeros_like(V)

    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            obs = np.array([V[i, j], SOC[i, j], 0.0, 0.5], dtype=np.float32)
            action = sac.select_action(obs, evaluate=True)
            Alpha[i, j] = float(np.asarray(action).reshape(-1)[0])

    plt.figure(figsize=(7, 4.8))
    cp = plt.contourf(V * env.v_max, SOC, Alpha, levels=20, cmap="viridis")
    plt.colorbar(cp, label=r"Learned Regen Split ($\alpha$)")
    plt.xlabel("Vehicle Speed (m/s)")
    plt.ylabel("Battery State of Charge (SoC)")
    plt.title("SAC Policy Heatmap (Flat Road, 50% Brake Demand)", fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/plots/heatmap.png", dpi=300)
    plt.close()
    print("Saved policy heatmap to results/plots/heatmap.png")


def plot_paired_differences(diffs: np.ndarray, seeds: list):
    plt.figure(figsize=(8, 3.8))
    colors = ["#2ca02c" if d >= 0 else "#d62728" for d in diffs]
    bars = plt.bar(range(1, len(diffs) + 1), diffs, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    plt.axhline(0, color="black", linestyle="-", linewidth=0.8)
    mean_diff = float(np.mean(diffs))
    plt.axhline(mean_diff, color="blue", linestyle="--", linewidth=1.5, label=f"Mean Improvement: +{mean_diff:.1f}")
    plt.xlabel("Test Cycle Index (Seeds 200..219)")
    plt.ylabel(r"Paired Reward Diff ($\mathrm{SAC} - \mathrm{Rule}$)")
    plt.title("Cycle-by-Cycle Paired Advantage of SAC over Rule-Based", fontweight="bold")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("results/plots/paired_differences.png", dpi=300)
    plt.close()
    print("Saved paired differences to results/plots/paired_differences.png")


if __name__ == "__main__":
    test_env = EVBrakingEnv(mode="mixed", num_segments=4)
    evaluate_all(test_env, num_episodes=20, seed_start=200)