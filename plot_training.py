"""
Plot SAC learning curve(s) from training_log.txt.
Supports multi-seed runs by plotting individual seeds with low alpha and
the mean curve with a shaded standard deviation band and rolling smoothing.
"""
import re
import os
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_training_log(log_path="training_log.txt"):
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")

    raw = open(log_path, "rb").read()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = raw.decode("utf-16")
    else:
        text = raw.decode("utf-8", errors="ignore")

    seed_pattern = re.compile(r"---\s*Seed\s*(\d+)")
    line_pattern = re.compile(r"Step:\s*(\d+)/\d+\s*\|\s*Episode:\s*(\d+)\s*\|\s*Reward:\s*(-?[\d.]+)")

    seeds_data = {}
    current_seed = 42

    for line in text.splitlines():
        s_match = seed_pattern.search(line)
        if s_match:
            current_seed = int(s_match.group(1))
            if current_seed not in seeds_data:
                seeds_data[current_seed] = {"steps": [], "episodes": [], "rewards": []}
            continue

        l_match = line_pattern.search(line)
        if l_match:
            if current_seed not in seeds_data:
                seeds_data[current_seed] = {"steps": [], "episodes": [], "rewards": []}
            seeds_data[current_seed]["steps"].append(int(l_match.group(1)))
            seeds_data[current_seed]["episodes"].append(int(l_match.group(2)))
            seeds_data[current_seed]["rewards"].append(float(l_match.group(3)))

    return seeds_data


def moving_average(data, window=5):
    if len(data) < window:
        return np.array(data)
    padded = np.pad(data, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def plot_learning_curves(log_path="training_log.txt", out_path="results/plots/training_curve.png"):
    seeds_data = parse_training_log(log_path)
    if not seeds_data:
        print("No training data found in log.")
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(7, 4.0))

    if len(seeds_data) == 1:
        # Single seed curve with moving average
        seed = list(seeds_data.keys())[0]
        data = seeds_data[seed]
        steps = np.array(data["steps"])
        rewards = np.array(data["rewards"])
        smoothed = moving_average(rewards, window=5)

        plt.plot(steps, rewards, color="dodgerblue", alpha=0.35, label="Raw Episode Reward")
        plt.plot(steps, smoothed, color="blue", linewidth=2.0, label="Moving Average (window=5)")
        plt.axvline(1000, color="gray", linestyle="--", label="Warm-up End (1k steps)")
    else:
        # Multi-seed curve with mean and shaded std
        # Dynamically determine step range from actual data
        all_steps = [data["steps"] for data in seeds_data.values() if len(data["steps"]) > 1]
        start_step = min(st[0] for st in all_steps) if all_steps else 1000
        end_step = max(st[-1] for st in all_steps) if all_steps else 15000
        common_steps = np.linspace(max(1000, start_step), end_step, 100)
        interp_rewards = []

        for seed, data in seeds_data.items():
            st = np.array(data["steps"])
            rw = np.array(data["rewards"])
            if len(st) > 1:
                sm = moving_average(rw, window=5)
                # Plot individual seed faint trace
                plt.plot(st, sm, color="steelblue", alpha=0.25, linewidth=1.0)
                # Interpolate
                interp = np.interp(common_steps, st, sm)
                interp_rewards.append(interp)

        if interp_rewards:
            interp_matrix = np.array(interp_rewards)
            mean_curve = np.mean(interp_matrix, axis=0)
            std_curve = np.std(interp_matrix, axis=0)

            plt.plot(common_steps, mean_curve, color="#1f77b4", linewidth=2.2, label=f"Mean SAC ({len(seeds_data)} Seeds)")
            plt.fill_between(
                common_steps,
                mean_curve - std_curve,
                mean_curve + std_curve,
                color="#1f77b4",
                alpha=0.25,
                label=r"$\pm 1$ Std Dev",
            )
            plt.axvline(1000, color="gray", linestyle="--", label="Warm-up End (1k steps)")

    plt.xlabel("Environment Steps", fontweight="bold")
    plt.ylabel("Episode Reward", fontweight="bold")
    plt.title(f"SAC Training Dynamics ({len(seeds_data)} Seeds)", fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved smoothed training curve to {out_path} across {len(seeds_data)} seeds")
    os.makedirs("report", exist_ok=True)
    shutil.copy(out_path, os.path.join("report", os.path.basename(out_path)))


if __name__ == "__main__":
    plot_learning_curves()