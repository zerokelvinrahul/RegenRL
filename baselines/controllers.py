import os
import sys

# Ensure repository root is on sys.path when executed directly as a script
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from typing import Dict, Any, List, Tuple
from env.ev_braking_env import EVBrakingEnv


class FixedRatioController:
    """
    Applies a static regen split fraction (alpha) regardless of vehicle state.
    Physical hardware limits within the environment will clip the actual torque.
    """
    def __init__(self, alpha: float = 0.5):
        self.alpha = float(np.clip(alpha, 0.0, 1.0))

    def predict(self, obs: np.ndarray) -> np.ndarray:
        return np.array([self.alpha], dtype=np.float32)


class RuleBasedController:
    """
    Simulates standard automotive production logic.
    Dynamically reduces the requested regen fraction at low speeds (where back-EMF is low)
    and at high States of Charge (to protect the battery).
    """
    def __init__(self):
        # Observation normalization bounds from the environment
        self.v_max = 35.0
        
        # Heuristic thresholds
        self.v_taper_start = 8.0 / self.v_max   # Start tapering regen below 8 m/s
        self.soc_taper_start = 0.80             # Start tapering regen above 80% SoC
        self.base_alpha = 0.9                   # Default to heavy regen when safe

    def predict(self, obs: np.ndarray) -> np.ndarray:
        # obs = [v_norm, SoC, slope_norm, T_dem_norm]
        v_norm = obs[0]
        soc = obs[1]
        
        alpha = self.base_alpha
        
        # 1. Low-speed taper
        if v_norm < self.v_taper_start:
            alpha *= (v_norm / self.v_taper_start)
            
        # 2. High-SoC taper
        if soc > self.soc_taper_start:
            soc_penalty = max(0.0, 1.0 - ((soc - self.soc_taper_start) / (0.95 - self.soc_taper_start)))
            alpha *= soc_penalty
            
        return np.array([np.clip(alpha, 0.0, 1.0)], dtype=np.float32)


class GreedyCapController:
    """
    Physics-Aware Oracle Controller:
    Uses privileged instantaneous knowledge of vehicle hardware limits (motor power curve,
    back-EMF, and BMS charge limits) to request alpha = min(1.0, T_cap / T_dem).
    Acts as an empirical theoretical upper bound with zero torque deficit.
    """
    def __init__(self, env: EVBrakingEnv):
        self.env = env

    def predict(self, obs: np.ndarray) -> np.ndarray:
        v = float(obs[0]) * self.env.v_max
        soc = float(obs[1])
        slope = float(obs[2]) * self.env.slope_max
        t_dem = float(obs[3]) * self.env.t_dem_max

        if t_dem < 1.0:
            return np.array([1.0], dtype=np.float32)

        t_cap, _, _ = self.env.compute_hardware_limits(v, soc, slope)
        ideal_alpha = np.clip(t_cap / t_dem, 0.0, 1.0)
        return np.array([ideal_alpha], dtype=np.float32)


def evaluate_controller(
    controller_name: str, controller: Any, env: EVBrakingEnv, seed: int = 42
) -> Dict[str, float]:
    obs, _ = env.reset(seed=seed)
    done = False
    
    total_reward = 0.0
    total_energy_kj = 0.0
    total_viol_m = 0.0
    steps = 0
    
    while not done:
        action = controller.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        total_energy_kj += info["energy_recovered_kj"]
        total_viol_m += info["dist_violation_m"]
        steps += 1
        done = terminated or truncated
        
    return {
        "reward": total_reward,
        "energy_kj": total_energy_kj,
        "violation_m": total_viol_m,
        "steps": steps,
    }


def find_best_fixed_alpha(
    env: EVBrakingEnv,
    candidate_alphas: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    val_seeds: range = range(500, 520),
) -> Tuple[float, Dict[float, float]]:
    """Sweeps fixed alphas over validation drive cycles to find the optimal fixed constant."""
    mean_rewards: Dict[float, float] = {}
    best_alpha = 0.5
    best_reward = -float("inf")

    for a in candidate_alphas:
        ctrl = FixedRatioController(alpha=a)
        rewards = []
        for s in val_seeds:
            res = evaluate_controller(f"Fixed-{a}", ctrl, env, seed=s)
            rewards.append(res["reward"])
        m_r = float(np.mean(rewards))
        mean_rewards[a] = m_r
        if m_r > best_reward:
            best_reward = m_r
            best_alpha = a

    return best_alpha, mean_rewards


if __name__ == "__main__":
    test_env = EVBrakingEnv(mode="mixed", num_segments=4, initial_soc_range=(0.5, 0.95), seed=101)
    
    print("--- Baseline Evaluation (Single Cycle: Seed 101) ---")
    res_fix50 = evaluate_controller("Fixed-Ratio (50%)", FixedRatioController(alpha=0.5), test_env, seed=101)
    print(f"Fixed-Ratio (50%): Reward={res_fix50['reward']:.2f}, Energy={res_fix50['energy_kj']:.2f} kJ, Viol={res_fix50['violation_m']:.3f} m")

    res_rule = evaluate_controller("Rule-Based Heuristic", RuleBasedController(), test_env, seed=101)
    print(f"Rule-Based:        Reward={res_rule['reward']:.2f}, Energy={res_rule['energy_kj']:.2f} kJ, Viol={res_rule['violation_m']:.3f} m")

    res_oracle = evaluate_controller("Greedy Oracle", GreedyCapController(test_env), test_env, seed=101)
    print(f"Greedy Oracle:     Reward={res_oracle['reward']:.2f}, Energy={res_oracle['energy_kj']:.2f} kJ, Viol={res_oracle['violation_m']:.3f} m")

    print("\n--- Sweeping Fixed Alphas on Validation Set (seeds 500-519) ---")
    best_a, sweep = find_best_fixed_alpha(test_env)
    for a, r in sorted(sweep.items()):
        print(f"  alpha = {a:.1f}: Mean Val Reward = {r:.2f}")
    print(f"Best Fixed Alpha: {best_a:.2f}")