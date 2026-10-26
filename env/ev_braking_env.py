from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np

import os
import sys

# Ensure repository root is on sys.path when executed directly as a script
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.drive_cycles import BrakingSegment, build_torque_demand_profile, generate_drive_cycle


class EVBrakingEnv(gym.Env):
    """
    Custom Gymnasium Environment for Electric Vehicle Regenerative Braking Split (RegenRL).

    State Space S (4D continuous, normalized):
        s_t = [v_norm, SoC, slope_norm, T_dem_norm]
        - v_t: Vehicle velocity in [0, 35] m/s -> normalized to [0, 1]
        - SoC_t: Battery State of Charge in [0.1, 1.0]
        - theta_t: Road slope angle in [-10, +10] deg -> normalized to [-1, 1]
        - T_dem_t: Driver braking torque demand in [0, 2500] Nm -> normalized to [0, 1]

    Action Space A (1D continuous):
        a_t = alpha_t in [0, 1], representing the regenerative braking torque fraction.
        - Requested motor regen torque: T_regen_req = alpha_t * T_dem_t
        - Friction brake torque:        T_fric      = (1 - alpha_t) * T_dem_t
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        mode: str = "mixed",
        num_segments: int = 6,
        initial_soc_range: Tuple[float, float] = (0.50, 0.95),
        w_viol: float = 14.0,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.mode = mode
        self.num_segments = num_segments
        self.initial_soc_range = initial_soc_range
        self.dt = 0.1  # 10 Hz control timestep (s)

        # Physical EV Parameters (Compact EV class)
        self.mass = 1650.0              # Vehicle mass (kg)
        self.gravity = 9.81             # Gravitational acceleration (m/s^2)
        self.wheel_radius = 0.31        # Effective wheel radius (m)
        self.rho_air = 1.225            # Air density (kg/m^3)
        self.cd = 0.29                  # Aerodynamic drag coefficient
        self.frontal_area = 2.27        # Frontal area (m^2)
        self.c_rr = 0.012               # Rolling resistance coefficient

        # Electric Motor & Powertrain Limits (referred to wheels)
        self.max_regen_Wheel_torque = 1850.0  # Max constant-torque region limit at wheel (Nm)
        self.max_regen_power = 60000.0        # Max motor/inverter regen power limit (60 kW)
        self.v_cutoff_low = 1.2               # Complete regen cutoff speed (m/s)
        self.v_cutoff_full = 4.0              # Speed above which full back-EMF is available (m/s)

        # Battery Pack Parameters (Equivalent Circuit Model)
        # Effective active buffer capacity scaled so SoC dynamics are observable within a single cycle
        self.nominal_capacity_joules = 3.6e6  # 1.0 kWh active buffer window
        self.r_internal = 0.085               # Pack internal resistance (Ohms)
        self.max_charge_power = 60000.0       # Max BMS charge power below taper threshold (W)
        self.soc_taper_start = 0.82           # SoC threshold where BMS begins CCCV charge tapering

        # Normalization bounds
        self.v_max = 35.0
        self.slope_max = 10.0
        self.t_dem_max = 2500.0

        # Reward weights (3-term dense formulation)
        self.w_energy = 1.5     # Reward per kJ of net electrical energy stored in battery
        self.w_viol = float(w_viol)  # Penalty weight for stopping-distance / deceleration shortfall
        self.w_smooth = 0.15    # Penalty weight for actuator jerk |alpha_t - alpha_{t-1}|^2

        # Gymnasium Spaces
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.rng = np.random.default_rng(seed)

        # Internal episode state variables
        self.segments: list[BrakingSegment] = []
        self.seg_idx: int = 0
        self.step_in_seg: int = 0
        self.v: float = 0.0
        self.soc: float = 0.7
        self.slope_deg: float = 0.0
        self.t_dem: float = 0.0
        self.prev_alpha: float = 0.5

    def _get_obs(self) -> np.ndarray:
        return np.array(
            [
                np.clip(self.v / self.v_max, 0.0, 1.0),
                np.clip(self.soc, 0.0, 1.0),
                np.clip(self.slope_deg / self.slope_max, -1.0, 1.0),
                np.clip(self.t_dem / self.t_dem_max, 0.0, 1.0),
            ],
            dtype=np.float32,
        )

    def compute_hardware_limits(
        self, v: float, soc: float, slope_deg: float
    ) -> Tuple[float, float, float]:
        """
        Computes the instantaneous wheel regenerative torque limits imposed by:
        1. Motor constant-torque / constant-power curve & low-speed back-EMF cutoff
        2. Axle stability factor on steep slopes
        3. Battery BMS high-SoC charge power tapering
        """
        # 1. Low-speed back-EMF fade factor
        if v <= self.v_cutoff_low:
            f_speed = 0.0
        elif v < self.v_cutoff_full:
            f_speed = (v - self.v_cutoff_low) / (self.v_cutoff_full - self.v_cutoff_low)
        else:
            f_speed = 1.0

        # Constant-power hyperbola: T_max(v) = min(T_max, P_max * r_w / v)
        v_eff = max(v, 0.5)
        t_motor_curve = min(
            self.max_regen_Wheel_torque, (self.max_regen_power * self.wheel_radius) / v_eff
        )

        # Downhill/uphill normal load stability factor
        slope_rad = np.deg2rad(slope_deg)
        f_stability = np.clip(np.cos(slope_rad) - 0.12 * abs(np.sin(slope_rad)), 0.75, 1.0)
        t_motor_limit = t_motor_curve * f_speed * f_stability

        # 2. Battery BMS high-SoC charge acceptance taper
        if soc <= self.soc_taper_start:
            f_soc = 1.0
        elif soc >= 0.98:
            f_soc = 0.02
        else:
            f_soc = max(0.02, 1.0 - ((soc - self.soc_taper_start) / (0.98 - self.soc_taper_start)))

        p_batt_max = self.max_charge_power * f_soc
        # Convert max battery terminal power to wheel torque limit
        eta_est = 0.88
        t_batt_limit = (p_batt_max * self.wheel_radius) / (v_eff * eta_est)

        t_regen_cap = min(t_motor_limit, t_batt_limit)
        return float(t_regen_cap), float(t_motor_limit), float(t_batt_limit)

    def _motor_efficiency(self, v: float, t_regen_wheel: float) -> float:
        """Models combined motor + inverter efficiency map in [0.72, 0.92]."""
        if t_regen_wheel <= 1.0 or v <= self.v_cutoff_low:
            return 0.0
        v_norm = np.clip(v / 25.0, 0.1, 1.0)
        t_norm = np.clip(t_regen_wheel / self.max_regen_Wheel_torque, 0.05, 1.0)
        # Peak efficiency in mid-speed, moderate-torque region
        eff = 0.92 - 0.10 * (v_norm - 0.55) ** 2 - 0.08 * (t_norm - 0.45) ** 2
        return float(np.clip(eff, 0.72, 0.92))

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.segments = generate_drive_cycle(
            self.rng, mode=self.mode, num_segments=self.num_segments
        )
        self.seg_idx = 0
        self.step_in_seg = 0

        if options and "initial_soc" in options:
            self.soc = float(options["initial_soc"])
        else:
            self.soc = float(self.rng.uniform(*self.initial_soc_range))

        seg = self.segments[0]
        self.v = seg.initial_speed
        self.slope_deg = seg.road_slope_deg
        self.t_dem = build_torque_demand_profile(
            seg.peak_torque_demand, seg.num_steps, 0, self.v, seg.target_speed
        )
        self.prev_alpha = 0.5

        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        alpha = float(np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[0], 0.0, 1.0))

        # Requested torque split
        t_regen_req = alpha * self.t_dem
        t_fric = (1.0 - alpha) * self.t_dem

        # Hardware capability limits at current state
        t_regen_cap, t_motor_lim, t_batt_lim = self.compute_hardware_limits(
            self.v, self.soc, self.slope_deg
        )

        # Actual achieved regen torque is bounded by physical motor & BMS limits
        t_regen_act = min(t_regen_req, t_regen_cap)
        t_total_act = t_regen_act + t_fric

        # Torque deficit when requested regen exceeds hardware limits
        t_deficit = max(0.0, self.t_dem - t_total_act)

        # Longitudinal resistive forces
        slope_rad = np.deg2rad(self.slope_deg)
        f_aero = 0.5 * self.rho_air * self.cd * self.frontal_area * (self.v ** 2)
        f_roll = self.mass * self.gravity * self.c_rr * np.cos(slope_rad) if self.v > 0.05 else 0.0
        f_grade = self.mass * self.gravity * np.sin(slope_rad)

        # Desired vs. actual deceleration (m/s^2)
        f_brake_req = self.t_dem / self.wheel_radius
        f_brake_act = t_total_act / self.wheel_radius

        decel_req = (f_brake_req + f_aero + f_roll + f_grade) / self.mass
        decel_act = (f_brake_act + f_aero + f_roll + f_grade) / self.mass
        decel_shortfall = max(0.0, decel_req - decel_act)

        # Stopping-distance violation incurred during this step (meters)
        # Proportional to speed and deceleration shortfall
        dist_violation_m = self.v * (decel_shortfall / self.gravity) * self.dt

        # Battery energy recovery with motor efficiency & I^2*R internal losses
        eta_mot = self._motor_efficiency(self.v, t_regen_act)
        p_mech = (t_regen_act * self.v) / self.wheel_radius
        p_elec_terminal = p_mech * eta_mot

        v_oc = 340.0 + 50.0 * self.soc
        i_chg = p_elec_terminal / v_oc
        p_loss_joule = (i_chg ** 2) * self.r_internal
        p_net_stored = max(0.0, p_elec_terminal - p_loss_joule)

        energy_recovered_kj = (p_net_stored * self.dt) / 1000.0
        delta_soc = (p_net_stored * self.dt) / self.nominal_capacity_joules
        self.soc = float(np.clip(self.soc + delta_soc, 0.0, 1.0))

        # Dense 3-term reward calculation
        r_energy = self.w_energy * energy_recovered_kj
        r_viol = self.w_viol * (dist_violation_m + 0.05 * (t_deficit / 1000.0))
        r_smooth = self.w_smooth * ((alpha - self.prev_alpha) ** 2)
        reward = float(r_energy - r_viol - r_smooth)

        # Update velocity for next step
        v_prev = self.v
        self.v = float(np.clip(self.v - decel_act * self.dt, 0.0, self.v_max))
        self.prev_alpha = alpha

        # Advance drive cycle segment state
        self.step_in_seg += 1
        seg = self.segments[self.seg_idx]
        terminated = False

        if self.step_in_seg >= seg.num_steps or self.v <= max(0.05, seg.target_speed):
            self.seg_idx += 1
            self.step_in_seg = 0
            if self.seg_idx >= len(self.segments):
                terminated = True
            else:
                next_seg = self.segments[self.seg_idx]
                self.v = next_seg.initial_speed
                self.slope_deg = next_seg.road_slope_deg
                self.prev_alpha = 0.5
                seg = next_seg

        if not terminated:
            self.t_dem = build_torque_demand_profile(
                seg.peak_torque_demand, seg.num_steps, self.step_in_seg, self.v, seg.target_speed
            )
        else:
            self.t_dem = 0.0

        info = {
            "speed_m_s": v_prev,
            "soc": self.soc,
            "slope_deg": self.slope_deg,
            "t_dem_nm": f_brake_req * self.wheel_radius,
            "alpha": alpha,
            "t_regen_req_nm": t_regen_req,
            "t_regen_act_nm": t_regen_act,
            "t_regen_cap_nm": t_regen_cap,
            "t_fric_nm": t_fric,
            "t_deficit_nm": t_deficit,
            "energy_recovered_kj": energy_recovered_kj,
            "dist_violation_m": dist_violation_m,
            "decel_shortfall": decel_shortfall,
        }

        return self._get_obs(), reward, terminated, False, info


if __name__ == "__main__":
    # Built-in sanity test
    env = EVBrakingEnv(seed=42)
    obs, _ = env.reset(seed=42)
    print(f"Initial Observation (normalized): {obs}")
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space:      {env.action_space}")

    total_reward = 0.0
    total_energy_kj = 0.0
    total_viol_m = 0.0
    steps = 0

    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        total_energy_kj += info["energy_recovered_kj"]
        total_viol_m += info["dist_violation_m"]
        steps += 1
        done = terminated or truncated

    print(f"\nSanity Check Passed over {steps} steps!")
    print(f"  Total Episode Reward:         {total_reward:.2f}")
    print(f"  Total Recovered Energy:       {total_energy_kj:.2f} kJ")
    print(f"  Cumulative Stopping Violation:{total_viol_m:.3f} m")
    print(f"  Final Battery SoC:            {info['soc']:.4f}")