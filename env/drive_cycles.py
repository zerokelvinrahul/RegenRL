import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class BrakingSegment:
    """Represents a single braking maneuver within a drive cycle."""
    initial_speed: float        # m/s (0 to 35 m/s)
    target_speed: float         # m/s
    road_slope_deg: float       # degrees (-10 to +10 deg)
    peak_torque_demand: float   # Nm (300 to 2500 Nm)
    num_steps: int              # duration in 0.1s steps


def generate_drive_cycle(
    rng: np.random.Generator,
    mode: str = "mixed",
    num_segments: int = 6,
) -> List[BrakingSegment]:
    """
    Generates a synthetic drive cycle composed of realistic braking maneuvers
    modeled after urban stop-and-go (UDDS-like) and highway (WLTP/HWFET-like)
    deceleration kinematics.
    """
    segments: List[BrakingSegment] = []

    for i in range(num_segments):
        if mode == "urban" or (mode == "mixed" and i % 2 == 0):
            # Urban stop-and-go braking (low-to-medium speed, frequent full stops)
            v_init = float(rng.uniform(8.0, 20.0))
            v_targ = 0.0 if rng.random() < 0.65 else float(rng.uniform(2.0, v_init - 4.0))
            slope = float(rng.uniform(-6.0, 6.0))
            peak_torque = float(rng.uniform(600.0, 2100.0))
            steps = int(rng.integers(20, 38))
        else:
            # Highway / suburban deceleration (high speed, moderate-to-heavy braking)
            v_init = float(rng.uniform(20.0, 33.0))
            v_targ = float(rng.uniform(5.0, v_init - 6.0))
            slope = float(rng.uniform(-8.0, 8.0))
            peak_torque = float(rng.uniform(800.0, 2400.0))
            steps = int(rng.integers(25, 45))

        segments.append(
            BrakingSegment(
                initial_speed=v_init,
                target_speed=v_targ,
                road_slope_deg=slope,
                peak_torque_demand=peak_torque,
                num_steps=steps,
            )
        )

    return segments


def build_torque_demand_profile(
    peak_torque: float, num_steps: int, step_idx: int, current_speed: float, target_speed: float
) -> float:
    """
    Shapes a smooth driver brake-pedal torque profile (ramp-up, hold, taper)
    that reduces demand once the vehicle reaches the target speed.
    """
    if current_speed <= target_speed + 0.2:
        return 0.0

    progress = step_idx / max(1, num_steps - 1)
    if progress < 0.2:
        # Smooth pedal application ramp
        envelope = 0.4 + 0.6 * (progress / 0.2)
    elif progress > 0.85:
        # Smooth pedal release taper
        envelope = max(0.2, 1.0 - 0.8 * ((progress - 0.85) / 0.15))
    else:
        envelope = 1.0

    # Modulate slightly as speed approaches target speed
    speed_gap_factor = np.clip((current_speed - target_speed) / 4.0, 0.25, 1.0)
    return float(np.clip(peak_torque * envelope * speed_gap_factor, 0.0, 2500.0))