"""
RegenRL Simulation Environment Package.
Provides EVBrakingEnv and synthetic drive cycle generator.
"""
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Seen only by type checkers / IDEs (never executed at runtime),
    # so __all__ below resolves without eager imports.
    from env.drive_cycles import BrakingSegment, generate_drive_cycle
    from env.ev_braking_env import EVBrakingEnv


def __getattr__(name: str) -> Any:
    if name == "EVBrakingEnv":
        from env.ev_braking_env import EVBrakingEnv
        return EVBrakingEnv
    if name == "generate_drive_cycle":
        from env.drive_cycles import generate_drive_cycle
        return generate_drive_cycle
    if name == "BrakingSegment":
        from env.drive_cycles import BrakingSegment
        return BrakingSegment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["EVBrakingEnv", "generate_drive_cycle", "BrakingSegment"]