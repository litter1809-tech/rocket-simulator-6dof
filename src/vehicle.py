from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class Rocket:
    """Single-stage rocket with constant thrust and linear propellant burn."""

    dry_mass: float
    propellant_mass: float
    burn_time: float
    average_thrust: float
    reference_area: float
    drag_coefficient: float
    moment_of_inertia: float

    time: float = field(default=0.0, init=False)
    mass: float = field(init=False)
    thrust: float = field(init=False)

    def __post_init__(self) -> None:
        for name in ("dry_mass", "burn_time", "reference_area", "moment_of_inertia"):
            value = getattr(self, name)
            if not (np.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be a positive number (got {value!r})")
        for name in ("propellant_mass", "average_thrust", "drag_coefficient"):
            value = getattr(self, name)
            if not (np.isfinite(value) and value >= 0.0):
                raise ValueError(f"{name} must be a non-negative number (got {value!r})")
        self.update(0.0)

    def update(self, t: float) -> tuple[float, float]:
        t = float(t)
        if not t >= 0.0:
            raise ValueError(f"flight time must be >= 0 (got {t!r})")
        burned = min(t / self.burn_time, 1.0)
        self.time = t
        self.mass = self.dry_mass + self.propellant_mass * (1.0 - burned)
        self.thrust = float(self.average_thrust) if t < self.burn_time else 0.0
        return self.mass, self.thrust

    def inertia_matrix(self) -> NDArray[np.float64]:
        wet_mass = self.dry_mass + self.propellant_mass
        mass_scale = self.mass / wet_mass if wet_mass > 0.0 else 1.0
        return np.eye(3, dtype=float) * self.moment_of_inertia * mass_scale
