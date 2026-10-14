from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class Rocket:
    """Single-stage vehicle with constant thrust and linearly depleted propellant."""

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
        """Update total mass and thrust from elapsed motor burn time."""
        t = float(t)
        if not t >= 0.0:
            raise ValueError(f"flight time must be >= 0 (got {t!r})")
        burn_fraction = min(t / self.burn_time, 1.0)
        self.time = t
        self.mass = self.dry_mass + self.propellant_mass * (1.0 - burn_fraction)
        self.thrust = float(self.average_thrust) if t < self.burn_time else 0.0
        return self.mass, self.thrust

    def inertia_matrix(self) -> NDArray[np.float64]:
        """Return the isotropic inertia tensor scaled from wet to current mass."""
        wet_mass_kg = self.dry_mass + self.propellant_mass
        mass_ratio = self.mass / wet_mass_kg if wet_mass_kg > 0.0 else 1.0
        return np.eye(3, dtype=float) * self.moment_of_inertia * mass_ratio
