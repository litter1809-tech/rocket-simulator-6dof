"""
Rocket vehicle mass, propulsion, geometry, and inertia model.
Author: Tegan Broody
School: Ridley High School
Date: October 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from src.motor import ThrustCurve


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
    thrust_curve: ThrustCurve | None = None
    thrust_scale: float = 1.0

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
        if not (np.isfinite(self.thrust_scale) and self.thrust_scale >= 0.0):
            raise ValueError("thrust_scale must be a finite non-negative number")
        if self.thrust_curve is not None:
            self.burn_time = self.thrust_curve.burn_time_s
        self.update(0.0)

        t = float(t)
        if not np.isfinite(t) or t < 0.0:
            raise ValueError(
                f"flight time must be finite and >= 0 (got {t!r})"
            )

        if self.thrust_curve is None:
            propellant_remaining_fraction = max(1.0 - t / self.burn_time, 0.0)
            thrust = self.average_thrust if t < self.burn_time else 0.0
        else:
            propellant_remaining_fraction = (
                self.thrust_curve.propellant_fraction_remaining(t)
            )
            thrust = self.thrust_curve.thrust_at(t)
        self.time = t
        self.mass = (
            self.dry_mass + self.propellant_mass * propellant_remaining_fraction
        )
        self.thrust = float(thrust * self.thrust_scale)
        return self.mass, self.thrust

    def inertia_matrix(self) -> NDArray[np.float64]:
        wet_mass = self.dry_mass + self.propellant_mass
        mass_scale = self.mass / wet_mass if wet_mass > 0.0 else 1.0
        return np.eye(3, dtype=float) * self.moment_of_inertia * mass_scale
