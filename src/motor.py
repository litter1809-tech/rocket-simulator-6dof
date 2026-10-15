"""
RASP/RockSim motor thrust-curve parsing and interpolation.
Author: Tegan Broody
School: Ridley High School
Date: October 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ThrustCurve:
    """RASP/RockSim thrust curve sampled in seconds and newtons."""

    times_s: NDArray[np.float64]
    thrust_n: NDArray[np.float64]
    propellant_mass_kg: float | None = None
    designation: str = "custom motor"
    _cumulative_impulse_ns: NDArray[np.float64] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        times = np.asarray(self.times_s, dtype=float)
        thrust = np.asarray(self.thrust_n, dtype=float)
        if times.ndim != 1 or thrust.ndim != 1 or times.shape != thrust.shape:
            raise ValueError("Thrust curve time and thrust must be matching 1D arrays.")
        if times.size < 2 or not np.all(np.isfinite(times)) or not np.all(np.isfinite(thrust)):
            raise ValueError("Thrust curve must contain at least two finite data points.")
        if times[0] < 0.0 or np.any(np.diff(times) <= 0.0):
            raise ValueError("Thrust curve times must be nonnegative and strictly increasing.")
        if np.any(thrust < 0.0) or not np.any(thrust > 0.0):
            raise ValueError("Thrust curve must contain nonnegative thrust and a positive sample.")
        if self.propellant_mass_kg is not None and (
            not np.isfinite(self.propellant_mass_kg) or self.propellant_mass_kg <= 0.0
        ):
            raise ValueError("Motor propellant mass must be a positive finite value.")
        object.__setattr__(self, "times_s", times)
        object.__setattr__(self, "thrust_n", thrust)
        interval_impulses = (
            0.5
            * (thrust[:-1] + thrust[1:])
            * np.diff(times)
        )
        object.__setattr__(
            self,
            "_cumulative_impulse_ns",
            np.concatenate(([0.0], np.cumsum(interval_impulses))),
        )

    @property
    def burn_time_s(self) -> float:
        return float(self.times_s[-1])

    @property
    def total_impulse_ns(self) -> float:
        return float(self._cumulative_impulse_ns[-1])

    def thrust_at(self, time_s: float) -> float:
        if time_s < self.times_s[0] or time_s > self.burn_time_s:
            return 0.0
        return float(np.interp(time_s, self.times_s, self.thrust_n))

    def impulse_until(self, time_s: float) -> float:
        """Integrate the piecewise-linear curve up to time_s."""
        if time_s <= self.times_s[0]:
            return 0.0
        if time_s >= self.burn_time_s:
            return self.total_impulse_ns

        segment = int(np.searchsorted(self.times_s, time_s, side="right") - 1)
        dt = time_s - self.times_s[segment]
        interval = self.times_s[segment + 1] - self.times_s[segment]
        slope = (self.thrust_n[segment + 1] - self.thrust_n[segment]) / interval
        prior_impulse = float(self._cumulative_impulse_ns[segment])
        return prior_impulse + float(self.thrust_n[segment] * dt + 0.5 * slope * dt**2)

    def propellant_fraction_remaining(self, time_s: float) -> float:
        if time_s <= self.times_s[0]:
            return 1.0
        if time_s >= self.burn_time_s:
            return 0.0
        total_impulse = self.total_impulse_ns
        if total_impulse <= 0.0:
            raise ValueError("Thrust curve must have positive total impulse.")
        return float(np.clip(1.0 - self.impulse_until(time_s) / total_impulse, 0.0, 1.0))


def parse_eng_motor(text: str) -> ThrustCurve:
    """Parse a standard RASP/RockSim .eng text file."""
    lines = [
        line.split(";", 1)[0].strip()
        for line in text.lstrip("\ufeff").splitlines()
    ]
    records = [line for line in lines if line]
    if not records:
        raise ValueError("Motor file is empty or contains only comments.")

    header = records[0].split()
    if len(header) < 6:
        raise ValueError(
            "RASP motor header must include designation, dimensions, delay, "
            "propellant mass, and total mass."
        )
    try:
        propellant_mass_kg = float(header[4]) / 1000.0
        float(header[1])
        float(header[2])
        float(header[5])
    except ValueError as error:
        raise ValueError("RASP motor header contains invalid numeric fields.") from error
    if propellant_mass_kg <= 0.0:
        raise ValueError("RASP motor propellant mass must be positive.")

    times_s: list[float] = []
    thrust_n: list[float] = []
    for line_number, record in enumerate(records[1:], start=2):
        fields = record.split()
        if len(fields) != 2:
            raise ValueError(
                f"Invalid thrust data on line {line_number}: expected time and thrust."
            )
        try:
            time_s, thrust_value_n = (float(field) for field in fields)
        except ValueError as error:
            raise ValueError(
                f"Invalid numeric thrust data on line {line_number}."
            ) from error
        if not np.isfinite(time_s) or not np.isfinite(thrust_value_n):
            raise ValueError(f"Non-finite thrust data on line {line_number}.")
        times_s.append(time_s)
        thrust_n.append(thrust_value_n)

    return ThrustCurve(
        times_s=np.asarray(times_s, dtype=float),
        thrust_n=np.asarray(thrust_n, dtype=float),
        propellant_mass_kg=propellant_mass_kg,
        designation=header[0],
    )
