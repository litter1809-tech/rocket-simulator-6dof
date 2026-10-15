"""
Flight telemetry parsing and simulation-comparison metrics.
Author: Tegan Broody
School: Ridley High School
Date: October 2026
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class FlightTelemetry:
    """Cleaned altimeter telemetry, with time and altitude relative to launch."""

    time_s: NDArray[np.float64]
    altitude_m: NDArray[np.float64]
    velocity_m_s: NDArray[np.float64]
    downrange_m: NDArray[np.float64] | None
    crossrange_m: NDArray[np.float64] | None


@dataclass(frozen=True)
class TelemetryComparison:
    """Altitude comparison evaluated at the real telemetry sample times."""

    simulated_apogee_m: float
    real_apogee_m: float
    apogee_delta_m: float
    apogee_delta_percent: float | None
    altitude_mae_m: float
    altitude_rmse_m: float
    compared_samples: int


def _column_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.strip().lower())


def _find_column(
    columns: dict[str, str],
    candidates: tuple[str, ...],
) -> str | None:
    return next((columns[key] for key in candidates if key in columns), None)


def parse_flight_csv(csv_content: bytes | str) -> FlightTelemetry:
    """Parse a CSV with Time [s] and Altitude [m] columns.

    Optional Velocity [m/s], X/Downrange [m], and Y/Crossrange [m] columns
    are accepted. Time and altitude are rebased to the first valid sample.
    """
    try:
        text = (
            csv_content.decode("utf-8-sig")
            if isinstance(csv_content, bytes)
            else csv_content.lstrip("\ufeff")
        )
    except UnicodeDecodeError as error:
        raise ValueError("CSV must be encoded as UTF-8.") from error

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("CSV is empty or missing a header row.")

    normalized: dict[str, str] = {}
    for original_name in reader.fieldnames:
        if original_name is None:
            continue
        key = _column_key(original_name)
        if key and key not in normalized:
            normalized[key] = original_name

    time_column = _find_column(
        normalized,
        ("time", "times", "timesec", "timeseconds"),
    )
    altitude_column = _find_column(
        normalized,
        ("altitude", "altitudem", "alt", "altm", "height", "heightm"),
    )
    if time_column is None or altitude_column is None:
        raise ValueError(
            "CSV must include Time (seconds) and Altitude (metres) columns."
        )

    velocity_column = _find_column(
        normalized,
        ("velocity", "velocityms", "speed", "speedms", "verticalvelocity"),
    )
    x_column = _find_column(
        normalized,
        ("x", "xm", "downrange", "downrangem"),
    )
    y_column = _find_column(
        normalized,
        ("y", "ym", "crossrange", "crossrangem"),
    )

    samples: list[tuple[float, float, float | None, float | None, float | None]] = []
    for row in reader:
        try:
            time = float(row[time_column] or "")
            altitude = float(row[altitude_column] or "")
        except (TypeError, ValueError):
            continue
        if not np.isfinite(time) or not np.isfinite(altitude):
            continue

        def optional_value(column: str | None) -> float | None:
            raw_value = row.get(column, "") if column is not None else ""
            if raw_value is None or not raw_value.strip():
                return None
            try:
                value = float(raw_value)
            except ValueError:
                return None
            return value if np.isfinite(value) else None

        velocity = optional_value(velocity_column)
        downrange = optional_value(x_column)
        crossrange = optional_value(y_column)
        samples.append((time, altitude, velocity, downrange, crossrange))

    if len(samples) < 2:
        raise ValueError("CSV must contain at least two valid time/altitude samples.")

    samples.sort(key=lambda sample: sample[0])
    grouped_samples: list[list[tuple[float, float, float | None, float | None, float | None]]] = []
    for sample in samples:
        if grouped_samples and sample[0] == grouped_samples[-1][0][0]:
            grouped_samples[-1].append(sample)
        else:
            grouped_samples.append([sample])

    def average_column(
        group: list[tuple[float, float, float | None, float | None, float | None]],
        column_index: int,
    ) -> float | None:
        values = [sample[column_index] for sample in group]
        present = [value for value in values if value is not None]
        return float(np.mean(present)) if present else None

    times = [group[0][0] for group in grouped_samples]
    altitudes = [float(average_column(group, 1)) for group in grouped_samples]
    velocities = [average_column(group, 2) for group in grouped_samples]
    downranges = [average_column(group, 3) for group in grouped_samples]
    crossranges = [average_column(group, 4) for group in grouped_samples]

    if len(times) < 2:
        raise ValueError("CSV must contain at least two distinct timestamps.")

    time_array = np.asarray(times, dtype=float)
    altitude_array = np.asarray(altitudes, dtype=float)
    time_array -= time_array[0]
    altitude_array -= altitude_array[0]

    if all(value is not None for value in velocities):
        velocity_array = np.asarray(velocities, dtype=float)
    else:
        velocity_array = np.gradient(altitude_array, time_array)

    has_spatial_coordinates = (
        all(value is not None for value in downranges)
        and all(value is not None for value in crossranges)
    )
    downrange_array = (
        np.asarray(downranges, dtype=float)
        if has_spatial_coordinates
        else None
    )
    crossrange_array = (
        np.asarray(crossranges, dtype=float)
        if has_spatial_coordinates
        else None
    )
    if downrange_array is not None and crossrange_array is not None:
        downrange_array -= downrange_array[0]
        crossrange_array -= crossrange_array[0]

    return FlightTelemetry(
        time_s=time_array,
        altitude_m=altitude_array,
        velocity_m_s=velocity_array,
        downrange_m=downrange_array,
        crossrange_m=crossrange_array,
    )


def compare_telemetry_to_simulation(
    simulation_time_s: NDArray[np.float64],
    simulation_altitude_m: NDArray[np.float64],
    telemetry: FlightTelemetry,
) -> TelemetryComparison:
    """Compare apogee and altitude error over the shared time interval."""
    simulation_time = np.asarray(simulation_time_s, dtype=float)
    simulation_altitude = np.asarray(simulation_altitude_m, dtype=float)
    if (
        simulation_time.ndim != 1
        or simulation_altitude.shape != simulation_time.shape
        or simulation_time.size < 2
        or not np.all(np.isfinite(simulation_time))
        or not np.all(np.isfinite(simulation_altitude))
        or np.any(np.diff(simulation_time) <= 0.0)
    ):
        raise ValueError("Simulation time and altitude must be finite, matching 1D arrays.")

    overlap = (
        (telemetry.time_s >= simulation_time[0])
        & (telemetry.time_s <= simulation_time[-1])
    )
    if not np.any(overlap):
        raise ValueError("Real telemetry and simulation have no overlapping flight time.")

    real_altitude = telemetry.altitude_m[overlap]
    simulated_altitude = np.interp(
        telemetry.time_s[overlap],
        simulation_time,
        simulation_altitude,
    )
    error = real_altitude - simulated_altitude
    simulated_apogee = float(np.max(simulation_altitude))
    real_apogee = float(np.max(telemetry.altitude_m))
    apogee_delta = real_apogee - simulated_apogee
    apogee_delta_percent = (
        100.0 * apogee_delta / simulated_apogee
        if simulated_apogee != 0.0
        else None
    )
    return TelemetryComparison(
        simulated_apogee_m=simulated_apogee,
        real_apogee_m=real_apogee,
        apogee_delta_m=apogee_delta,
        apogee_delta_percent=apogee_delta_percent,
        altitude_mae_m=float(np.mean(np.abs(error))),
        altitude_rmse_m=float(np.sqrt(np.mean(error**2))),
        compared_samples=int(np.count_nonzero(overlap)),
    )
