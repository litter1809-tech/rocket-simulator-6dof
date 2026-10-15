"""
Vectorized point-mass Monte Carlo analysis of rocket landing dispersion.
Author: Tegan Broody
School: Ridley High School
Date: October 2026
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.environment import H_MAX, density
from src.physics import G0
from src.vehicle import Rocket


def run_monte_carlo_dispersion(
    rocket: Rocket,
    *,
    iterations: int,
    launch_elevation_deg: float,
    launch_azimuth_deg: float,
    wind_speed_m_s: float,
    seed: int = 2026,
) -> dict[str, NDArray[np.float64]]:
    """Estimate landing dispersion using vectorized point-mass kinematics.

    This intentionally trades 6-DOF fidelity for a fast fixed-step model. Every
    stochastic sample is advanced together in NumPy arrays; nominal flight
    simulation continues to use the full RK45 6-DOF engine.
    """
    if not 50 <= iterations <= 100:
        raise ValueError("Monte Carlo iterations must be between 50 and 100.")
    if not np.isfinite(wind_speed_m_s) or wind_speed_m_s < 0.0:
        raise ValueError("Wind speed must be finite and nonnegative.")
    if not np.isfinite(launch_elevation_deg) or not 0.0 <= launch_elevation_deg <= 90.0:
        raise ValueError("Launch elevation must be between 0 and 90 degrees.")
    if not np.isfinite(launch_azimuth_deg):
        raise ValueError("Launch azimuth must be finite.")

    random = np.random.default_rng(seed)
    nominal_azimuth_rad = np.deg2rad(launch_azimuth_deg)
    wind_sigma_m_s = max(0.2 * wind_speed_m_s, 0.5)
    sampled_wind_m_s = np.maximum(
        random.normal(wind_speed_m_s, wind_sigma_m_s, iterations),
        0.0,
    )
    wind_direction_rad = (
        nominal_azimuth_rad
        + np.pi / 2.0
        + random.uniform(-np.pi / 6.0, np.pi / 6.0, iterations)
    )
    sampled_thrust_scale = np.maximum(
        random.normal(1.0, 0.03, iterations),
        0.0,
    )
    sampled_elevation_deg = np.clip(
        random.normal(launch_elevation_deg, 0.5, iterations),
        0.1,
        90.0,
    )
    sampled_azimuth_rad = nominal_azimuth_rad + np.deg2rad(
        random.normal(0.0, 1.0, iterations)
    )

    elevation_rad = np.deg2rad(sampled_elevation_deg)
    thrust_direction = np.column_stack(
        (
            np.cos(elevation_rad) * np.cos(sampled_azimuth_rad),
            np.cos(elevation_rad) * np.sin(sampled_azimuth_rad),
            np.sin(elevation_rad),
        )
    )
    wind_east = sampled_wind_m_s * np.cos(wind_direction_rad)
    wind_north = sampled_wind_m_s * np.sin(wind_direction_rad)
    wind_x = wind_east * np.cos(nominal_azimuth_rad) + wind_north * np.sin(
        nominal_azimuth_rad
    )
    wind_y = -wind_east * np.sin(nominal_azimuth_rad) + wind_north * np.cos(
        nominal_azimuth_rad
    )

    burn_time_s = rocket.burn_time
    if rocket.thrust_curve is None:
        mean_thrust_n = rocket.average_thrust
    else:
        mean_thrust_n = rocket.thrust_curve.total_impulse_ns / burn_time_s
    mean_burn_mass_kg = rocket.dry_mass + 0.5 * rocket.propellant_mass
    thrust_acceleration = (
        mean_thrust_n
        * sampled_thrust_scale[:, None]
        * thrust_direction
        / mean_burn_mass_kg
    )
    vertical_burn_acceleration = thrust_acceleration[:, 2] - G0
    airborne = vertical_burn_acceleration > 0.0
    vertical_velocity_at_burnout = np.where(
        airborne,
        vertical_burn_acceleration * burn_time_s,
        0.0,
    )
    altitude_at_burnout = np.where(
        airborne,
        0.5 * vertical_burn_acceleration * burn_time_s**2,
        0.0,
    )
    coast_time_s = np.zeros(iterations, dtype=float)
    coast_time_s[airborne] = (
        vertical_velocity_at_burnout[airborne]
        + np.sqrt(
            vertical_velocity_at_burnout[airborne] ** 2
            + 2.0 * G0 * altitude_at_burnout[airborne]
        )
    ) / G0
    apogee_m = np.where(
        airborne,
        altitude_at_burnout + vertical_velocity_at_burnout**2 / (2.0 * G0),
        0.0,
    )
    flight_time_s = np.where(airborne, burn_time_s + coast_time_s, 0.0)

    jet_fraction = np.exp(
        -0.5 * ((0.5 * apogee_m - 11_000.0) / 3_500.0) ** 2
    )
    jet_east_m_s = 40.0 * jet_fraction
    jet_north_m_s = 12.0 * jet_fraction
    wind_x += jet_east_m_s * np.cos(nominal_azimuth_rad) + jet_north_m_s * np.sin(
        nominal_azimuth_rad
    )
    wind_y += -jet_east_m_s * np.sin(nominal_azimuth_rad) + jet_north_m_s * np.cos(
        nominal_azimuth_rad
    )
    local_wind = np.column_stack(
        (wind_x, wind_y, np.zeros(iterations, dtype=float))
    )

    burnout_velocity_m_s = thrust_acceleration.copy()
    burnout_velocity_m_s[:, 2] -= G0
    burnout_velocity_m_s *= burn_time_s
    burnout_velocity_m_s[~airborne] = 0.0
    representative_altitude_m = np.clip(0.5 * apogee_m, 0.0, H_MAX)
    representative_density = np.asarray(
        density(representative_altitude_m),
        dtype=float,
    )
    representative_speed_m_s = np.maximum(
        np.linalg.norm(burnout_velocity_m_s - local_wind, axis=1),
        1.0,
    )
    drag_rate_s = (
        0.5
        * representative_density
        * rocket.drag_coefficient
        * rocket.reference_area
        * representative_speed_m_s
        / mean_burn_mass_kg
    )

    def exponential_response(time_s: NDArray[np.float64]) -> NDArray[np.float64]:
        scaled_time = drag_rate_s * time_s
        response = np.empty(iterations, dtype=float)
        small = np.abs(scaled_time) < 1e-4
        response[small] = time_s[small] * (
            1.0
            - 0.5 * scaled_time[small]
            + scaled_time[small] ** 2 / 6.0
        )
        response[~small] = (
            -np.expm1(-scaled_time[~small]) / drag_rate_s[~small]
        )
        return response

    powered_velocity_gain_s = exponential_response(
        np.full(iterations, burn_time_s, dtype=float)
    )
    powered_displacement_gain_s = np.empty(iterations, dtype=float)
    scaled_burn_time = drag_rate_s * burn_time_s
    small_drag = np.abs(scaled_burn_time) < 1e-4
    powered_displacement_gain_s[small_drag] = burn_time_s**2 * (
        0.5
        - scaled_burn_time[small_drag] / 6.0
        + scaled_burn_time[small_drag] ** 2 / 24.0
    )
    powered_displacement_gain_s[~small_drag] = (
        burn_time_s - powered_velocity_gain_s[~small_drag]
    ) / drag_rate_s[~small_drag]

    powered_position_xy = (
        local_wind[:, :2]
        * (burn_time_s - powered_velocity_gain_s)[:, None]
        + thrust_acceleration[:, :2] * powered_displacement_gain_s[:, None]
    )
    burnout_velocity_xy = (
        local_wind[:, :2]
        * (1.0 - np.exp(-scaled_burn_time))[:, None]
        + thrust_acceleration[:, :2] * powered_velocity_gain_s[:, None]
    )
    coast_response_s = exponential_response(coast_time_s)
    coast_position_xy = (
        local_wind[:, :2] * coast_time_s[:, None]
        + (burnout_velocity_xy - local_wind[:, :2])
        * coast_response_s[:, None]
    )
    landing_xy = powered_position_xy + coast_position_xy
    landing_x_m = np.where(airborne, landing_xy[:, 0], 0.0)
    landing_y_m = np.where(airborne, landing_xy[:, 1], 0.0)

    return {
        "landing_x_m": landing_x_m,
        "landing_y_m": landing_y_m,
        "apogee_m": apogee_m,
        "flight_time_s": flight_time_s,
        "wind_speed_m_s": sampled_wind_m_s,
        "thrust_scale": sampled_thrust_scale,
        "launch_elevation_deg": sampled_elevation_deg,
        "launch_azimuth_deg": np.rad2deg(sampled_azimuth_rad),
    }
