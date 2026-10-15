
"""
Six-degree-of-freedom rocket dynamics and trajectory integration.
Author: Tegan Broody
School: Ridley High School
Date: October 2026
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp
from scipy.optimize import OptimizeResult

from src.environment import density, wind_velocity
from src.vehicle import Rocket

G0: float = 9.80665


def _normalize_quaternion(q_quat: ArrayLike) -> NDArray[np.float64]:
    q_quat = np.asarray(q_quat, dtype=float)
    norm = np.linalg.norm(q_quat)

    if norm == 0.0:
        raise ValueError("quaternion norm cannot be zero")

    return q_quat / norm


def _quat_multiply(
    q_left: ArrayLike,
    q_right: ArrayLike,
) -> NDArray[np.float64]:
    q_left = np.asarray(q_left, dtype=float)
    q_right = np.asarray(q_right, dtype=float)

    return np.array(
        [
            q_left[0] * q_right[0]
            - q_left[1] * q_right[1]
            - q_left[2] * q_right[2]
            - q_left[3] * q_right[3],
            q_left[0] * q_right[1]
            + q_left[1] * q_right[0]
            + q_left[2] * q_right[3]
            - q_left[3] * q_right[2],
            q_left[0] * q_right[1]
            * 0.0
            + q_left[0] * q_right[2]
            - q_left[1] * q_right[3]
            + q_left[2] * q_right[0]
            + q_left[3] * q_right[1],
            q_left[0] * q_right[3]
            + q_left[1] * q_right[2]
            - q_left[2] * q_right[1]
            + q_left[3] * q_right[0],
        ],
        dtype=float,
    )


def _quat_to_rotation_matrix(
    q_quat: ArrayLike,
) -> NDArray[np.float64]:
    q_quat = _normalize_quaternion(q_quat)
    scalar, x, y, z = q_quat

    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * scalar),
                2.0 * (x * z + y * scalar),
            ],
            [
                2.0 * (x * y + z * scalar),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * scalar),
            ],
            [
                2.0 * (x * z - y * scalar),
                2.0 * (y * z + x * scalar),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=float,
    )


def initial_state_from_launch_angles(
    elevation_deg: float = 90.0,
    azimuth_deg: float = 0.0,
) -> NDArray[np.float64]:
    """Create a stationary launch state with body +Z along the rail.

    Elevation is measured above the horizontal; azimuth is measured from
    +X toward +Y. The returned quaternion is scalar-first and maps body
    coordinates to inertial coordinates.
    """
    if not np.isfinite(elevation_deg) or not 0.0 <= elevation_deg <= 90.0:
        raise ValueError(
            "launch elevation must be finite and between 0 and 90 degrees"
        )

    if not np.isfinite(azimuth_deg):
        raise ValueError("launch azimuth must be finite")

    elevation = np.deg2rad(elevation_deg)
    azimuth = np.deg2rad(azimuth_deg)

    body_inclination = np.pi / 2.0 - elevation
    half_yaw = azimuth / 2.0
    half_pitch = body_inclination / 2.0

    quaternion = np.array(
        [
            np.cos(half_yaw) * np.cos(half_pitch),
            -np.sin(half_yaw) * np.sin(half_pitch),
            np.cos(half_yaw) * np.sin(half_pitch),
            np.sin(half_yaw) * np.cos(half_pitch),
        ],
        dtype=float,
    )

    return np.array(
        [
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            *quaternion,
            0.0, 0.0, 0.0,
        ],
        dtype=float,
    )


def quaternion_to_pitch_yaw_roll(
    q_quat: ArrayLike,
) -> tuple[float, float, float]:
    """Return ZYX Euler pitch, yaw, and roll angles in radians."""
    rotation = _quat_to_rotation_matrix(q_quat)

    pitch = np.arcsin(
        np.clip(-rotation[2, 0], -1.0, 1.0)
    )
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    roll = np.arctan2(rotation[2, 1], rotation[2, 2])

    return float(pitch), float(yaw), float(roll)


def _quat_derivative(
    q_quat: ArrayLike,
    omega_body: ArrayLike,
) -> NDArray[np.float64]:
    q_quat = _normalize_quaternion(q_quat)
    omega_body = np.asarray(omega_body, dtype=float)
    omega_quat = np.array([0.0, *omega_body], dtype=float)

    return 0.5 * _quat_multiply(q_quat, omega_quat)

def _aero_forces_and_moments(
    rocket: Rocket,
    position_inertial_m: NDArray[np.float64],
    velocity_inertial_m_s: NDArray[np.float64],
    q_quat: NDArray[np.float64],
    surface_wind: tuple[float, float] = (3.0, 1.0),
    wind_enabled: bool = True,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Calculate simplified aerodynamic force and moment estimates."""
    alt = max(float(position_inertial_m[2]), 0.0)
    rho = density(alt)

    wind = (
        np.asarray(
            wind_velocity(alt, surface_wind=surface_wind),
            dtype=float,
        )
        if wind_enabled
        else np.zeros(3, dtype=float)
    )

    v_rel = velocity_inertial_m_s - wind
    speed = float(np.linalg.norm(v_rel))

    if speed < 1e-12:
        zero_force = np.zeros(3, dtype=float)
        return zero_force, zero_force.copy(), zero_force.copy()

    # Rotation matrix from body coordinates to inertial coordinates.
    C_bi = _quat_to_rotation_matrix(q_quat)

    v_body = C_bi.T @ v_rel
    v_hat = v_body / speed

    qbar = 0.5 * rho * speed**2
    area = rocket.reference_area

    lift_body = np.zeros(3, dtype=float)
    moment = np.zeros(3, dtype=float)

    cd = rocket.drag_coefficient
    
    axial_velocity = float(v_body[2])


    if axial_velocity > 1e-9:
        transverse_velocity = v_body[:2]
        transverse_speed = float(np.linalg.norm(transverse_velocity))

        angle_of_attack = float(
            np.arctan2(transverse_speed, axial_velocity)
        )

        cd += 2.0 * angle_of_attack**2

        if transverse_speed >= 1e-12:
            normal_coefficient = 2.0 * np.pi * angle_of_attack
            lift_body[:2] = (
                -qbar
                * area
                * normal_coefficient
                * transverse_velocity
                / transverse_speed
            )

            lift_body -= np.dot(lift_body, v_hat) * v_hat

            moment_arm = 0.15
            moment[0] = moment_arm * lift_body[1]
            moment[1] = -moment_arm * lift_body[0]

        if transverse_speed >= 1e-12:
            normal_coefficient = 2.0 * np.pi * angle_of_attack
            lift_body[:2] = (
                -qbar
                * area
                * normal_coefficient
                * transverse_velocity
                / transverse_speed
            )

            lift_body -= np.dot(lift_body, v_hat) * v_hat
            moment_arm = 0.15
            flight_direction = (
                1.0 if axial_velocity > 0.0 else -1.0
            )

            moment[0] = (
                flight_direction * moment_arm * lift_body[1]
            )
            moment[1] = (
                -flight_direction * moment_arm * lift_body[0]
            )

    drag_body = -qbar * area * cd * v_hat
    drag = C_bi @ drag_body
    lift = C_bi @ lift_body

    return drag, lift, moment


def rocket_state_derivative(
    t: float,
    y: ArrayLike,
    rocket: Rocket,
    *,
    surface_wind: tuple[float, float] = (3.0, 1.0),
    wind_enabled: bool = True,
) -> NDArray[np.float64]:
    """Return the derivative of the 13-state vehicle model."""
    state = np.asarray(y, dtype=float)

    if state.shape != (13,):
        raise ValueError(
            f"state vector must have shape (13,), got {state.shape}"
        )

    pos = state[0:3]
    vel = state[3:6]
    quat = _normalize_quaternion(state[6:10])
    omega = state[10:13]

    rocket.update(t)

    m = rocket.mass
    inertia = rocket.inertia_matrix()

    gravity = np.array(
        [0.0, 0.0, -G0 * m],
        dtype=float,
    )

    C_bi = _quat_to_rotation_matrix(quat)

    thrust_body = np.array(
        [0.0, 0.0, rocket.thrust],
        dtype=float,
    )
    thrust = C_bi @ thrust_body

    drag, lift, moment = _aero_forces_and_moments(
        rocket,
        pos,
        vel,
        quat,
        surface_wind=surface_wind,
        wind_enabled=wind_enabled,
    )

    force = gravity + thrust + drag + lift
    accel = force / m

    angular_momentum = inertia @ omega

    angular_accel = np.linalg.solve(
        inertia,
        moment - np.cross(omega, angular_momentum),
    )

    derivative = np.empty(13, dtype=float)

    derivative[0:3] = vel
    derivative[3:6] = accel
    derivative[6:10] = _quat_derivative(quat, omega)
    derivative[10:13] = angular_accel

    return derivative


def ground_impact(t: float, y: ArrayLike) -> float:
    
    alt = float(np.asarray(y, dtype=float)[2])

    return 1.0 if t <= 0.0 else alt

ground_impact.terminal = True
ground_impact.direction = -1


def apogee_arrival(t: float, y: ArrayLike) -> float:
    
    state = np.asarray(y, dtype=float)

    altitude = float(state[2])
    vertical_velocity = float(state[5])

    if t <= 1e-6 or altitude <= 0.0:
        return 1.0
    
    return vertical_velocity

apogee_arrival.terminal = False
apogee_arrival.direction = -1

def simulate_rocket(
    rocket: Rocket,
    initial_state: ArrayLike | None = None,
    t_span: tuple[float, float] = (0.0, 30.0),
    *,
    surface_wind: tuple[float, float] = (3.0, 1.0),
    wind_enabled: bool = True,
) -> OptimizeResult:
    """Integrate the simplified vehicle model with SciPy RK45."""
    if initial_state is None:
        initial_state = np.array(
            [
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
            ],
            dtype=float,
        )

    y0 = np.asarray(initial_state, dtype=float)

    if y0.shape != (13,):
        raise ValueError(
            f"initial_state must have shape (13,), got {y0.shape}"
        )

    def dynamics(
        t: float,
        state: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        derivative = rocket_state_derivative(
            t,
            state,
            rocket,
            surface_wind=surface_wind,
            wind_enabled=wind_enabled,
        )

        if (
            state[2] <= 1e-6
            and state[5] >= -1e-8
            and derivative[5] <= 0.0
        ):
            derivative[2] = 0.0
            derivative[5] = 0.0

        return derivative
    
    def impact_event(
        t: float,
        state: NDArray[np.float64],
    ) -> float:
        # Ignore the initial launch point.
        if t <= 1e-6:
            return 1.0

        altitude = float(state[2])
        vertical_velocity = float(state[5])

        # Detect ground impact only while descending.
        if vertical_velocity < 0.0:
            return altitude

        # Keep the event positive while ascending.
        return max(altitude, 1e-6)

    impact_event.terminal = True
    impact_event.direction = -1

    def burnout_event(
        t: float,
        _y: NDArray[np.float64],
    ) -> float:
        return t - rocket.burn_time

    burnout_event.terminal = False
    burnout_event.direction = 1

    solution = solve_ivp(
        fun=dynamics,
        t_span=t_span,
        y0=y0,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
        events=(
            impact_event,
            apogee_arrival,
            burnout_event,
        ),
        dense_output=True,
    )

    if not solution.success:
        raise RuntimeError(
            f"Simulation integration failed: {solution.message}"
        )
    print("Simulation end time:", solution.t[-1])
    print("Event times:", solution.t_events)
    print("Final state:", solution.y[:, -1])

    return solution
