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


def _quat_multiply(q_left: ArrayLike, q_right: ArrayLike) -> NDArray[np.float64]:
    q_left = np.asarray(q_left, dtype=float)
    q_right = np.asarray(q_right, dtype=float)
    return np.array([
        q_left[0] * q_right[0] - q_left[1] * q_right[1] - q_left[2] * q_right[2] - q_left[3] * q_right[3],
        q_left[0] * q_right[1] + q_left[1] * q_right[0] + q_left[2] * q_right[3] - q_left[3] * q_right[2],
        q_left[0] * q_right[2] - q_left[1] * q_right[3] + q_left[2] * q_right[0] + q_left[3] * q_right[1],
        q_left[0] * q_right[3] + q_left[1] * q_right[2] - q_left[2] * q_right[1] + q_left[3] * q_right[0],
    ], dtype=float)


def _quat_to_rotation_matrix(q_quat: ArrayLike) -> NDArray[np.float64]:
    q_quat = _normalize_quaternion(q_quat)
    scalar, x, y, z = q_quat
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * scalar), 2.0 * (x * z + y * scalar)],
        [2.0 * (x * y + z * scalar), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * scalar)],
        [2.0 * (x * z - y * scalar), 2.0 * (y * z + x * scalar), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=float)


def _quat_derivative(q_quat: ArrayLike, omega_body: ArrayLike) -> NDArray[np.float64]:
    q_quat = _normalize_quaternion(q_quat)
    omega_body = np.asarray(omega_body, dtype=float)
    omega_quat = np.array([0.0, *omega_body], dtype=float)
    return 0.5 * _quat_multiply(q_quat, omega_quat)


def _aero_forces_and_moments(
    rocket: Rocket,
    position_inertial_m: NDArray[np.float64],
    velocity_inertial_m_s: NDArray[np.float64],
    q_quat: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    alt = max(float(position_inertial_m[2]), 0.0)
    rho = density(alt)
    wind = np.asarray(wind_velocity(alt), dtype=float)
    v_rel = velocity_inertial_m_s - wind
    speed = np.linalg.norm(v_rel)
    if speed < 1e-12:
        zero_force = np.zeros(3, dtype=float)
        return zero_force, zero_force.copy(), zero_force.copy()

    C_bi = _quat_to_rotation_matrix(q_quat)
    v_body = C_bi.T @ v_rel
    alpha = float(np.arctan2(v_body[2], v_body[0]))
    qbar = 0.5 * rho * speed**2
    area = rocket.reference_area
    cd = 0.5 + 2.0 * alpha**2
    cl = 2.0 * np.pi * alpha

    v_hat = v_body / speed
    drag_body = -qbar * area * cd * v_hat
    lift_dir = np.array([0.0, 0.0, 1.0], dtype=float)
    lift_body = -qbar * area * cl * lift_dir
    drag = C_bi @ drag_body
    lift = C_bi @ lift_body

    # Fixed arm and simple coefficients; replace with vehicle aero data when available.
    moment_arm = 0.15
    moment = np.array([
        0.0,
        -qbar * area * moment_arm * cl,
        0.0,
    ], dtype=float)
    return drag, lift, moment


def rocket_state_derivative(t: float, y: ArrayLike, rocket: Rocket) -> NDArray[np.float64]:
    """Return the derivative of the 13-state 6-DoF vehicle model."""
    state = np.asarray(y, dtype=float)
    if state.shape != (13,):
        raise ValueError(f"state vector must have shape (13,), got {state.shape}")

    pos = state[0:3]
    vel = state[3:6]
    quat = _normalize_quaternion(state[6:10])
    omega = state[10:13]

    rocket.update(t)
    m = rocket.mass
    inertia = rocket.inertia_matrix()

    gravity = np.array([0.0, 0.0, -G0 * m], dtype=float)
    C_bi = _quat_to_rotation_matrix(quat)
    thrust_body = np.array([0.0, 0.0, rocket.thrust], dtype=float)
    thrust = C_bi @ thrust_body

    drag, lift, moment = _aero_forces_and_moments(rocket, pos, vel, quat)
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
    """Altitude event used to stop integration at ground contact."""
    alt = float(np.asarray(y, dtype=float)[2])
    return 1.0 if t <= 0.0 else alt


ground_impact.terminal = True
ground_impact.direction = -1


def simulate_rocket(
    rocket: Rocket,
    initial_state: ArrayLike | None = None,
    t_span: tuple[float, float] = (0.0, 30.0),
) -> OptimizeResult:
    """Integrate the 6-DoF state with RK45 until ground impact."""
    if initial_state is None:
        initial_state = np.array([
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ], dtype=float)

    y0 = np.asarray(initial_state, dtype=float)
    if y0.shape != (13,):
        raise ValueError(f"initial_state must have shape (13,), got {y0.shape}")

    return solve_ivp(
        fun=lambda t, y: rocket_state_derivative(t, y, rocket),
        t_span=t_span,
        y0=y0,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
        events=ground_impact,
        dense_output=True,
    )
