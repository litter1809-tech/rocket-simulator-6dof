from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp
from scipy.optimize import OptimizeResult

from src.environment import density, wind_velocity
from src.vehicle import Rocket

G0: float = 9.80665


def _normalize_quaternion(q_quat: ArrayLike) -> NDArray[np.float64]:
    """Normalize a scalar-first quaternion to unit magnitude."""
    q_quat = np.asarray(q_quat, dtype=float)
    quaternion_norm = np.linalg.norm(q_quat)
    if quaternion_norm == 0.0:
        raise ValueError("quaternion norm cannot be zero")
    return q_quat / quaternion_norm


def _quat_multiply(q_left: ArrayLike, q_right: ArrayLike) -> NDArray[np.float64]:
    """Multiply two scalar-first quaternions using the Hamilton product."""
    q_left = np.asarray(q_left, dtype=float)
    q_right = np.asarray(q_right, dtype=float)
    return np.array([
        q_left[0] * q_right[0] - q_left[1] * q_right[1] - q_left[2] * q_right[2] - q_left[3] * q_right[3],
        q_left[0] * q_right[1] + q_left[1] * q_right[0] + q_left[2] * q_right[3] - q_left[3] * q_right[2],
        q_left[0] * q_right[2] - q_left[1] * q_right[3] + q_left[2] * q_right[0] + q_left[3] * q_right[1],
        q_left[0] * q_right[3] + q_left[1] * q_right[2] - q_left[2] * q_right[1] + q_left[3] * q_right[0],
    ], dtype=float)


def _quat_to_rotation_matrix(q_quat: ArrayLike) -> NDArray[np.float64]:
    """Return the body-to-inertial direction-cosine matrix for a quaternion."""
    q_quat = _normalize_quaternion(q_quat)
    scalar, x, y, z = q_quat
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * scalar), 2.0 * (x * z + y * scalar)],
        [2.0 * (x * y + z * scalar), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * scalar)],
        [2.0 * (x * z - y * scalar), 2.0 * (y * z + x * scalar), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=float)


def _quat_derivative(q_quat: ArrayLike, omega_body: ArrayLike) -> NDArray[np.float64]:
    """Evaluate quaternion kinematics: q_dot = 0.5 * q_quat tensor [0, omega]."""
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
    """Compute drag, lift, and aerodynamic moment from relative air velocity."""
    altitude_m = max(float(position_inertial_m[2]), 0.0)
    rho = density(altitude_m)
    wind_velocity_inertial_m_s = np.asarray(wind_velocity(altitude_m), dtype=float)
    velocity_relative_inertial_m_s = velocity_inertial_m_s - wind_velocity_inertial_m_s
    speed_relative_m_s = np.linalg.norm(velocity_relative_inertial_m_s)
    if speed_relative_m_s < 1e-12:
        zero_force = np.zeros(3, dtype=float)
        return zero_force, zero_force.copy(), zero_force.copy()

    C_body_to_inertial = _quat_to_rotation_matrix(q_quat)
    velocity_relative_body_m_s = C_body_to_inertial.T @ velocity_relative_inertial_m_s
    angle_of_attack_rad = float(np.arctan2(velocity_relative_body_m_s[2], velocity_relative_body_m_s[0]))
    q_inf = 0.5 * rho * speed_relative_m_s**2
    reference_area_m2 = rocket.reference_area
    C_d = 0.5 + 2.0 * angle_of_attack_rad**2
    C_l = 2.0 * np.pi * angle_of_attack_rad

    velocity_unit_body = velocity_relative_body_m_s / speed_relative_m_s
    F_drag_body = -q_inf * reference_area_m2 * C_d * velocity_unit_body
    lift_direction_body = np.array([0.0, 0.0, 1.0], dtype=float)
    F_lift_body = -q_inf * reference_area_m2 * C_l * lift_direction_body
    F_drag_inertial = C_body_to_inertial @ F_drag_body
    F_lift_inertial = C_body_to_inertial @ F_lift_body

    moment_arm_m = 0.15
    M_aero_body = np.array([
        0.0,
        -q_inf * reference_area_m2 * moment_arm_m * C_l,
        0.0,
    ], dtype=float)
    return F_drag_inertial, F_lift_inertial, M_aero_body


def rocket_state_derivative(t: float, y: ArrayLike, rocket: Rocket) -> NDArray[np.float64]:
    """Evaluate translational Newton and rotational Euler equations for a 13-state vehicle."""
    state_vector = np.asarray(y, dtype=float)
    if state_vector.shape != (13,):
        raise ValueError(f"state vector must have shape (13,), got {state_vector.shape}")

    position_inertial_m = state_vector[0:3]
    velocity_inertial_m_s = state_vector[3:6]
    q_quat = _normalize_quaternion(state_vector[6:10])
    omega_body_rad_s = state_vector[10:13]

    rocket.update(t)
    m = rocket.mass
    I_matrix = rocket.inertia_matrix()

    F_gravity_inertial = np.array([0.0, 0.0, -G0 * m], dtype=float)
    C_body_to_inertial = _quat_to_rotation_matrix(q_quat)
    F_thrust_body = np.array([0.0, 0.0, rocket.thrust], dtype=float)
    F_thrust_inertial = C_body_to_inertial @ F_thrust_body

    F_drag_inertial, F_lift_inertial, M_aero_body = _aero_forces_and_moments(
        rocket, position_inertial_m, velocity_inertial_m_s, q_quat
    )
    F_total_inertial = F_gravity_inertial + F_thrust_inertial + F_drag_inertial + F_lift_inertial

    acceleration_inertial_m_s2 = F_total_inertial / m
    angular_momentum_body = I_matrix @ omega_body_rad_s
    angular_acceleration_body_rad_s2 = np.linalg.solve(
        I_matrix,
        M_aero_body - np.cross(omega_body_rad_s, angular_momentum_body),
    )

    state_derivative = np.empty(13, dtype=float)
    state_derivative[0:3] = velocity_inertial_m_s
    state_derivative[3:6] = acceleration_inertial_m_s2
    state_derivative[6:10] = _quat_derivative(q_quat, omega_body_rad_s)
    state_derivative[10:13] = angular_acceleration_body_rad_s2
    return state_derivative


def ground_impact(t: float, y: ArrayLike) -> float:
    """Return altitude for the terminal ground-contact event after launch."""
    altitude_m = float(np.asarray(y, dtype=float)[2])
    return 1.0 if t <= 0.0 else altitude_m


ground_impact.terminal = True
ground_impact.direction = -1


def simulate_rocket(
    rocket: Rocket,
    initial_state: ArrayLike | None = None,
    t_span: tuple[float, float] = (0.0, 30.0),
) -> OptimizeResult:
    """Integrate the 6-DoF state with RK45 and terminate at ground impact."""
    if initial_state is None:
        initial_state = np.array([
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ], dtype=float)

    initial_state_vector = np.asarray(initial_state, dtype=float)
    if initial_state_vector.shape != (13,):
        raise ValueError(f"initial_state must have shape (13,), got {initial_state_vector.shape}")

    return solve_ivp(
        fun=lambda t, y: rocket_state_derivative(t, y, rocket),
        t_span=t_span,
        y0=initial_state_vector,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
        events=ground_impact,
        dense_output=True,
    )
