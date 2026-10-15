import numpy as np
import pytest

from src.environment import wind_velocity
from src.physics import (
    _aero_forces_and_moments,
    _normalize_quaternion,
    _quat_to_rotation_matrix,
    initial_state_from_launch_angles,
    quaternion_to_pitch_yaw_roll,
    rocket_state_derivative,
    simulate_rocket,
)
from src.vehicle import Rocket


def test_drag_coefficient_changes_longitudinal_acceleration() -> None:
    state = np.array(
        [
            0.0, 0.0, 100.0,
            100.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ],
        dtype=float,
    )
    common = {
        "dry_mass": 10.0,
        "propellant_mass": 0.0,
        "burn_time": 1.0,
        "average_thrust": 0.0,
        "reference_area": 0.01,
        "moment_of_inertia": 1.0,
    }
    low_drag = Rocket(**common, drag_coefficient=0.2)
    high_drag = Rocket(**common, drag_coefficient=0.8)

    low_drag_acceleration = rocket_state_derivative(1.0, state, low_drag)[3]
    high_drag_acceleration = rocket_state_derivative(1.0, state, high_drag)[3]

    assert high_drag_acceleration < low_drag_acceleration


def test_quaternion_and_solver_inputs_must_be_finite() -> None:
    with pytest.raises(ValueError, match="four finite"):
        _normalize_quaternion([1.0, 0.0, 0.0, np.nan])

    rocket = Rocket(
        dry_mass=1.0,
        propellant_mass=0.0,
        burn_time=1.0,
        average_thrust=0.0,
        reference_area=0.01,
        drag_coefficient=0.0,
        moment_of_inertia=1.0,
    )
    invalid_state = initial_state_from_launch_angles()
    invalid_state[3] = np.inf
    with pytest.raises(ValueError, match="only finite"):
        simulate_rocket(rocket, invalid_state)

    with pytest.raises(ValueError, match="start < end"):
        simulate_rocket(rocket, t_span=(1.0, 1.0))


def test_launch_state_points_along_requested_rail_angles() -> None:
    elevation_deg = 70.0
    azimuth_deg = 35.0
    initial_state = initial_state_from_launch_angles(elevation_deg, azimuth_deg)

    direction = _quat_to_rotation_matrix(initial_state[6:10])[:, 2]
    elevation = np.deg2rad(elevation_deg)
    azimuth = np.deg2rad(azimuth_deg)

    np.testing.assert_allclose(
        direction,
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ],
        atol=1e-12,
    )
    assert initial_state.shape == (13,)
    assert np.all(initial_state[:6] == 0.0)


def test_attitude_angles_report_launch_orientation() -> None:
    initial_state = initial_state_from_launch_angles(70.0, 35.0)

    pitch, yaw, roll = np.rad2deg(
        quaternion_to_pitch_yaw_roll(initial_state[6:10])
    )

    np.testing.assert_allclose([pitch, yaw, roll], [20.0, 35.0, 0.0], atol=1e-12)


def test_aerodynamic_angle_uses_body_z_as_rocket_longitudinal_axis() -> None:
    rocket = Rocket(
        dry_mass=10.0,
        propellant_mass=0.0,
        burn_time=1.0,
        average_thrust=0.0,
        reference_area=0.01,
        drag_coefficient=0.5,
        moment_of_inertia=1.0,
    )
    altitude_m = 100.0
    wind = np.asarray(wind_velocity(altitude_m))
    velocity_m_s = wind + np.array([0.0, 0.0, 100.0])
    _, lift, moment = _aero_forces_and_moments(
        rocket,
        np.array([0.0, 0.0, altitude_m]),
        velocity_m_s,
        np.array([1.0, 0.0, 0.0, 0.0]),
    )

    np.testing.assert_allclose(lift, np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(moment, np.zeros(3), atol=1e-12)


def test_positive_angle_of_attack_produces_restoring_pitch_moment() -> None:
    rocket = Rocket(
        dry_mass=10.0,
        propellant_mass=0.0,
        burn_time=1.0,
        average_thrust=0.0,
        reference_area=0.01,
        drag_coefficient=0.5,
        moment_of_inertia=1.0,
    )
    _, _, moment = _aero_forces_and_moments(
        rocket,
        np.array([0.0, 0.0, 100.0]),
        np.array([10.0, 0.0, 100.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        surface_wind=(0.0, 0.0),
        wind_enabled=False,
    )

    assert moment[1] > 0.0


def test_positive_crossflow_produces_restoring_yaw_moment() -> None:
    rocket = Rocket(
        dry_mass=10.0,
        propellant_mass=0.0,
        burn_time=1.0,
        average_thrust=0.0,
        reference_area=0.01,
        drag_coefficient=0.5,
        moment_of_inertia=1.0,
    )
    _, lift, moment = _aero_forces_and_moments(
        rocket,
        np.array([0.0, 0.0, 100.0]),
        np.array([0.0, 10.0, 100.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        surface_wind=(0.0, 0.0),
        wind_enabled=False,
    )

    assert lift[1] < 0.0
    assert moment[0] < 0.0


def test_reverse_axial_flow_does_not_create_spurious_lift_or_moment() -> None:
    rocket = Rocket(
        dry_mass=10.0,
        propellant_mass=0.0,
        burn_time=1.0,
        average_thrust=0.0,
        reference_area=0.01,
        drag_coefficient=0.5,
        moment_of_inertia=1.0,
    )
    drag, lift, moment = _aero_forces_and_moments(
        rocket,
        np.array([0.0, 0.0, 100.0]),
        np.array([10.0, 0.0, -100.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        surface_wind=(0.0, 0.0),
        wind_enabled=False,
    )

    assert drag[2] > 0.0
    np.testing.assert_array_equal(lift, np.zeros(3))
    np.testing.assert_array_equal(moment, np.zeros(3))


def test_wind_produces_crossrange_drift_during_flight() -> None:
    rocket = Rocket(
        dry_mass=12.0,
        propellant_mass=3.0,
        burn_time=5.0,
        average_thrust=1200.0,
        reference_area=np.pi * 0.05**2,
        drag_coefficient=0.5,
        moment_of_inertia=8.0,
    )
    solution = simulate_rocket(
        rocket,
        initial_state_from_launch_angles(70.0, 35.0),
        (0.0, 60.0),
    )

    assert solution.success
    assert solution.t_events[0].size == 1
    assert solution.t_events[1].size == 1
    assert solution.t_events[2].size == 1
    assert solution.t_events[2][0] == rocket.burn_time
    apogee_state = solution.sol(solution.t_events[1][0])
    assert abs(apogee_state[5]) < 1e-7
    assert np.any(np.abs(solution.y[0]) > 0.0)
    assert np.any(np.abs(solution.y[1]) > 0.0)


def test_configurable_crosswind_changes_crossrange_trajectory() -> None:
    def launch(surface_wind: tuple[float, float]) -> float:
        rocket = Rocket(
            dry_mass=12.0,
            propellant_mass=3.0,
            burn_time=5.0,
            average_thrust=1200.0,
            reference_area=np.pi * 0.05**2,
            drag_coefficient=0.5,
            moment_of_inertia=8.0,
        )
        solution = simulate_rocket(
            rocket,
            initial_state_from_launch_angles(70.0, 0.0),
            (0.0, 30.0),
            surface_wind=surface_wind,
        )
        assert solution.success
        return float(solution.y[1, -1])

    calm_crosswind = launch((0.0, 0.0))
    strong_crosswind = launch((0.0, 20.0))

    assert not np.allclose(calm_crosswind, strong_crosswind)


def test_zero_wind_vertical_launch_has_exactly_zero_crossrange() -> None:
    rocket = Rocket(
        dry_mass=12.0,
        propellant_mass=3.0,
        burn_time=5.0,
        average_thrust=1200.0,
        reference_area=np.pi * 0.05**2,
        drag_coefficient=0.5,
        moment_of_inertia=8.0,
    )
    solution = simulate_rocket(
        rocket,
        initial_state_from_launch_angles(90.0, 0.0),
        (0.0, 200.0),
        surface_wind=(0.0, 0.0),
        wind_enabled=False,
    )

    assert solution.success
    np.testing.assert_array_equal(solution.y[1], 0.0)
    np.testing.assert_array_equal(solution.y[4], 0.0)
    np.testing.assert_array_equal(solution.y[0], 0.0)
    np.testing.assert_array_equal(solution.y[3], 0.0)
    assert solution.t_events[0].size == 1
    assert solution.t[-1] == solution.t_events[0][0]


def test_drag_free_ballistic_flight_conserves_mechanical_energy_and_momentum() -> None:
    mass_kg = 2.0
    initial_altitude_m = 100.0
    initial_vertical_velocity_m_s = 20.0
    rocket = Rocket(
        dry_mass=mass_kg,
        propellant_mass=0.0,
        burn_time=1.0,
        average_thrust=0.0,
        reference_area=0.01,
        drag_coefficient=0.0,
        moment_of_inertia=1.0,
    )
    initial_state = np.array(
        [
            0.0, 0.0, initial_altitude_m,
            0.0, 0.0, initial_vertical_velocity_m_s,
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ],
        dtype=float,
    )
    solution = simulate_rocket(
        rocket,
        initial_state,
        (0.0, 1.9),
        surface_wind=(0.0, 0.0),
        wind_enabled=False,
    )

    assert solution.success
    potential_energy = mass_kg * 9.80665 * solution.y[2]
    kinetic_energy = 0.5 * mass_kg * np.sum(solution.y[3:6] ** 2, axis=0)
    total_energy = potential_energy + kinetic_energy
    vertical_momentum_corrected = (
        mass_kg * solution.y[5] + mass_kg * 9.80665 * solution.t
    )
    np.testing.assert_allclose(total_energy, total_energy[0], rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(
        vertical_momentum_corrected,
        vertical_momentum_corrected[0],
        rtol=2e-6,
        atol=2e-6,
    )
