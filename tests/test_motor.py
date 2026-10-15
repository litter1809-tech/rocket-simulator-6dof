import pytest
import numpy as np

from src.motor import parse_eng_motor
from src.physics import initial_state_from_launch_angles, simulate_rocket
from src.vehicle import Rocket
from src.motor import ThrustCurve

SAMPLE_ENG = """; RASP motor file
MOTOR-A 18 70 0 4.0 8.0 Example manufacturer
0.0 0.0
1.0 10.0
2.0 10.0
3.0 0.0
"""

def test_parse_eng_motor_and_integrate_curve() -> None:
    motor = parse_eng_motor(SAMPLE_ENG)

    assert motor.designation == "MOTOR-A"
    assert motor.propellant_mass_kg == pytest.approx(0.004)
    assert motor.burn_time_s == pytest.approx(3.0)
    assert motor.total_impulse_ns == pytest.approx(20.0)
    assert motor.thrust_at(0.5) == pytest.approx(5.0)
    assert motor.impulse_until(1.0) == pytest.approx(5.0)
    assert motor.impulse_until(2.0) == pytest.approx(15.0)


def test_custom_motor_curve_drives_thrust_and_propellant_depletion() -> None:
    motor = parse_eng_motor(SAMPLE_ENG)
    rocket = Rocket(
        dry_mass=1.0,
        propellant_mass=motor.propellant_mass_kg or 0.0,
        burn_time=1.0,
        average_thrust=100.0,
        reference_area=0.01,
        drag_coefficient=0.5,
        moment_of_inertia=1.0,
        thrust_curve=motor,
    )

    mass_at_one_second, thrust_at_one_second = rocket.update(1.0)
    mass_at_two_seconds, thrust_at_two_seconds = rocket.update(2.0)
    mass_at_burnout, thrust_at_burnout = rocket.update(3.0)

    assert rocket.burn_time == pytest.approx(3.0)
    assert thrust_at_one_second == pytest.approx(10.0)
    assert mass_at_one_second == pytest.approx(1.003)
    assert thrust_at_two_seconds == pytest.approx(10.0)
    assert mass_at_two_seconds == pytest.approx(1.001)
    assert thrust_at_burnout == pytest.approx(0.0)
    assert mass_at_burnout == pytest.approx(1.0)


def test_custom_motor_curve_sets_integrated_burnout_event() -> None:
    motor = parse_eng_motor(SAMPLE_ENG)
    rocket = Rocket(
        dry_mass=0.1,
        propellant_mass=motor.propellant_mass_kg or 0.0,
        burn_time=1.0,
        average_thrust=100.0,
        reference_area=0.001,
        drag_coefficient=0.5,
        moment_of_inertia=0.01,
        thrust_curve=motor,
    )

    solution = simulate_rocket(
        rocket,
        initial_state_from_launch_angles(90.0, 0.0),
        (0.0, 3.5),
        surface_wind=(0.0, 0.0),
        wind_enabled=False,
    )

    assert solution.success
    assert solution.t_events[2][0] == pytest.approx(motor.burn_time_s)
    assert solution.t_events[1].size == 0
    assert rocket.mass == pytest.approx(rocket.dry_mass)
    assert rocket.thrust == pytest.approx(0.0)

def test_thrust_is_zero_at_and_after_burnout():
    motor = ThrustCurve(
        times_s=np.array([0.0, 1.0, 2.0]),
        thrust_n=np.array([0.0, 100.0, 0.0]),
        propellant_mass_kg=0.1,
    )

    assert motor.thrust_at(2.0) == 0.0
    assert motor.thrust_at(3.0) == 0.0


def test_positive_final_thrust_is_rejected():
    with pytest.raises(ValueError, match="final burnout time"):
        ThrustCurve(
            times_s=np.array([0.0, 1.0, 2.0]),
            thrust_n=np.array([0.0, 100.0, 10.0]),
            propellant_mass_kg=0.1,
        )


def test_rocket_rejects_mismatched_propellant_mass():
    motor = ThrustCurve(
        times_s=np.array([0.0, 1.0, 2.0]),
        thrust_n=np.array([0.0, 100.0, 0.0]),
        propellant_mass_kg=0.1,
    )

    with pytest.raises(ValueError, match="does not match"):
        Rocket(
            dry_mass=1.0,
            propellant_mass=0.2,
            burn_time=2.0,
            average_thrust=50.0,
            reference_area=0.01,
            drag_coefficient=0.5,
            moment_of_inertia=0.1,
            thrust_curve=motor,
        )


def test_rocket_accepts_matching_propellant_mass():
    motor = ThrustCurve(
        times_s=np.array([0.0, 1.0, 2.0]),
        thrust_n=np.array([0.0, 100.0, 0.0]),
        propellant_mass_kg=0.1,
    )

    rocket = Rocket(
        dry_mass=1.0,
        propellant_mass=0.1,
        burn_time=2.0,
        average_thrust=50.0,
        reference_area=0.01,
        drag_coefficient=0.5,
        moment_of_inertia=0.1,
        thrust_curve=motor,
    )

    assert rocket.update(2.0) == pytest.approx((1.0, 0.0))


@pytest.mark.parametrize(
    "contents",
    [
        "",
        "MOTOR-A 18 70 0 4.0 8.0 Example\n0 0\n0 10\n",
        "MOTOR-A 18 70 0 0 8.0 Example\n0 0\n1 1\n",
    ],
)
def test_parse_eng_motor_rejects_invalid_files(contents: str) -> None:
    with pytest.raises(ValueError):
        parse_eng_motor(contents)
