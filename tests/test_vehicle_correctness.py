import numpy as np
import pytest

from src.vehicle import Rocket

@pytest.fixture
def rocket():
    return Rocket(
        dry_mass=12.0,
        propellant_mass=3.0,
        burn_time=5.0,
        average_thrust=1200.0,
        reference_area=np.pi * (0.10 ** 2) / 4.0,
        drag_coefficient=0.50,
        moment_of_inertia=0.1,
    )

def test_initial_mass_and_thrust(rocket):
    mass, thrust = rocket.update(0.0)

    assert mass == pytest.approx(15.0)
    assert thrust == pytest.approx(1200.0)

def test_halfway_through_burn(rocket):
    mass, thrust = rocket.update(2.5)

    assert mass == pytest.approx(13.5)
    assert thrust == pytest.approx(1200.0)

def test_mass_and_thrust_at_burnout(rocket):
    mass, thrust = rocket.update(5.0)

    assert mass == pytest.approx(12.0)
    assert thrust == pytest.approx(0.0)

def test_mass_and_thrust_after_burnout(rocket):
    mass, thrust = rocket.update(10.0)

    assert mass == pytest.approx(12.0)
    assert thrust == pytest.approx(0.0)

def test_negative_time_is_rejected(rocket):
    with pytest.raises(ValueError):
        rocket.update(-1.0)

def test_nonfinite_time_is_rejected(rocket):
    with pytest.raises(ValueError):
        rocket.update(float("inf"))
