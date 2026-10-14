import pytest

from src.environment import density, speed_of_sound


def test_sea_level_density() -> None:
    assert density(0.0) == pytest.approx(1.225, rel=1e-4)


def test_sea_level_speed_of_sound() -> None:
    assert speed_of_sound(0.0) == pytest.approx(340.29, rel=1e-4)
