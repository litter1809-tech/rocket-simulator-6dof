import pytest

from src.environment import density, pressure, speed_of_sound, temperature


def test_sea_level_density() -> None:
    assert density(0.0) == pytest.approx(1.225, rel=1e-4)


def test_sea_level_speed_of_sound() -> None:
    assert speed_of_sound(0.0) == pytest.approx(340.29, rel=1e-4)


@pytest.mark.parametrize(
    ("altitude_m", "expected_temperature_k", "expected_pressure_pa", "expected_density"),
    [
        (11_000.0, 216.77, 22_699.0, 0.3648),
        (20_000.0, 216.65, 5_529.0, 0.0889),
    ],
)
def test_1976_standard_atmosphere_reference_layers(
    altitude_m: float,
    expected_temperature_k: float,
    expected_pressure_pa: float,
    expected_density: float,
) -> None:
    assert temperature(altitude_m) == pytest.approx(expected_temperature_k, rel=5e-4)
    assert pressure(altitude_m) == pytest.approx(expected_pressure_pa, rel=5e-3)
    assert density(altitude_m) == pytest.approx(expected_density, rel=2e-3)
