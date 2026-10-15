import numpy as np
import pytest

from src.environment import (
    density,
    pressure,
    speed_of_sound,
    temperature,
    wind_velocity,
)

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

def test_disabled_wind_is_zero_at_all_altitudes() -> None:
    altitudes = np.array([
        0.0,
        1000.0,
        11000.0,
        20000.0,
        30000.0,
        50000.0,
        86000.0,
    ])

    wind = wind_velocity(altitudes, enabled=False)

    assert wind.shape == (7, 3)
    np.testing.assert_array_equal(wind, np.zeros((7, 3)))

def test_disabled_wind_is_zero_at_scalar_altitude() -> None:
    wind = wind_velocity(1000.0, enabled=False)

    assert wind.shape == (3,)
    np.testing.assert_array_equal(wind, np.zeros(3))

def test_surface_wind_at_sea_level() -> None:
    wind = wind_velocity(
        0.0,
        surface_wind=(3.0, 1.0),
        jet_wind=(0.0, 0.0),
        vertical_amplitude=0.0,
    )

    np.testing.assert_allclose(wind, [3.0, 1.0, 0.0])
    
@pytest.mark.parametrize(
    "altitude_m",
    [25000.0, 30000.0, 50000.0, 71000.0, 86000.0],
)
def test_atmosphere_works_at_high_altitudes(
    altitude_m: float,
) -> None:
    temp = temperature(altitude_m)
    p = pressure(altitude_m)
    rho = density(altitude_m)
    sound = speed_of_sound(altitude_m)

    assert np.isfinite(temp)
    assert np.isfinite(p)
    assert np.isfinite(rho)
    assert np.isfinite(sound)

    assert temp > 0.0
    assert p > 0.0
    assert rho > 0.0
    assert sound > 0.0

def test_pressure_and_density_decrease_at_high_altitudes() -> None:
    altitudes = [20000.0, 30000.0, 50000.0, 70000.0, 86000.0]

    pressures = [pressure(h) for h in altitudes]
    densities = [density(h) for h in altitudes]

    assert all(
        higher < lower
        for lower, higher in zip(pressures, pressures[1:])
    )

    assert all(
        higher < lower
        for lower, higher in zip(densities, densities[1:])
    )

def test_altitude_above_limit_is_rejected() -> None:
    with pytest.raises(ValueError):
        temperature(87000.0)

def test_wind_works_at_high_altitudes() -> None:
    altitudes = np.array([30000.0, 50000.0, 86000.0])

    wind = wind_velocity(altitudes)

    assert wind.shape == (3, 3)
    assert np.all(np.isfinite(wind))
