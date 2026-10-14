from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

G0: float = 9.80665
R_STAR: float = 8.31432
M0: float = 0.0289644
R_AIR: float = R_STAR / M0
GAMMA: float = 1.4
R_EARTH: float = 6_356_766.0
P_SEA_LEVEL: float = 101_325.0

H_MIN: float = 0.0
H_MAX: float = 20_000.0

_LAYERS: NDArray[np.float64] = np.array([
    [0.0, 288.15, -0.0065],
    [11_000.0, 216.65, 0.0000],
])
_HB, _TB, _LAPSE = _LAYERS.T


def _layer_temperature(
    H: NDArray[np.float64] | np.float64,
    Hb: NDArray[np.float64] | np.float64,
    Tb: NDArray[np.float64] | np.float64,
    L: NDArray[np.float64] | np.float64,
) -> NDArray[np.float64] | np.float64:
    """Evaluate the ISA 1976 linear temperature profile within a layer."""
    return Tb + L * (H - Hb)


def _layer_pressure(
    H: NDArray[np.float64] | np.float64,
    Hb: NDArray[np.float64] | np.float64,
    Tb: NDArray[np.float64] | np.float64,
    L: NDArray[np.float64] | np.float64,
    Pb: NDArray[np.float64] | np.float64,
) -> NDArray[np.float64] | np.float64:
    """Evaluate hydrostatic pressure for gradient and isothermal ISA layers."""
    temperature_K = _layer_temperature(H, Hb, Tb, L)
    safe_lapse_rate = np.where(L == 0.0, 1.0, L)
    pressure_gradient = Pb * (temperature_K / Tb) ** (-G0 / (R_AIR * safe_lapse_rate))
    pressure_isothermal = Pb * np.exp(-G0 * (H - Hb) / (R_AIR * Tb))
    return np.where(L == 0.0, pressure_isothermal, pressure_gradient)


def _build_base_pressures() -> NDArray[np.float64]:
    Pb = np.empty(len(_LAYERS))
    Pb[0] = P_SEA_LEVEL
    for i in range(1, len(_LAYERS)):
        Pb[i] = _layer_pressure(_HB[i], _HB[i - 1], _TB[i - 1], _LAPSE[i - 1], Pb[i - 1])
    return Pb


_PB: NDArray[np.float64] = _build_base_pressures()


def _unwrap(
    value: NDArray[np.float64] | np.float64,
) -> NDArray[np.float64] | np.float64:
    return np.asarray(value)[()]


def _as_altitude(h: ArrayLike) -> NDArray[np.float64]:
    h = np.asarray(h, dtype=float)
    if not np.all((h >= H_MIN) & (h <= H_MAX)):
        raise ValueError(
            f"altitude must be between {H_MIN:,.0f} and {H_MAX:,.0f} m (got values outside, or NaN)"
        )
    return h


def geopotential_altitude(h: ArrayLike) -> NDArray[np.float64]:
    """Convert geometric altitude to geopotential altitude for the ISA model."""
    altitude_m = _as_altitude(h)
    return R_EARTH * altitude_m / (R_EARTH + altitude_m)


def _temperature_pressure(
    geopotential_altitude_m: NDArray[np.float64] | np.float64,
) -> tuple[NDArray[np.float64] | np.float64, NDArray[np.float64] | np.float64]:
    layer_index = np.searchsorted(_HB, geopotential_altitude_m, side="right") - 1
    base_altitude_m = _HB[layer_index]
    base_temperature_K = _TB[layer_index]
    lapse_rate_K_m = _LAPSE[layer_index]
    base_pressure_Pa = _PB[layer_index]
    temperature_K = _layer_temperature(
        geopotential_altitude_m, base_altitude_m, base_temperature_K, lapse_rate_K_m
    )
    pressure_Pa = _layer_pressure(
        geopotential_altitude_m,
        base_altitude_m,
        base_temperature_K,
        lapse_rate_K_m,
        base_pressure_Pa,
    )
    return temperature_K, pressure_Pa


def _state(
    altitude_m: ArrayLike,
) -> tuple[NDArray[np.float64] | np.float64, NDArray[np.float64] | np.float64]:
    temperature_K, pressure_Pa = _temperature_pressure(geopotential_altitude(altitude_m))
    return _unwrap(temperature_K), _unwrap(pressure_Pa)


def density(h: ArrayLike) -> NDArray[np.float64] | np.float64:
    """Return ISA 1976 air density in kg/m^3 for geometric altitude in metres."""
    temperature_K, pressure_Pa = _state(h)
    return pressure_Pa / (R_AIR * temperature_K)


def speed_of_sound(h: ArrayLike) -> NDArray[np.float64] | np.float64:
    """Return the ideal-gas speed of sound in m/s for geometric altitude."""
    temperature_K, _ = _state(h)
    return np.sqrt(GAMMA * R_AIR * temperature_K)


class AtmosphereState(NamedTuple):
    """ISA atmospheric properties at one or more geometric altitudes."""

    altitude: NDArray[np.float64] | np.float64
    temperature: NDArray[np.float64] | np.float64
    pressure: NDArray[np.float64] | np.float64
    density: NDArray[np.float64] | np.float64
    speed_of_sound: NDArray[np.float64] | np.float64


def temperature(h: ArrayLike) -> NDArray[np.float64] | np.float64:
    """Return ISA 1976 air temperature in kelvin."""
    return _state(h)[0]


def pressure(h: ArrayLike) -> NDArray[np.float64] | np.float64:
    """Return ISA 1976 static pressure in pascals."""
    return _state(h)[1]


def standard_atmosphere(h: ArrayLike) -> AtmosphereState:
    """Return ISA 1976 temperature, pressure, density, and sound speed."""
    altitude_m = _as_altitude(h)
    temperature_K, pressure_Pa = _state(altitude_m)
    return AtmosphereState(
        _unwrap(altitude_m),
        temperature_K,
        pressure_Pa,
        density(altitude_m),
        speed_of_sound(altitude_m),
    )


def wind_velocity(
    h: ArrayLike,
    *,
    surface_wind: tuple[float, float] = (3.0, 1.0),
    jet_wind: tuple[float, float] = (40.0, 12.0),
    jet_altitude: float = 11_000.0,
    jet_width: float = 3_500.0,
    vertical_amplitude: float = 0.3,
    vertical_wavelength: float = 8_000.0,
) -> NDArray[np.float64]:
    """Return the configured east-north-up wind vector in m/s."""
    altitude_m = _as_altitude(h)
    jet_profile = np.exp(-0.5 * ((altitude_m - jet_altitude) / jet_width) ** 2)
    wind_east_m_s = surface_wind[0] + jet_wind[0] * jet_profile
    wind_north_m_s = surface_wind[1] + jet_wind[1] * jet_profile
    wind_up_m_s = vertical_amplitude * np.sin(2.0 * np.pi * altitude_m / vertical_wavelength)
    return np.stack([wind_east_m_s, wind_north_m_s, wind_up_m_s], axis=-1)
