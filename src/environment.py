"""
1976 US Standard Atmosphere and wind environment models.
Author: Tegan Broody
School: Ridley High School
Date: October 2026
"""

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
# Extend this table and H_MAX together if the model needs to cover higher altitudes.
_HB, _TB, _LAPSE = _LAYERS.T


def _layer_temperature(
    H: NDArray[np.float64] | np.float64,
    Hb: NDArray[np.float64] | np.float64,
    Tb: NDArray[np.float64] | np.float64,
    L: NDArray[np.float64] | np.float64,
) -> NDArray[np.float64] | np.float64:
    return Tb + L * (H - Hb)


def _layer_pressure(
    H: NDArray[np.float64] | np.float64,
    Hb: NDArray[np.float64] | np.float64,
    Tb: NDArray[np.float64] | np.float64,
    L: NDArray[np.float64] | np.float64,
    Pb: NDArray[np.float64] | np.float64,
) -> NDArray[np.float64] | np.float64:
    temp = _layer_temperature(H, Hb, Tb, L)
    safe_lapse = np.where(L == 0.0, 1.0, L)
    p_gradient = Pb * (temp / Tb) ** (-G0 / (R_AIR * safe_lapse))
    p_isothermal = Pb * np.exp(-G0 * (H - Hb) / (R_AIR * Tb))
    return np.where(L == 0.0, p_isothermal, p_gradient)


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
    """Convert geometric altitude to geopotential altitude."""
    alt = _as_altitude(h)
    return R_EARTH * alt / (R_EARTH + alt)


def _temperature_pressure(
    geopotential_altitude_m: NDArray[np.float64] | np.float64,
) -> tuple[NDArray[np.float64] | np.float64, NDArray[np.float64] | np.float64]:
    layer_index = np.searchsorted(_HB, geopotential_altitude_m, side="right") - 1
    base_alt = _HB[layer_index]
    base_temp = _TB[layer_index]
    lapse = _LAPSE[layer_index]
    base_pressure = _PB[layer_index]
    temp = _layer_temperature(
        geopotential_altitude_m, base_alt, base_temp, lapse
    )
    pressure = _layer_pressure(
        geopotential_altitude_m,
        base_alt,
        base_temp,
        lapse,
        base_pressure,
    )
    return temp, pressure


def _state(
    altitude_m: ArrayLike,
) -> tuple[NDArray[np.float64] | np.float64, NDArray[np.float64] | np.float64]:
    temp, pressure = _temperature_pressure(geopotential_altitude(altitude_m))
    return _unwrap(temp), _unwrap(pressure)


def density(h: ArrayLike) -> NDArray[np.float64] | np.float64:
    """Air density [kg/m^3] at geometric altitude in metres."""
    temp, pressure = _state(h)
    return pressure / (R_AIR * temp)


def speed_of_sound(h: ArrayLike) -> NDArray[np.float64] | np.float64:
    """Speed of sound [m/s] at geometric altitude."""
    temp, _ = _state(h)
    return np.sqrt(GAMMA * R_AIR * temp)


class AtmosphereState(NamedTuple):
    """Atmospheric properties at one or more geometric altitudes."""

    altitude: NDArray[np.float64] | np.float64
    temperature: NDArray[np.float64] | np.float64
    pressure: NDArray[np.float64] | np.float64
    density: NDArray[np.float64] | np.float64
    speed_of_sound: NDArray[np.float64] | np.float64


def temperature(h: ArrayLike) -> NDArray[np.float64] | np.float64:
    """Air temperature [K]."""
    return _state(h)[0]


def pressure(h: ArrayLike) -> NDArray[np.float64] | np.float64:
    """Static pressure [Pa]."""
    return _state(h)[1]


def standard_atmosphere(h: ArrayLike) -> AtmosphereState:
    """Return temperature, pressure, density, and sound speed."""
    altitude_m = _as_altitude(h)
    temp, pressure = _state(altitude_m)
    return AtmosphereState(
        _unwrap(altitude_m),
        temp,
        pressure,
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
    enabled: bool = True,
) -> NDArray[np.float64]:
    
    alt = _as_altitude(h)

    if not enabled:
        return np.zeros((*alt.shape, 3), dtype=float)

    if jet_width <= 0.0:
        raise ValueError("jet_width must be positive")
    if vertical_wavelength <= 0.0:
        raise ValueError("vertical_wavelength must be positive")

    jet = np.exp(
        -0.5 * ((alt - jet_altitude) / jet_width) ** 2
    )

    east = surface_wind[0] + jet_wind[0] * jet
    north = surface_wind[1] + jet_wind[1] * jet
    up = vertical_amplitude * np.sin(
        2.0 * np.pi * alt / vertical_wavelength
    )

    return np.stack([east, north, up], axis=-1)
