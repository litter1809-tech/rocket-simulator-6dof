  #!/usr/bin/env python3
"""Atmosphere, vehicle, and 6-DoF rocket flight model."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation as R

G0 = 9.80665                  # standard gravity                         [m/s^2]
R_STAR = 8.31432              # universal gas constant                   [J/(mol K)]
M0 = 0.0289644                # mean molar mass of dry air at sea level  [kg/mol]
R_AIR = R_STAR / M0           # specific gas constant of air (~287.053)  [J/(kg K)]
GAMMA = 1.4                   # ratio of specific heats of air           [-]
R_EARTH = 6_356_766.0         # effective Earth radius (geopotential)    [m]
P_SEA_LEVEL = 101_325.0       # standard sea-level pressure              [Pa]

H_MIN, H_MAX = 0.0, 20_000.0  # supported geometric-altitude range       [m]

# One row per layer:
#   base geopotential altitude Hb [m], base temperature Tb [K], lapse rate L [K/m]
# (L = 0 means an isothermal layer.)
_LAYERS = np.array([
    [0.0,      288.15, -0.0065],   # troposphere
    [11_000.0, 216.65,  0.0000],   # lower stratosphere (isothermal)
])
# Add layers here if H_MAX is raised.
_HB, _TB, _LAPSE = _LAYERS.T


def _layer_temperature(H, Hb, Tb, L):
    return Tb + L * (H - Hb)


def _layer_pressure(H, Hb, Tb, L, Pb):
    T = _layer_temperature(H, Hb, Tb, L)
    safe_L = np.where(L == 0.0, 1.0, L)      # dummy value: the unused branch must not divide by 0
    gradient = Pb * (T / Tb) ** (-G0 / (R_AIR * safe_L))
    isothermal = Pb * np.exp(-G0 * (H - Hb) / (R_AIR * Tb))
    return np.where(L == 0.0, isothermal, gradient)


def _build_base_pressures() -> NDArray[np.float64]:
    Pb = np.empty(len(_LAYERS))
    Pb[0] = P_SEA_LEVEL
    for i in range(1, len(_LAYERS)):
        Pb[i] = _layer_pressure(_HB[i], _HB[i - 1], _TB[i - 1], _LAPSE[i - 1], Pb[i - 1])
    return Pb


_PB = _build_base_pressures()


def _unwrap(x):
    return np.asarray(x)[()]


def _as_altitude(h: ArrayLike) -> NDArray[np.float64]:
    h = np.asarray(h, dtype=float)
    if not np.all((h >= H_MIN) & (h <= H_MAX)):          # NaN fails this test too
        raise ValueError(
            f"altitude must be between {H_MIN:,.0f} and {H_MAX:,.0f} m (got values outside, or NaN)"
        )
    return h


def geopotential_altitude(h: ArrayLike) -> NDArray[np.float64]:
    """Convert geometric to geopotential altitude."""
    h = _as_altitude(h)
    return R_EARTH * h / (R_EARTH + h)


def _temperature_pressure(H):
    i = np.searchsorted(_HB, H, side="right") - 1        # layer index for each altitude
    Hb, Tb, L, Pb = _HB[i], _TB[i], _LAPSE[i], _PB[i]
    return _layer_temperature(H, Hb, Tb, L), _layer_pressure(H, Hb, Tb, L, Pb)


def _state(h: ArrayLike):
    T, P = _temperature_pressure(geopotential_altitude(h))
    return _unwrap(T), _unwrap(P)


def _density(P, T):
    return P / (R_AIR * T)


def _sound_speed(T):
    return np.sqrt(GAMMA * R_AIR * T)


class AtmosphereState(NamedTuple):
    """Atmosphere values at one or more altitudes."""
    altitude: NDArray[np.float64] | float         # geometric altitude    [m]
    temperature: NDArray[np.float64] | float      # T                     [K]
    pressure: NDArray[np.float64] | float         # P                     [Pa]
    density: NDArray[np.float64] | float          # rho                   [kg/m^3]
    speed_of_sound: NDArray[np.float64] | float   # a                     [m/s]


def temperature(h: ArrayLike):
    """Air temperature [K]."""
    return _state(h)[0]


def pressure(h: ArrayLike):
    """Static pressure [Pa]."""
    return _state(h)[1]


def density(h: ArrayLike):
    """Air density [kg/m^3]."""
    T, P = _state(h)
    return _density(P, T)


def speed_of_sound(h: ArrayLike):
    """Speed of sound [m/s]."""
    T, _ = _state(h)
    return _sound_speed(T)


def standard_atmosphere(h: ArrayLike) -> AtmosphereState:
    """Return temperature, pressure, density, and sound speed."""
    h = _as_altitude(h)
    T, P = _state(h)
    return AtmosphereState(_unwrap(h), T, P, _density(P, T), _sound_speed(T))


def wind_velocity(
    h: ArrayLike,
    *,
    surface_wind: tuple[float, float] = (3.0, 1.0),    # (X, Y) background wind [m/s]
    jet_wind: tuple[float, float] = (40.0, 12.0),      # (X, Y) extra wind at the jet core [m/s]
    jet_altitude: float = 11_000.0,                    # altitude of the jet core [m]
    jet_width: float = 3_500.0,                        # jet thickness (std. deviation) [m]
    vertical_amplitude: float = 0.3,                   # peak vertical wind [m/s]
    vertical_wavelength: float = 8_000.0,              # altitude period of vertical wind [m]
) -> NDArray[np.float64]:
    """Return the illustrative east-north-up wind profile [m/s]."""
    h = _as_altitude(h)
    g = np.exp(-0.5 * ((h - jet_altitude) / jet_width) ** 2)     # 0..1 bump around the jet core
    x = surface_wind[0] + jet_wind[0] * g
    y = surface_wind[1] + jet_wind[1] * g
    z = vertical_amplitude * np.sin(2.0 * np.pi * h / vertical_wavelength)
    return np.stack([x, y, z], axis=-1)


@dataclass
class Rocket:
    """Single-stage rocket with constant thrust and linear propellant burn."""

    dry_mass: float
    propellant_mass: float
    burn_time: float
    average_thrust: float
    reference_area: float
    drag_coefficient: float
    moment_of_inertia: float

    # Time-dependent state: set by update(), not constructor arguments.
    time: float = field(default=0.0, init=False)
    mass: float = field(init=False)
    thrust: float = field(init=False)

    def __post_init__(self) -> None:
        for name in ("dry_mass", "burn_time", "reference_area", "moment_of_inertia"):
            value = getattr(self, name)
            if not (np.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be a positive number (got {value!r})")
        for name in ("propellant_mass", "average_thrust", "drag_coefficient"):
            value = getattr(self, name)
            if not (np.isfinite(value) and value >= 0.0):
                raise ValueError(f"{name} must be a non-negative number (got {value!r})")
        self.update(0.0)                          # start at ignition: full mass, full thrust

    def update(self, t: float) -> tuple[float, float]:
        """Refresh mass and thrust for elapsed flight time t."""
        t = float(t)
        if not t >= 0.0:                          # also rejects NaN
            raise ValueError(f"flight time must be >= 0 (got {t!r})")
        burned = min(t / self.burn_time, 1.0)     # propellant used so far: 0 at ignition -> 1 at burnout
        self.time = t
        self.mass = self.dry_mass + self.propellant_mass * (1.0 - burned)
        self.thrust = float(self.average_thrust) if t < self.burn_time else 0.0
        return self.mass, self.thrust

    def inertia_matrix(self) -> NDArray[np.float64]:
        wet_mass = self.dry_mass + self.propellant_mass
        scale = self.mass / wet_mass if wet_mass > 0.0 else 1.0
        return np.eye(3, dtype=float) * self.moment_of_inertia * scale


def _normalize_quaternion(q: ArrayLike) -> NDArray[np.float64]:
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if norm == 0.0:
        raise ValueError("quaternion norm cannot be zero")
    return q / norm


def _quat_conjugate(q: ArrayLike) -> NDArray[np.float64]:
    q = np.asarray(q, dtype=float)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def _quat_multiply(q1: ArrayLike, q2: ArrayLike) -> NDArray[np.float64]:
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    return np.array([
        q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
        q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2],
        q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1],
        q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0],
    ], dtype=float)


def _quat_to_rotation_matrix(q: ArrayLike) -> NDArray[np.float64]:
    q = _normalize_quaternion(q)
    w, x, y, z = q
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=float)


def _quat_derivative(q: ArrayLike, w_body: ArrayLike) -> NDArray[np.float64]:
    q = _normalize_quaternion(q)
    w_body = np.asarray(w_body, dtype=float)
    q_w = np.array([0.0, *w_body], dtype=float)
    return 0.5 * _quat_multiply(q, q_w)


def _aero_forces_and_moments(
    rocket: Rocket,
    position: NDArray[np.float64],
    velocity_inertial: NDArray[np.float64],
    q: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float, float]:
    altitude = max(float(position[2]), 0.0)
    rho = density(altitude)
    a = speed_of_sound(altitude)
    wind = np.asarray(wind_velocity(altitude), dtype=float)
    rel = velocity_inertial - wind
    rel_speed = np.linalg.norm(rel)
    if rel_speed < 1e-12:
        return np.zeros(3, dtype=float), np.zeros(3, dtype=float), 0.0, 0.0, rho

    R_b2i = _quat_to_rotation_matrix(q)
    rel_body = R_b2i.T @ rel
    alpha = float(np.arctan2(rel_body[2], rel_body[0]))
    beta = float(np.arctan2(rel_body[1], rel_body[0]))
    q_inf = 0.5 * rho * rel_speed**2
    S = rocket.reference_area
    C_D = 0.5 + 2.0 * alpha**2
    C_L = 2.0 * np.pi * alpha

    u_hat = rel_body / rel_speed
    drag_body = -q_inf * S * C_D * u_hat
    lift_dir = np.array([0.0, 0.0, 1.0], dtype=float)
    lift_body = -q_inf * S * C_L * lift_dir
    F_aero_body = drag_body + lift_body
    F_aero_inertial = R_b2i @ F_aero_body

    # Fixed moment arm and simple coefficients; swap in vehicle aero data when available.
    moment_arm = 0.15
    moment_body = np.array([
        0.0,
        -q_inf * S * moment_arm * C_L,
        0.0,
    ], dtype=float)
    return F_aero_inertial, moment_body, alpha, beta, rho


def rocket_state_derivative(
    t: float,
    y: ArrayLike,
    rocket: Rocket,
) -> NDArray[np.float64]:
    """Return the derivative of the 13-state 6-DoF model."""
    y = np.asarray(y, dtype=float)
    if y.shape[0] != 13:
        raise ValueError(f"state vector must have length 13, got {y.shape[0]}")

    position = y[0:3]
    velocity = y[3:6]
    q = _normalize_quaternion(y[6:10])
    w_body = y[10:13]

    rocket.update(t)
    mass = rocket.mass
    I_matrix = rocket.inertia_matrix()

    grav = np.array([0.0, 0.0, -G0 * mass], dtype=float)
    R_b2i = _quat_to_rotation_matrix(q)
    F_thrust_body = np.array([0.0, 0.0, rocket.thrust], dtype=float)
    F_thrust_inertial = R_b2i @ F_thrust_body

    F_aero_inertial, M_aero_body, alpha, beta, rho = _aero_forces_and_moments(
        rocket, position, velocity, q
    )
    total_force_inertial = grav + F_thrust_inertial + F_aero_inertial

    acceleration = total_force_inertial / mass
    angular_momentum = I_matrix @ w_body
    rotational_accel = np.linalg.solve(I_matrix, M_aero_body - np.cross(w_body, angular_momentum))

    dstate = np.empty(13, dtype=float)
    dstate[0:3] = velocity
    dstate[3:6] = acceleration
    dstate[6:10] = _quat_derivative(q, w_body)
    dstate[10:13] = rotational_accel
    return dstate


def ground_impact(t: float, y: ArrayLike) -> float:
    """Altitude event used to stop integration at ground contact."""
    z = float(np.asarray(y, dtype=float)[2])
    return 1.0 if t <= 0.0 else z


ground_impact.terminal = True
ground_impact.direction = -1


def simulate_rocket(
    rocket: Rocket,
    initial_state: ArrayLike | None = None,
    t_span: tuple[float, float] = (0.0, 30.0),
) -> object:
    """Integrate a 6-DoF trajectory with RK45 until ground impact."""
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

    sol = solve_ivp(
        fun=lambda t, y: rocket_state_derivative(t, y, rocket),
        t_span=t_span,
        y0=y0,
        method='RK45',
        rtol=1e-6,
        atol=1e-8,
        events=ground_impact,
        dense_output=True,
    )
    return sol


def _flight_metrics(sol: object, rocket: Rocket) -> dict[str, float]:
    if not hasattr(sol, 'y') or sol.y.size == 0:
        raise ValueError("simulation produced no trajectory data")

    t = np.asarray(sol.t, dtype=float)
    x = np.asarray(sol.y[0], dtype=float)
    y = np.asarray(sol.y[1], dtype=float)
    z = np.asarray(sol.y[2], dtype=float)
    vx = np.asarray(sol.y[3], dtype=float)
    vy = np.asarray(sol.y[4], dtype=float)
    vz = np.asarray(sol.y[5], dtype=float)

    altitude = np.maximum(z, 0.0)
    speed = np.sqrt(vx**2 + vy**2 + vz**2)
    mach = np.zeros_like(speed)
    for i in range(len(t)):
        alt_i = float(np.clip(altitude[i], 0.0, H_MAX))
        sound_speed = speed_of_sound(alt_i)
        mach[i] = speed[i] / sound_speed if sound_speed > 0.0 else 0.0

    q_inf = np.zeros_like(speed)
    for i in range(len(t)):
        alt_i = float(np.clip(altitude[i], 0.0, H_MAX))
        rho = density(alt_i)
        q_inf[i] = 0.5 * rho * speed[i]**2

    metrics = {
        "apogee_m": float(np.max(altitude)),
        "apogee_ft": float(np.max(altitude) * 3.280839895),
        "max_velocity_m_s": float(np.max(speed)),
        "max_mach": float(np.max(mach)),
        "burnout_time_s": float(rocket.burn_time),
        "flight_duration_s": float(t[-1]),
        "impact_downrange_m": float(np.hypot(x[-1], y[-1])),
        "max_dynamic_pressure_pa": float(np.max(q_inf)),
    }
    return metrics


def _plot_performance_dashboard(sol: object, rocket: Rocket) -> None:
    t = np.asarray(sol.t, dtype=float)
    x = np.asarray(sol.y[0], dtype=float)
    y = np.asarray(sol.y[1], dtype=float)
    z = np.asarray(sol.y[2], dtype=float)
    vx = np.asarray(sol.y[3], dtype=float)
    vy = np.asarray(sol.y[4], dtype=float)
    vz = np.asarray(sol.y[5], dtype=float)
    speed = np.sqrt(vx**2 + vy**2 + vz**2)

    alt = np.maximum(z, 0.0)
    q_inf = np.zeros_like(speed)
    mach = np.zeros_like(speed)
    thrust_hist = np.zeros_like(t)
    mass_hist = np.zeros_like(t)

    for i, ti in enumerate(t):
        rocket.update(ti)
        thrust_hist[i] = rocket.thrust
        mass_hist[i] = rocket.mass
        alt_i = float(np.clip(alt[i], 0.0, H_MAX))
        rho = density(alt_i)
        q_inf[i] = 0.5 * rho * speed[i]**2
        sound = speed_of_sound(alt_i)
        mach[i] = speed[i] / sound if sound > 0.0 else 0.0

    fig = plt.figure(figsize=(12, 10))
    fig.suptitle("Rocket Performance Dashboard", fontsize=16, fontweight="bold")

    ax = fig.add_subplot(2, 2, 1, projection="3d")
    ax.plot(x, y, z, color="tab:blue", linewidth=2)
    ax.scatter([x[0]], [y[0]], [z[0]], color="green", s=40, label="launch")
    ax.scatter([x[-1]], [y[-1]], [z[-1]], color="red", s=50, label="impact")
    ax.set_title("3D Trajectory")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2_twin = ax2.twinx()
    ax2.plot(t, alt, color="tab:blue", linewidth=2, label="Altitude")
    ax2_twin.plot(t, vz, color="tab:orange", linestyle="--", linewidth=2, label="Vertical velocity")
    ax2.set_title("Altitude & Vertical Velocity")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Altitude [m]", color="tab:blue")
    ax2_twin.set_ylabel("Vertical velocity [m/s]", color="tab:orange")
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(2, 2, 3)
    ax3_twin = ax3.twinx()
    ax3.plot(t, thrust_hist, color="tab:green", linewidth=2, label="Thrust")
    ax3_twin.plot(t, mass_hist, color="tab:red", linestyle="--", linewidth=2, label="Mass")
    ax3.set_title("Thrust & Vehicle Mass")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Thrust [N]", color="tab:green")
    ax3_twin.set_ylabel("Mass [kg]", color="tab:red")
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4_twin = ax4.twinx()
    ax4.plot(t, q_inf, color="tab:purple", linewidth=2, label="Dynamic pressure")
    ax4_twin.plot(t, mach, color="tab:brown", linestyle="--", linewidth=2, label="Mach")
    ax4.set_title("Dynamic Pressure & Mach Number")
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Dynamic pressure [Pa]", color="tab:purple")
    ax4_twin.set_ylabel("Mach number", color="tab:brown")
    ax4.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_path = "rocket_performance_dashboard.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved performance dashboard to {output_path}")


def _self_check() -> None:
    # Layer-boundary values at geopotential altitude H: (H [m], T [K], P [Pa]).
    for H, T_ref, P_ref in [
        (0.0, 288.15, 101_325.0),
        (11_000.0, 216.65, 22_632.06),
        (20_000.0, 216.65, 5_474.889),
    ]:
        T, P = _temperature_pressure(np.float64(H))
        np.testing.assert_allclose([T, P], [T_ref, P_ref], rtol=1e-5)

    sea = standard_atmosphere(0.0)
    np.testing.assert_allclose(
        [sea.density, sea.speed_of_sound], [1.2250, 340.294], rtol=1e-4
    )

    # Table entry at *geometric* 20 km: P = 5529.3 Pa, rho = 0.08891 kg/m^3.
    top = standard_atmosphere(20_000.0)
    np.testing.assert_allclose([top.pressure, top.density], [5529.3, 0.08891], rtol=1e-4)


def _self_check_rocket() -> None:
    r = Rocket(dry_mass=10.0, propellant_mass=4.0, burn_time=2.0, average_thrust=500.0,
               reference_area=0.01, drag_coefficient=0.4, moment_of_inertia=2.0)
    np.testing.assert_allclose([r.mass, r.thrust], [14.0, 500.0], rtol=1e-12)   # at ignition
    np.testing.assert_allclose(r.update(1.0), [12.0, 500.0], rtol=1e-12)        # half the propellant gone
    np.testing.assert_allclose(r.update(2.0), [10.0, 0.0], rtol=1e-12)          # burn time expired
    np.testing.assert_allclose(r.update(60.0), [10.0, 0.0], rtol=1e-12)         # stays burnt out
    try:
        r.update(-1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("negative flight time should raise ValueError")


def main() -> None:
    _self_check()
    _self_check_rocket()
    print("Self-checks passed: atmosphere reference values and Rocket mass/thrust model\n")

    h = np.array([0, 1_000, 2_000, 5_000, 8_000, 11_000, 15_000, 20_000], dtype=float)
    atm = standard_atmosphere(h)
    wind = wind_velocity(h)                      # shape (8, 3): columns are X, Y, Z
    wind = np.round(wind, 2) + 0.0               # display only: avoids "-0.00" from float noise

    header = (
        f"{'h [m]':>8}{'T [K]':>9}{'P [Pa]':>11}{'rho [kg/m^3]':>14}{'a [m/s]':>10}"
        f"{'Wx':>8}{'Wy':>8}{'Wz':>8}"
    )
    print(header)
    print("-" * len(header))
    for k in range(len(h)):
        print(
            f"{atm.altitude[k]:8.0f}{atm.temperature[k]:9.2f}{atm.pressure[k]:11.1f}"
            f"{atm.density[k]:14.5f}{atm.speed_of_sound[k]:10.2f}"
            f"{wind[k, 0]:8.2f}{wind[k, 1]:8.2f}{wind[k, 2]:8.2f}"
        )
    print("\nWind components in m/s: X = east, Y = north, Z = up.")

    # Example vehicle (placeholder numbers): 12 kg dry + 3 kg propellant, 1200 N for 5 s.
    rocket = Rocket(
        dry_mass=12.0,
        propellant_mass=3.0,
        burn_time=5.0,
        average_thrust=1200.0,
        reference_area=np.pi * 0.05**2,          # 10 cm diameter body [m^2]
        drag_coefficient=0.5,
        moment_of_inertia=8.0,                   # [kg m^2]
    )
    print("\nExample rocket, mass and thrust vs flight time:")
    print(f"{'t [s]':>8}{'mass [kg]':>12}{'thrust [N]':>12}")
    for t in (0.0, 1.0, 2.5, 4.9, 5.0, 6.0):
        mass, thrust = rocket.update(t)
        print(f"{t:8.1f}{mass:12.3f}{thrust:12.1f}")

    # Example 6-DoF launch integration using RK45.
    launch_rocket = Rocket(
        dry_mass=12.0,
        propellant_mass=3.0,
        burn_time=5.0,
        average_thrust=1200.0,
        reference_area=np.pi * 0.05**2,
        drag_coefficient=0.5,
        moment_of_inertia=8.0,
    )
    y0 = np.array([
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    ], dtype=float)
    launch = simulate_rocket(launch_rocket, initial_state=y0, t_span=(0.0, 20.0))
    metrics = _flight_metrics(launch, launch_rocket)

    print("\n6-DoF flight integration: RK45 solution")
    print(f"status={launch.status}; events={len(launch.t_events[0])}; final time={launch.t[-1]:.2f} s")
    if launch.t.size:
        x, y, z = launch.y[0:3, -1]
        print(f"final position (x, y, z) = ({x:.2f}, {y:.2f}, {z:.2f}) m")

    print("\nPerformance metrics:")
    print(f"  Apogee altitude     : {metrics['apogee_m']:.1f} m ({metrics['apogee_ft']:.1f} ft)")
    print(f"  Max velocity        : {metrics['max_velocity_m_s']:.1f} m/s ({metrics['max_velocity_m_s'] / 343.0:.2f} Mach)")
    print(f"  Burnout time        : {metrics['burnout_time_s']:.2f} s")
    print(f"  Flight duration     : {metrics['flight_duration_s']:.2f} s")
    print(f"  Downrange impact    : {metrics['impact_downrange_m']:.1f} m")

    _plot_performance_dashboard(launch, launch_rocket)


if __name__ == "__main__":
    main()