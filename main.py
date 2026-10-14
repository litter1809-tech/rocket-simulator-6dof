from __future__ import annotations

import numpy as np
from scipy.optimize import OptimizeResult

from src.dashboard import generate_dashboard
from src.environment import R_EARTH, density, speed_of_sound, standard_atmosphere, wind_velocity
from src.physics import simulate_rocket
from src.vehicle import Rocket


def _self_check_atmosphere() -> None:
    for alt_h, temp_ref, pressure_ref in [
        (0.0, 288.15, 101_325.0),
        (11_000.0, 216.65, 22_632.06),
    ]:
        alt = R_EARTH * alt_h / (R_EARTH - alt_h)
        atmosphere = standard_atmosphere(alt)
        np.testing.assert_allclose(
            [atmosphere.temperature, atmosphere.pressure],
            [temp_ref, pressure_ref],
            rtol=1e-5,
        )

    sea = standard_atmosphere(0.0)
    np.testing.assert_allclose([sea.density, sea.speed_of_sound], [1.2250, 340.294], rtol=1e-4)

    top = standard_atmosphere(20_000.0)
    np.testing.assert_allclose([top.pressure, top.density], [5529.3, 0.08891], rtol=1e-4)


def _self_check_rocket() -> None:
    rocket = Rocket(
        dry_mass=10.0,
        propellant_mass=4.0,
        burn_time=2.0,
        average_thrust=500.0,
        reference_area=0.01,
        drag_coefficient=0.4,
        moment_of_inertia=2.0,
    )
    np.testing.assert_allclose([rocket.mass, rocket.thrust], [14.0, 500.0], rtol=1e-12)
    np.testing.assert_allclose(rocket.update(1.0), [12.0, 500.0], rtol=1e-12)
    np.testing.assert_allclose(rocket.update(2.0), [10.0, 0.0], rtol=1e-12)
    np.testing.assert_allclose(rocket.update(60.0), [10.0, 0.0], rtol=1e-12)
    try:
        rocket.update(-1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("negative flight time should raise ValueError")


def _flight_metrics(solution: OptimizeResult, rocket: Rocket) -> dict[str, float]:
    time = np.asarray(solution.t, dtype=float)
    x, y, z = (np.asarray(solution.y[i], dtype=float) for i in range(3))
    vx, vy, vz = (np.asarray(solution.y[i], dtype=float) for i in range(3, 6))

    altitude = np.maximum(z, 0.0)
    speed = np.sqrt(vx**2 + vy**2 + vz**2)
    mach = np.zeros_like(speed)
    qbar = np.zeros_like(speed)
    for i in range(len(time)):
        alt = float(np.clip(altitude[i], 0.0, 20_000.0))
        rho = density(alt)
        sound = speed_of_sound(alt)
        qbar[i] = 0.5 * rho * speed[i] ** 2
        mach[i] = speed[i] / sound if sound > 0.0 else 0.0

    return {
        "apogee_m": float(np.max(altitude)),
        "apogee_ft": float(np.max(altitude) * 3.280839895),
        "max_velocity_m_s": float(np.max(speed)),
        "max_mach": float(np.max(mach)),
        "burnout_time_s": float(rocket.burn_time),
        "flight_duration_s": float(time[-1]),
        "impact_downrange_m": float(np.hypot(x[-1], y[-1])),
        "max_dynamic_pressure_pa": float(np.max(qbar)),
    }


def run() -> None:
    _self_check_atmosphere()
    _self_check_rocket()
    print("Self-checks passed: atmosphere reference values and Rocket mass/thrust model\n")

    altitudes = np.array(
        [0, 1_000, 2_000, 5_000, 8_000, 11_000, 15_000, 20_000], dtype=float
    )
    atmosphere = standard_atmosphere(altitudes)
    wind = np.round(wind_velocity(altitudes), 2) + 0.0

    headings = (
        f"{'h [m]':>8}{'T [K]':>9}{'P [Pa]':>11}{'rho [kg/m^3]':>14}{'a [m/s]':>10}"
        f"{'Wx':>8}{'Wy':>8}{'Wz':>8}"
    )
    print(headings)
    print("-" * len(headings))
    for i in range(len(altitudes)):
        print(
            f"{atmosphere.altitude[i]:8.0f}"
            f"{atmosphere.temperature[i]:9.2f}"
            f"{atmosphere.pressure[i]:11.1f}"
            f"{atmosphere.density[i]:14.5f}"
            f"{atmosphere.speed_of_sound[i]:10.2f}"
            f"{wind[i, 0]:8.2f}"
            f"{wind[i, 1]:8.2f}"
            f"{wind[i, 2]:8.2f}"
        )
    print("\nWind components in m/s: X = east, Y = north, Z = up.")

    rocket = Rocket(
        dry_mass=12.0,
        propellant_mass=3.0,
        burn_time=5.0,
        average_thrust=1200.0,
        reference_area=np.pi * 0.05**2,
        drag_coefficient=0.5,
        moment_of_inertia=8.0,
    )
    print("\nExample rocket, mass and thrust vs flight time:")
    print(f"{'t [s]':>8}{'mass [kg]':>12}{'thrust [N]':>12}")
    for time in (0.0, 1.0, 2.5, 4.9, 5.0, 6.0):
        mass, thrust = rocket.update(time)
        print(f"{time:8.1f}{mass:12.3f}{thrust:12.1f}")

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
    solution = simulate_rocket(
        launch_rocket,
        initial_state=y0,
        t_span=(0.0, 20.0),
    )

    print("\n6-DoF flight integration: RK45 solution")
    print(
        f"status={solution.status}; events={len(solution.t_events[0])}; "
        f"final time={solution.t[-1]:.2f} s"
    )
    if solution.t.size:
        final_pos = solution.y[0:3, -1]
        print(
            "final position (x, y, z) = "
            f"({final_pos[0]:.2f}, {final_pos[1]:.2f}, "
            f"{final_pos[2]:.2f}) m"
        )

    metrics = _flight_metrics(solution, launch_rocket)
    print("\nPerformance metrics:")
    print(f"  Apogee altitude     : {metrics['apogee_m']:.1f} m ({metrics['apogee_ft']:.1f} ft)")
    print(f"  Max velocity        : {metrics['max_velocity_m_s']:.1f} m/s ({metrics['max_velocity_m_s'] / 343.0:.2f} Mach)")
    print(f"  Burnout time        : {metrics['burnout_time_s']:.2f} s")
    print(f"  Flight duration     : {metrics['flight_duration_s']:.2f} s")
    print(f"  Downrange impact    : {metrics['impact_downrange_m']:.1f} m")

    dashboard_path = generate_dashboard(solution, launch_rocket)
    print(f"\nSaved performance dashboard to {dashboard_path}")


if __name__ == "__main__":
    run()
