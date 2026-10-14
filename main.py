from __future__ import annotations

import numpy as np
from scipy.optimize import OptimizeResult

from src.dashboard import generate_dashboard
from src.environment import R_EARTH, density, speed_of_sound, standard_atmosphere, wind_velocity
from src.physics import simulate_rocket
from src.vehicle import Rocket


def _self_check_atmosphere() -> None:
    for geopotential_altitude_m, temperature_ref_K, pressure_ref_Pa in [
        (0.0, 288.15, 101_325.0),
        (11_000.0, 216.65, 22_632.06),
    ]:
        geometric_altitude_m = (
            R_EARTH * geopotential_altitude_m / (R_EARTH - geopotential_altitude_m)
        )
        atmosphere = standard_atmosphere(geometric_altitude_m)
        np.testing.assert_allclose(
            [atmosphere.temperature, atmosphere.pressure],
            [temperature_ref_K, pressure_ref_Pa],
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
    time_s = np.asarray(solution.t, dtype=float)
    position_x_m = np.asarray(solution.y[0], dtype=float)
    position_y_m = np.asarray(solution.y[1], dtype=float)
    position_z_m = np.asarray(solution.y[2], dtype=float)
    velocity_x_m_s = np.asarray(solution.y[3], dtype=float)
    velocity_y_m_s = np.asarray(solution.y[4], dtype=float)
    velocity_z_m_s = np.asarray(solution.y[5], dtype=float)

    altitude_m = np.maximum(position_z_m, 0.0)
    speed_m_s = np.sqrt(velocity_x_m_s**2 + velocity_y_m_s**2 + velocity_z_m_s**2)
    mach_number = np.zeros_like(speed_m_s)
    q_inf = np.zeros_like(speed_m_s)
    for sample_index in range(len(time_s)):
        sample_altitude_m = float(np.clip(altitude_m[sample_index], 0.0, 20_000.0))
        rho = density(sample_altitude_m)
        sound_speed_m_s = speed_of_sound(sample_altitude_m)
        q_inf[sample_index] = 0.5 * rho * speed_m_s[sample_index] ** 2
        mach_number[sample_index] = (
            speed_m_s[sample_index] / sound_speed_m_s if sound_speed_m_s > 0.0 else 0.0
        )

    return {
        "apogee_m": float(np.max(altitude_m)),
        "apogee_ft": float(np.max(altitude_m) * 3.280839895),
        "max_velocity_m_s": float(np.max(speed_m_s)),
        "max_mach": float(np.max(mach_number)),
        "burnout_time_s": float(rocket.burn_time),
        "flight_duration_s": float(time_s[-1]),
        "impact_downrange_m": float(np.hypot(position_x_m[-1], position_y_m[-1])),
        "max_dynamic_pressure_pa": float(np.max(q_inf)),
    }


def run() -> None:
    _self_check_atmosphere()
    _self_check_rocket()
    print("Self-checks passed: atmosphere reference values and Rocket mass/thrust model\n")

    altitude_samples_m = np.array(
        [0, 1_000, 2_000, 5_000, 8_000, 11_000, 15_000, 20_000], dtype=float
    )
    atmosphere = standard_atmosphere(altitude_samples_m)
    wind_m_s = np.round(wind_velocity(altitude_samples_m), 2) + 0.0

    header = (
        f"{'h [m]':>8}{'T [K]':>9}{'P [Pa]':>11}{'rho [kg/m^3]':>14}{'a [m/s]':>10}"
        f"{'Wx':>8}{'Wy':>8}{'Wz':>8}"
    )
    print(header)
    print("-" * len(header))
    for sample_index in range(len(altitude_samples_m)):
        print(
            f"{atmosphere.altitude[sample_index]:8.0f}"
            f"{atmosphere.temperature[sample_index]:9.2f}"
            f"{atmosphere.pressure[sample_index]:11.1f}"
            f"{atmosphere.density[sample_index]:14.5f}"
            f"{atmosphere.speed_of_sound[sample_index]:10.2f}"
            f"{wind_m_s[sample_index, 0]:8.2f}"
            f"{wind_m_s[sample_index, 1]:8.2f}"
            f"{wind_m_s[sample_index, 2]:8.2f}"
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
    for time_s in (0.0, 1.0, 2.5, 4.9, 5.0, 6.0):
        mass_kg, thrust_N = rocket.update(time_s)
        print(f"{time_s:8.1f}{mass_kg:12.3f}{thrust_N:12.1f}")

    launch_rocket = Rocket(
        dry_mass=12.0,
        propellant_mass=3.0,
        burn_time=5.0,
        average_thrust=1200.0,
        reference_area=np.pi * 0.05**2,
        drag_coefficient=0.5,
        moment_of_inertia=8.0,
    )
    initial_state = np.array([
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    ], dtype=float)
    solution = simulate_rocket(
        launch_rocket,
        initial_state=initial_state,
        t_span=(0.0, 20.0),
    )

    print("\n6-DoF flight integration: RK45 solution")
    print(
        f"status={solution.status}; events={len(solution.t_events[0])}; "
        f"final time={solution.t[-1]:.2f} s"
    )
    if solution.t.size:
        final_position_m = solution.y[0:3, -1]
        print(
            "final position (x, y, z) = "
            f"({final_position_m[0]:.2f}, {final_position_m[1]:.2f}, "
            f"{final_position_m[2]:.2f}) m"
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
