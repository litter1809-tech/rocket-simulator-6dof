from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import OptimizeResult

from src.environment import H_MAX, density, speed_of_sound, wind_velocity
from src.vehicle import Rocket


def generate_dashboard(
    sol: OptimizeResult,
    rocket: Rocket,
    output_path: str = "rocket_performance_dashboard.png",
) -> str:
    """Save trajectory and flight-load plots for an integrated rocket solution."""
    time_s = np.asarray(sol.t, dtype=float)
    position_x_m = np.asarray(sol.y[0], dtype=float)
    position_y_m = np.asarray(sol.y[1], dtype=float)
    position_z_m = np.asarray(sol.y[2], dtype=float)
    velocity_x_m_s = np.asarray(sol.y[3], dtype=float)
    velocity_y_m_s = np.asarray(sol.y[4], dtype=float)
    velocity_z_m_s = np.asarray(sol.y[5], dtype=float)
    speed_m_s = np.sqrt(velocity_x_m_s**2 + velocity_y_m_s**2 + velocity_z_m_s**2)

    altitude_m = np.maximum(position_z_m, 0.0)
    q_inf = np.zeros_like(speed_m_s)
    mach_number = np.zeros_like(speed_m_s)
    thrust_history_N = np.zeros_like(time_s)
    mass_history_kg = np.zeros_like(time_s)

    for sample_index, sample_time_s in enumerate(time_s):
        rocket.update(sample_time_s)
        thrust_history_N[sample_index] = rocket.thrust
        mass_history_kg[sample_index] = rocket.mass
        sample_altitude_m = float(np.clip(altitude_m[sample_index], 0.0, H_MAX))
        rho = density(sample_altitude_m)
        q_inf[sample_index] = 0.5 * rho * speed_m_s[sample_index] ** 2
        sound_speed_m_s = speed_of_sound(sample_altitude_m)
        mach_number[sample_index] = (
            speed_m_s[sample_index] / sound_speed_m_s if sound_speed_m_s > 0.0 else 0.0
        )

    fig = plt.figure(figsize=(12, 10))
    fig.suptitle("Rocket Performance Dashboard", fontsize=16, fontweight="bold")

    ax_trajectory = fig.add_subplot(2, 2, 1, projection="3d")
    ax_trajectory.plot(position_x_m, position_y_m, position_z_m, color="tab:blue", linewidth=2)
    ax_trajectory.scatter(
        [position_x_m[0]], [position_y_m[0]], [position_z_m[0]], color="green", s=40, label="launch"
    )
    ax_trajectory.scatter(
        [position_x_m[-1]], [position_y_m[-1]], [position_z_m[-1]], color="red", s=50, label="impact"
    )
    ax_trajectory.set_title("3D Trajectory")
    ax_trajectory.set_xlabel("X [m]")
    ax_trajectory.set_ylabel("Y [m]")
    ax_trajectory.set_zlabel("Z [m]")
    ax_trajectory.grid(True, alpha=0.3)

    ax_altitude = fig.add_subplot(2, 2, 2)
    ax_altitude_twin = ax_altitude.twinx()
    ax_altitude.plot(time_s, altitude_m, color="tab:blue", linewidth=2, label="Altitude")
    ax_altitude_twin.plot(
        time_s, velocity_z_m_s, color="tab:orange", linestyle="--", linewidth=2, label="Vertical velocity"
    )
    ax_altitude.set_title("Altitude & Vertical Velocity")
    ax_altitude.set_xlabel("Time [s]")
    ax_altitude.set_ylabel("Altitude [m]", color="tab:blue")
    ax_altitude_twin.set_ylabel("Vertical velocity [m/s]", color="tab:orange")
    ax_altitude.grid(True, alpha=0.3)

    ax_propulsion = fig.add_subplot(2, 2, 3)
    ax_propulsion_twin = ax_propulsion.twinx()
    ax_propulsion.plot(time_s, thrust_history_N, color="tab:green", linewidth=2, label="Thrust")
    ax_propulsion_twin.plot(
        time_s, mass_history_kg, color="tab:red", linestyle="--", linewidth=2, label="Mass"
    )
    ax_propulsion.set_title("Thrust & Vehicle Mass")
    ax_propulsion.set_xlabel("Time [s]")
    ax_propulsion.set_ylabel("Thrust [N]", color="tab:green")
    ax_propulsion_twin.set_ylabel("Mass [kg]", color="tab:red")
    ax_propulsion.grid(True, alpha=0.3)

    ax_flight_load = fig.add_subplot(2, 2, 4)
    ax_flight_load_twin = ax_flight_load.twinx()
    ax_flight_load.plot(time_s, q_inf, color="tab:purple", linewidth=2, label="Dynamic pressure")
    ax_flight_load_twin.plot(
        time_s, mach_number, color="tab:brown", linestyle="--", linewidth=2, label="Mach"
    )
    ax_flight_load.set_title("Dynamic Pressure & Mach Number")
    ax_flight_load.set_xlabel("Time [s]")
    ax_flight_load.set_ylabel("Dynamic pressure [Pa]", color="tab:purple")
    ax_flight_load_twin.set_ylabel("Mach number", color="tab:brown")
    ax_flight_load.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path
