
"""
6-DOF Rocket Simulator Streamlit engineering dashboard.
Author: Tegan Broody
School: Ridley High School
Date: October 2026
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

from src.dispersion import run_monte_carlo_dispersion
from src.environment import H_MAX, density, speed_of_sound, wind_velocity
from src.motor import ThrustCurve, parse_eng_motor
from src.physics import (
    G0,
    initial_state_from_launch_angles,
    quaternion_to_pitch_yaw_roll,
    rocket_state_derivative,
    simulate_rocket,
    _quat_to_rotation_matrix,
)
from src.telemetry import (
    FlightTelemetry,
    TelemetryComparison,
    compare_telemetry_to_simulation,
    parse_flight_csv,
)
from src.vehicle import Rocket


st.set_page_config(
    page_title="6-DOF Rocket Simulator",
    layout="wide",
)

COLOR_BLUE = "#6F9FC8"
COLOR_AMBER = "#D2A66B"
COLOR_RED = "#C98282"
COLOR_GREEN = "#79A89B"
COLOR_SLATE = "#AAB6C2"


def _padded_axis_range(
    values: np.ndarray,
    *,
    minimum_padding_m: float = 1.0,
) -> list[float]:
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return [-1.0, 1.0]
    lower = float(np.min(finite_values))
    upper = float(np.max(finite_values))
    padding = max((upper - lower) * 0.05, minimum_padding_m)
    return [lower - padding, upper + padding]


def _sync_control(source_key: str, target_key: str) -> None:
    st.session_state[target_key] = st.session_state[source_key]


def _parameter_control(
    label: str,
    *,
    key: str,
    minimum: float,
    maximum: float,
    default: float,
    step: float,
    format_string: str,
) -> float:
    number_key = f"{key}_number"
    slider_key = f"{key}_slider"
    st.session_state.setdefault(number_key, default)
    st.session_state.setdefault(slider_key, default)

    st.sidebar.number_input(
        label,
        min_value=minimum,
        max_value=maximum,
        step=step,
        format=format_string,
        key=number_key,
        on_change=_sync_control,
        args=(number_key, slider_key),
    )
    st.sidebar.slider(
        f"Adjust {label}",
        min_value=minimum,
        max_value=maximum,
        step=step,
        format=format_string,
        key=slider_key,
        label_visibility="collapsed",
        on_change=_sync_control,
        args=(slider_key, number_key),
    )
    return float(st.session_state[number_key])


def _make_rocket(
    specs: dict[str, float],
    thrust_curve: ThrustCurve | None = None,
    *,
    thrust_scale: float = 1.0,
) -> Rocket:
    radius_m = specs["diameter_m"] / 2.0
    length_m = max(10.0 * specs["diameter_m"], 0.1)
    initial_mass_kg = specs["dry_mass_kg"] + specs["propellant_mass_kg"]
    transverse_inertia = (
        initial_mass_kg * (3.0 * radius_m**2 + length_m**2) / 12.0
    )
    return Rocket(
        dry_mass=specs["dry_mass_kg"],
        propellant_mass=specs["propellant_mass_kg"],
        burn_time=specs["burn_time_s"],
        average_thrust=specs["peak_thrust_N"],
        reference_area=np.pi * radius_m**2,
        drag_coefficient=specs["drag_coefficient"],
        moment_of_inertia=transverse_inertia,
        thrust_curve=thrust_curve,
        thrust_scale=thrust_scale,
    )


def _surface_wind_for_launch(
    wind_speed_m_s: float,
    launch_azimuth_deg: float,
) -> tuple[float, float]:
    azimuth = np.deg2rad(launch_azimuth_deg)
    return (
        -wind_speed_m_s * float(np.sin(azimuth)),
        wind_speed_m_s * float(np.cos(azimuth)),
    )


def _stability_analysis(specs: dict[str, float]) -> dict[str, float]:
    diameter_m = specs["diameter_m"]
    length_m = max(10.0 * diameter_m, 0.1)
    dry_cg_m = 0.45 * length_m
    propellant_cg_m = 0.80 * length_m
    center_of_pressure_m = 0.67 * length_m

    def static_margin(propellant_remaining_kg: float) -> tuple[float, float]:
        total_mass_kg = specs["dry_mass_kg"] + propellant_remaining_kg
        center_of_gravity_m = (
            specs["dry_mass_kg"] * dry_cg_m
            + propellant_remaining_kg * propellant_cg_m
        ) / total_mass_kg
        margin_calibers = (
            center_of_pressure_m - center_of_gravity_m
        ) / diameter_m
        return center_of_gravity_m, margin_calibers

    launch_cg_m, launch_margin = static_margin(specs["propellant_mass_kg"])
    burnout_cg_m, burnout_margin = static_margin(0.0)
    return {
        "length_m": length_m,
        "center_of_pressure_m": center_of_pressure_m,
        "launch_cg_m": launch_cg_m,
        "burnout_cg_m": burnout_cg_m,
        "launch_margin_calibers": launch_margin,
        "burnout_margin_calibers": burnout_margin,
    }


def _stability_history(
    result: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    specs = result["specs"]
    time_s = result["time_s"]
    propellant_remaining_kg = np.maximum(
        np.asarray(result["mass_kg"], dtype=float)
        - specs["dry_mass_kg"],
        0.0,
    )
    dry_cg_m = 0.45 * result["stability"]["length_m"]
    propellant_cg_m = 0.80 * result["stability"]["length_m"]
    total_mass_kg = specs["dry_mass_kg"] + propellant_remaining_kg
    cg_m = (
        specs["dry_mass_kg"] * dry_cg_m
        + propellant_remaining_kg * propellant_cg_m
    ) / total_mass_kg
    margin_calibers = (
        result["stability"]["center_of_pressure_m"] - cg_m
    ) / specs["diameter_m"]
    return cg_m, margin_calibers


def _run_simulation(
    specs: dict[str, float],
    thrust_curve: ThrustCurve | None = None,
) -> dict[str, Any]:
    rocket = _make_rocket(specs, thrust_curve)
    surface_wind = _surface_wind_for_launch(
        specs["wind_speed_m_s"],
        specs["launch_azimuth_deg"],
    )
    initial_state = initial_state_from_launch_angles(
        specs["launch_elevation_deg"],
        specs["launch_azimuth_deg"],
    )
    solution = simulate_rocket(
        rocket,
        initial_state,
        (0.0, 180.0),
        surface_wind=surface_wind,
        wind_enabled=True,
    )
    if not solution.success:
        raise RuntimeError(
            f"Trajectory integration failed: {solution.message}"
        )
    if solution.t.size < 2 or not np.all(np.isfinite(solution.y)):
        raise RuntimeError(
            "The integrator did not produce a finite flight trajectory."
        )
    if not solution.t_events or solution.t_events[0].size == 0:
        raise RuntimeError(
            "The rocket did not return to the ground within 180 seconds. "
            "Try adjusting the vehicle parameters."
        )

    time_s = np.asarray(solution.t, dtype=float)
    azimuth_rad = np.deg2rad(specs["launch_azimuth_deg"])
    launch_frame_rotation = np.array(
        [
            [np.cos(azimuth_rad), np.sin(azimuth_rad), 0.0],
            [-np.sin(azimuth_rad), np.cos(azimuth_rad), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    position_m = launch_frame_rotation @ np.asarray(
        solution.y[0:3], dtype=float
    )
    position_m[2] = np.maximum(position_m[2], 0.0)
    velocity_inertial_m_s = np.asarray(solution.y[3:6], dtype=float)
    velocity_m_s = launch_frame_rotation @ velocity_inertial_m_s
    altitude_m = position_m[2]
    speed_m_s = np.linalg.norm(velocity_m_s, axis=0)
    attitude_rad = np.asarray(
        [
            quaternion_to_pitch_yaw_roll(q)
            for q in solution.y[6:10].T
        ],
        dtype=float,
    )
    attitude_deg = np.rad2deg(attitude_rad)
    attitude_deg[:, 1] = (
        attitude_deg[:, 1] - specs["launch_azimuth_deg"] + 180.0
    ) % 360.0 - 180.0
    orientation_axes = np.asarray(
        [
            launch_frame_rotation @ _quat_to_rotation_matrix(q)
            for q in solution.y[6:10].T
        ],
        dtype=float,
    )
    airspeed_m_s = np.empty_like(speed_m_s)
    mach = np.empty_like(speed_m_s)
    dynamic_pressure_pa = np.empty_like(speed_m_s)
    acceleration_g = np.empty_like(speed_m_s)

    for index, sample_time_s in enumerate(time_s):
        altitude = float(np.clip(altitude_m[index], 0.0, H_MAX))
        air_velocity = velocity_inertial_m_s[:, index] - np.asarray(
            wind_velocity(altitude, surface_wind=surface_wind)
        )
        airspeed_m_s[index] = np.linalg.norm(air_velocity)
        sound_speed = float(speed_of_sound(altitude))
        mach[index] = airspeed_m_s[index] / sound_speed
        dynamic_pressure_pa[index] = (
            0.5 * float(density(altitude)) * airspeed_m_s[index] ** 2
        )
        state_derivative = rocket_state_derivative(
            float(sample_time_s),
            solution.y[:, index],
            rocket,
            surface_wind=surface_wind,
        )
        acceleration_g[index] = (
            np.linalg.norm(state_derivative[3:6]) / G0
        )

    apogee_detected = (
        len(solution.t_events) > 1
        and len(solution.t_events[1]) > 0
        and len(solution.y_events) > 1
        and len(solution.y_events[1]) > 0
    )

    apogee_time_s = (
        float(solution.t_events[1][0])
        if apogee_detected
        else None
    )

    apogee_altitude_m = (
        float(solution.y_events[1][0][2])
        if apogee_detected
        else float("nan")
    )

    burnout_time_s = (
        float(solution.t_events[2][0])
        if len(solution.t_events) > 2 and len(solution.t_events[2]) > 0
        else None
    )

    flight_time_s = (
        float(solution.t_events[0][0])
        if len(solution.t_events) > 0 and len(solution.t_events[0]) > 0
        else float(time_s[-1])
    )

    transonic_band = (mach >= 0.8) & (mach <= 1.2)
    stability = _stability_analysis(specs)

    return {
        "time_s": time_s,
        "position_m": position_m,
        "velocity_m_s": velocity_m_s,
        "attitude_deg": attitude_deg,
        "orientation_axes": orientation_axes,
        "altitude_m": altitude_m,
        "speed_m_s": speed_m_s,
        "mach": mach,
        "acceleration_g": acceleration_g,
        "dynamic_pressure_pa": dynamic_pressure_pa,
        "events": {
            "liftoff_s": 0.0,
            "burnout_s": burnout_time_s,
            "apogee_s": apogee_time_s,
            "apogee_altitude_m": apogee_altitude_m,
            "flight_time_s": flight_time_s,
        },
        "stability": stability,
        "mass_kg": np.asarray(
            [rocket.update(sample_time)[0] for sample_time in time_s],
            dtype=float,
        ),
        "specs": specs.copy(),
        "metrics": {
            "apogee_m": apogee_altitude_m,
            "max_speed_m_s": float(np.max(speed_m_s)),
            "max_mach": float(np.max(mach)),
            "transonic_band_reached": bool(np.any(transonic_band)),
            "max_dynamic_pressure_pa": float(
                np.max(dynamic_pressure_pa)
            ),
            "peak_acceleration_g": float(np.max(acceleration_g)),
            "flight_time_s": flight_time_s,
        },
    }


def _orthonormal_body_axes(axes: np.ndarray) -> np.ndarray:
    axis_x = axes[:, 0]
    axis_x /= max(float(np.linalg.norm(axis_x)), 1e-12)
    axis_y = axes[:, 1] - axis_x * float(np.dot(axes[:, 1], axis_x))
    axis_y /= max(float(np.linalg.norm(axis_y)), 1e-12)
    axis_z = np.cross(axis_x, axis_y)
    return np.column_stack((axis_x, axis_y, axis_z))


def _pose_traces_at_time(
    result: dict[str, Any],
    selected_time_s: float,
    model_length_m: float,
) -> tuple[go.Mesh3d | go.Scatter3d, ...]:
    time_s = result["time_s"]
    position_m = result["position_m"]
    selected_position = np.asarray(
        [
            np.interp(selected_time_s, time_s, position_m[axis])
            for axis in range(3)
        ],
        dtype=float,
    )
    selected_axes = np.asarray(
        [
            [
                np.interp(
                    selected_time_s,
                    time_s,
                    result["orientation_axes"][:, row, column],
                )
                for column in range(3)
            ]
            for row in range(3)
        ],
        dtype=float,
    )
    selected_axes = _orthonormal_body_axes(selected_axes)
    selected_attitude = np.asarray(
        [
            np.interp(
                selected_time_s,
                time_s,
                result["attitude_deg"][:, angle],
            )
            for angle in range(3)
        ],
        dtype=float,
    )
    body_length_m = model_length_m
    body_radius_m = body_length_m * 0.08
    ring_angles = np.linspace(0.0, 2.0 * np.pi, 9)[:-1]
    body_vertices = np.asarray(
        [
            (
                body_radius_m * np.cos(angle),
                body_radius_m * np.sin(angle),
                z,
            )
            for z in (
                -0.5 * body_length_m,
                0.22 * body_length_m,
            )
            for angle in ring_angles
        ]
        + [(0.0, 0.0, 0.5 * body_length_m)],
        dtype=float,
    )
    ring_size = len(ring_angles)
    nose_tip_index = 2 * ring_size
    face_i: list[int] = []
    face_j: list[int] = []
    face_k: list[int] = []
    for ring_index in range(ring_size):
        next_index = (ring_index + 1) % ring_size
        face_i.extend(
            (ring_index, ring_index, ring_size + ring_index)
        )
        face_j.extend(
            (
                next_index,
                ring_size + next_index,
                ring_size + next_index,
            )
        )
        face_k.extend(
            (
                ring_size + next_index,
                ring_size + ring_index,
                nose_tip_index,
            )
        )
    for fin_index in range(4):
        angle = fin_index * np.pi / 2.0
        fin_root = np.asarray(
            [
                body_radius_m * np.cos(angle),
                body_radius_m * np.sin(angle),
                -0.45 * body_length_m,
            ]
        )
        fin_tip = np.asarray(
            [
                body_radius_m * 2.8 * np.cos(angle),
                body_radius_m * 2.8 * np.sin(angle),
                -0.48 * body_length_m,
            ]
        )
        fin_trailing = np.asarray(
            [
                body_radius_m * np.cos(angle),
                body_radius_m * np.sin(angle),
                -0.12 * body_length_m,
            ]
        )
        first = len(body_vertices)
        body_vertices = np.vstack(
            (body_vertices, fin_root, fin_tip, fin_trailing)
        )
        face_i.append(first)
        face_j.append(first + 1)
        face_k.append(first + 2)

    world_vertices = (
        selected_position[:, np.newaxis]
        + selected_axes @ body_vertices.T
    )
    body_trace = go.Mesh3d(
        x=world_vertices[0],
        y=world_vertices[1],
        z=world_vertices[2],
        i=face_i,
        j=face_j,
        k=face_k,
        name="Rocket body",
        color="#E2EAF0",
        opacity=1.0,
        flatshading=True,
        lighting={
            "ambient": 0.85,
            "diffuse": 0.9,
            "specular": 0.35,
            "roughness": 0.55,
        },
        customdata=np.tile(
            [selected_time_s, *selected_attitude.tolist()],
            (len(body_vertices), 1),
        ),
        hovertemplate=(
            "Rocket · %{customdata[0]:.2f} s<br>"
            "Pitch: %{customdata[1]:.1f}°<br>"
            "Yaw: %{customdata[2]:.1f}°<br>"
            "Roll: %{customdata[3]:.1f}°<extra></extra>"
        ),
        showscale=False,
    )
    axis_traces = tuple(
        go.Scatter3d(
            x=[
                selected_position[0],
                selected_position[0]
                + selected_axes[0, axis_index] * body_length_m,
            ],
            y=[
                selected_position[1],
                selected_position[1]
                + selected_axes[1, axis_index] * body_length_m,
            ],
            z=[
                selected_position[2],
                selected_position[2]
                + selected_axes[2, axis_index] * body_length_m,
            ],
            mode="lines+markers",
            name=axis_label,
            line={"color": axis_color, "width": 10},
            marker={"color": axis_color, "size": [0, 7]},
            hoverinfo="skip",
            showlegend=False,
        )
        for axis_index, (axis_label, axis_color) in enumerate(
            (
                ("Body X", COLOR_RED),
                ("Body Y", COLOR_GREEN),
                ("Body Z", COLOR_BLUE),
            )
        )
    )
    return (body_trace, *axis_traces)


def _trajectory_chart(
    result: dict[str, Any],
    telemetry: FlightTelemetry | None = None,
) -> go.Figure:
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(
            x=result["time_s"],
            y=result["altitude_m"],
            name="Altitude",
            mode="lines+markers",
            line={"color": COLOR_BLUE, "width": 2},
            marker={"size": 3},
            customdata=result["time_s"],
            fill="tozeroy",
            fillcolor="rgba(111, 159, 200, 0.10)",
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=result["time_s"],
            y=result["speed_m_s"],
            name="Velocity",
            mode="lines+markers",
            line={"color": COLOR_AMBER, "width": 2},
            marker={"size": 3},
            customdata=result["time_s"],
        ),
        secondary_y=True,
    )
    if telemetry is not None:
        figure.add_trace(
            go.Scatter(
                x=telemetry.time_s,
                y=telemetry.altitude_m,
                name="Real flight · altitude",
                mode="lines+markers",
                line={"color": COLOR_RED, "width": 2, "dash": "dash"},
                marker={"size": 4},
                hovertemplate=(
                    "Real flight<br>Time: %{x:.2f} s<br>"
                    "Altitude: %{y:.1f} m<extra></extra>"
                ),
            ),
            secondary_y=False,
        )
        figure.add_trace(
            go.Scatter(
                x=telemetry.time_s,
                y=telemetry.velocity_m_s,
                name="Real flight · vertical velocity",
                mode="lines",
                line={"color": COLOR_SLATE, "width": 2, "dash": "dot"},
                hovertemplate=(
                    "Real flight<br>Time: %{x:.2f} s<br>"
                    "Vertical velocity: %{y:.1f} m/s<extra></extra>"
                ),
            ),
            secondary_y=True,
        )
    figure.update_layout(
        template="plotly_dark",
        title="Altitude and velocity over flight time",
        font={"family": "Arial, sans-serif", "size": 10},
        title_font={"size": 16},
        xaxis_title="Flight time (s)",
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.3,
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 10},
            "itemclick": "toggleothers",
            "itemdoubleclick": "toggle",
        },
        margin={"t": 50, "b": 80, "l": 20, "r": 20},
    )
    figure.update_yaxes(
        title_text="Altitude (m)",
        secondary_y=False,
        rangemode="tozero",
    )
    figure.update_yaxes(title_text="Velocity (m/s)", secondary_y=True)
    return figure


def _loads_chart(result: dict[str, Any]) -> go.Figure:
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.16,
        subplot_titles=("Acceleration", "Dynamic pressure"),
    )
    figure.add_trace(
        go.Scatter(
            x=result["time_s"],
            y=result["acceleration_g"],
            name="Acceleration",
            mode="lines",
            line={"color": COLOR_BLUE, "width": 2},
            fill="tozeroy",
            fillcolor="rgba(111, 159, 200, 0.10)",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=result["time_s"],
            y=result["dynamic_pressure_pa"],
            name="Dynamic pressure",
            mode="lines",
            line={"color": COLOR_AMBER, "width": 2},
            fill="tozeroy",
            fillcolor="rgba(210, 166, 107, 0.10)",
        ),
        row=2,
        col=1,
    )
    figure.update_layout(
        template="plotly_dark",
        title="Flight loads",
        font={"family": "Arial, sans-serif", "size": 12},
        title_font={"size": 16},
        xaxis2_title="Flight time (s)",
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.25,
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 10},
            "itemclick": "toggleothers",
            "itemdoubleclick": "toggle",
        },
        margin={"t": 50, "b": 80, "l": 20, "r": 20},
    )
    figure.update_yaxes(
        title_text="Acceleration (G)",
        rangemode="tozero",
        row=1,
        col=1,
    )
    figure.update_yaxes(
        title_text="Dynamic pressure (Pa)",
        rangemode="tozero",
        row=2,
        col=1,
    )
    return figure

def _position_3d_chart(
    result: dict[str, Any],
    telemetry: FlightTelemetry | None = None,
    selected_time_s: float | None = None,
    *,
    animated: bool = False,
    playback_speed: float = 1.0,
    equal_proportional_scale: bool = False,
) -> go.Figure:
    position_m = result["position_m"]

    apogee_time_s = result["events"]["apogee_s"]
    if apogee_time_s is not None:
        apogee_index = int(
            np.argmin(
                np.abs(
                    np.asarray(result["time_s"], dtype=float)
                    - apogee_time_s
                )
            )
        )
    else:
        apogee_index = int(np.argmax(position_m[2]))
    x_values = [np.asarray(position_m[0], dtype=float)]
    y_values = [np.asarray(position_m[1], dtype=float)]
    z_values = [np.asarray(position_m[2], dtype=float)]
    if telemetry is not None and not animated:
        x_values.append(
            np.asarray(
                telemetry.downrange_m
                if telemetry.downrange_m is not None
                else np.zeros_like(telemetry.altitude_m),
                dtype=float,
            )
        )
        y_values.append(
            np.asarray(
                telemetry.crossrange_m
                if telemetry.crossrange_m is not None
                else np.zeros_like(telemetry.altitude_m),
                dtype=float,
            )
        )
        z_values.append(np.asarray(telemetry.altitude_m, dtype=float))

    vehicle_length_m = max(
        10.0 * float(result["specs"]["diameter_m"]),
        0.1,
    )
    trajectory_scale_m = max(
        float(np.ptp(position_m[0])),
        float(np.ptp(position_m[1])),
        float(np.ptp(position_m[2])),
        1.0,
    )
    body_length_m = max(
        vehicle_length_m * 8.0,
        trajectory_scale_m * 0.04,
    )
    if animated and selected_time_s is None:
        selected_time_s = 0.0
    model_padding_m = (
        body_length_m * 1.1
        if selected_time_s is not None
        else 1.0
    )
    data_ranges = (
        np.concatenate(x_values),
        np.concatenate(y_values),
        np.concatenate(z_values),
    )
    padded_ranges = [
        _padded_axis_range(
            axis_values,
            minimum_padding_m=model_padding_m,
        )
        for axis_values in data_ranges
    ]
    if selected_time_s is not None:
        target_axis_span_m = max(
            max(
                axis_range[1] - axis_range[0]
                for axis_range in padded_ranges
            ) / 3.0,
            2.0 * model_padding_m,
        )
        for axis_range in padded_ranges:
            current_span_m = axis_range[1] - axis_range[0]
            if current_span_m < target_axis_span_m:
                extra_padding_m = (
                    target_axis_span_m - current_span_m
                ) / 2.0
                axis_range[0] -= extra_padding_m
                axis_range[1] += extra_padding_m
    x_range, y_range, z_range = padded_ranges

    scene_aspect: dict[str, Any]
    if equal_proportional_scale:
        padded_spans = np.asarray(
            [
                x_range[1] - x_range[0],
                y_range[1] - y_range[0],
                z_range[1] - z_range[0],
            ],
            dtype=float,
        )
        capped_spans = np.maximum(
            padded_spans,
            np.max(padded_spans) / 3.0,
        )
        aspect_ratio = capped_spans / np.max(capped_spans)
        scene_aspect = {
            "aspectmode": "manual",
            "aspectratio": {
                "x": float(aspect_ratio[0]),
                "y": float(aspect_ratio[1]),
                "z": float(aspect_ratio[2]),
            },
        }
    else:
        scene_aspect = {"aspectmode": "data"}

    figure = go.Figure()
    if animated:
        figure.add_trace(
            go.Scatter3d(
                x=position_m[0],
                y=position_m[1],
                z=position_m[2],
                mode="lines",
                name="Position path",
                line={"color": "#6F8799", "width": 3},
                hovertemplate=(
                    "X: %{x:.1f} m<br>Y: %{y:.1f} m<br>Z: %{z:.1f} m"
                    "<extra>Position path</extra>"
                ),
            )
        )
    else:
        figure.add_trace(
            go.Scatter3d(
                x=position_m[0],
                y=position_m[1],
                z=position_m[2],
                mode="lines+markers",
                name="Flight path",
                line={
                    "color": result["time_s"],
                    "colorscale": "Viridis_r",
                    "cmin": float(np.min(result["time_s"])),
                    "cmax": float(np.max(result["time_s"])),
                    "showscale": True,
                    "colorbar": {
                        "title": {"text": "Flight time (s)"},
                        "thickness": 14,
                        "len": 0.8,
                        "x": 1.2,
                    },
                    "width": 4,
                },
                marker={
                    "size": 2,
                    "color": result["time_s"],
                    "colorscale": "Viridis_r",
                    "showscale": False,
                },
                customdata=np.column_stack(
                    (
                        result["time_s"],
                        result["speed_m_s"],
                        result["attitude_deg"],
                    )
                ),
                hovertemplate=(
                    "Time: %{customdata[0]:.2f} s<br>"
                    "X: %{x:.1f} m<br>Y: %{y:.1f} m<br>Z: %{z:.1f} m<br>"
                    "Speed: %{customdata[1]:.1f} m/s<br>"
                    "Pitch: %{customdata[2]:.1f}°<br>"
                    "Yaw: %{customdata[3]:.1f}°<br>"
                    "Roll: %{customdata[4]:.1f}°<extra></extra>"
                ),
            )
        )

    if animated or selected_time_s is not None:
        figure.add_traces(
            list(
                _pose_traces_at_time(
                    result,
                    float(selected_time_s or 0.0),
                    body_length_m,
                )
            )
        )

    if telemetry is not None and not animated:
        has_empirical_xy = (
            telemetry.downrange_m is not None
            and telemetry.crossrange_m is not None
        )
        figure.add_trace(
            go.Scatter3d(
                x=(
                    telemetry.downrange_m
                    if has_empirical_xy
                    else np.zeros_like(telemetry.altitude_m)
                ),
                y=(
                    telemetry.crossrange_m
                    if has_empirical_xy
                    else np.zeros_like(telemetry.altitude_m)
                ),
                z=telemetry.altitude_m,
                mode="lines+markers",
                name=(
                    "Real flight"
                    if has_empirical_xy
                    else "Real flight · altimeter only (X/Y unavailable)"
                ),
                line={"color": COLOR_RED, "width": 3, "dash": "dash"},
                marker={"color": COLOR_RED, "size": 2},
                customdata=np.column_stack(
                    (telemetry.time_s, telemetry.velocity_m_s)
                ),
                hovertemplate=(
                    "Real flight<br>Time: %{customdata[0]:.2f} s<br>"
                    "X: %{x:.1f} m<br>Y: %{y:.1f} m<br>Z: %{z:.1f} m<br>"
                    "Velocity: %{customdata[1]:.1f} m/s<extra></extra>"
                ),
            )
        )

    if not animated:
        marker_indices = [
            0,
            apogee_index,
            len(result["time_s"]) - 1,
        ]
        marker_labels = ["Launch", "Apogee", "Impact"]
        figure.add_trace(
            go.Scatter3d(
                x=position_m[0, marker_indices],
                y=position_m[1, marker_indices],
                z=position_m[2, marker_indices],
                mode="markers+text",
                name="Flight milestones",
                text=marker_labels,
                textposition="top center",
                marker={
                    "size": 4,
                    "color": [COLOR_GREEN, COLOR_AMBER, COLOR_SLATE],
                },
                customdata=np.column_stack(
                    (
                        result["time_s"][marker_indices],
                        result["attitude_deg"][marker_indices],
                    )
                ),
                hovertemplate=(
                    "%{text} · %{customdata[0]:.2f} s<br>"
                    "X: %{x:.1f} m<br>Y: %{y:.1f} m<br>"
                    "Z: %{z:.1f} m<br>"
                    "Pitch: %{customdata[1]:.1f}°<br>"
                    "Yaw: %{customdata[2]:.1f}°<br>"
                    "Roll: %{customdata[3]:.1f}°<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        template="plotly_dark",
        title=(
            "Ground Station replay · inertial position and rocket body axes"
            if animated
            else "3D flight path · downrange (X), crossrange (Y), altitude (Z)"
        ),
        font={"family": "Arial, sans-serif", "size": 12},
        title_font={"size": 15},
        title_x=0.5,
        scene={
            "xaxis_title": "X — downrange along launch azimuth (m)",
            "yaxis_title": "Y — crossrange from wind and asymmetry (m)",
            "zaxis_title": "Z — altitude (m)",
            "xaxis": {"range": x_range, "autorange": False},
            "yaxis": {"range": y_range, "autorange": False},
            "zaxis": {"range": z_range, "autorange": False},
            **scene_aspect,
            "camera": {
                "eye": (
                    {"x": 0.82, "y": -0.82, "z": 0.65}
                    if animated
                    else {"x": 1.5, "y": -1.5, "z": 1.1}
                ),
                "up": {"x": 0.0, "y": 0.0, "z": 1.0},
                "projection": {"type": "orthographic"},
            },
            "dragmode": "orbit",
            "uirevision": "rocket-trajectory-camera-v4",
        },
        uirevision="rocket-trajectory-layout",
        hovermode="closest",
        showlegend=not animated,
        legend={
            "orientation": "h",
            "x": 0.01,
            "y": 0.98,
            "xanchor": "left",
            "yanchor": "top",
            "bgcolor": "rgba(15, 23, 32, 0.75)",
            "itemclick": "toggleothers",
            "itemdoubleclick": "toggle",
            "font": {"size": 10},
        },
        margin={
            "l": 40,
            "r": 35 if animated else 120,
            "t": 75,
            "b": 45,
        },
        height=760 if animated else 560,
    )

    if animated:
        time_s = np.asarray(result["time_s"], dtype=float)
        flight_time_s = float(np.ptp(time_s))
        frame_count = min(
            1200,
            max(
                2,
                int(
                    np.ceil(
                        flight_time_s * 30.0 / max(playback_speed, 0.5)
                    )
                ),
            ),
        )
        apogee_time_s = float(time_s[apogee_index])
        frame_times_s = np.unique(
            np.append(
                np.linspace(
                    float(time_s[0]),
                    float(time_s[-1]),
                    frame_count,
                ),
                apogee_time_s,
            )
        )
        frame_names = [
            f"{frame_time_s:.6f}"
            for frame_time_s in frame_times_s
        ]
        figure.frames = tuple(
            go.Frame(
                name=frame_name,
                data=(
                    *_pose_traces_at_time(
                        result,
                        float(frame_time_s),
                        body_length_m,
                    ),
                ),
                traces=[1, 2, 3, 4],
            )
            for frame_time_s, frame_name in zip(
                frame_times_s,
                frame_names,
            )
        )
        active_frame = int(
            np.argmin(
                np.abs(
                    frame_times_s
                    - float(selected_time_s or time_s[0])
                )
            )
        )
        frame_interval_ms = (
            float(np.ptp(frame_times_s))
            / max(len(frame_times_s) - 1, 1)
            / max(playback_speed, 0.5)
            * 1000.0
        )
        animation_options = {
            "frame": {
                "duration": max(1, round(frame_interval_ms)),
                "redraw": True,
            },
            "transition": {"duration": 0},
            "mode": "immediate",
            "fromcurrent": True,
        }
        figure.update_layout(
            updatemenus=[
                {
                    "type": "buttons",
                    "direction": "left",
                    "x": 0.0,
                    "y": -0.12,
                    "showactive": False,
                    "buttons": [
                        {
                            "label": "Start / Restart",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    **animation_options,
                                    "fromcurrent": False,
                                },
                            ],
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [
                                [None],
                                {
                                    "frame": {
                                        "duration": 0,
                                        "redraw": False,
                                    },
                                    "transition": {"duration": 0},
                                    "mode": "immediate",
                                },
                            ],
                        },
                    ],
                }
            ],
            sliders=[
                {
                    "active": active_frame,
                    "x": 0.0,
                    "y": -0.2,
                    "len": 0.95,
                    "currentvalue": {
                        "prefix": "Animation time: ",
                        "suffix": " s",
                    },
                    "steps": [
                        {
                            "label": f"{frame_time_s:.2f}",
                            "method": "animate",
                            "args": [
                                [frame_name],
                                {
                                    **animation_options,
                                    "mode": "immediate",
                                },
                            ],
                        }
                        for frame_time_s, frame_name in zip(
                            frame_times_s,
                            frame_names,
                        )
                    ],
                }
            ],
            margin={"l": 40, "r": 35, "t": 75, "b": 90},
        )
    return figure


def _render_ground_station_animation(figure: go.Figure) -> None:
    camera_sync_script = """
const plot = document.getElementById('{plot_id}');
let savedCamera = JSON.parse(JSON.stringify(
    plot._fullLayout.scene._scene.getCamera()
));
let restoringCamera = false;
let cameraGestureActive = false;
const readCamera = () => JSON.parse(JSON.stringify(
    plot._fullLayout.scene._scene.getCamera()
));

plot.addEventListener('pointerdown', () => {
    cameraGestureActive = true;
});

const saveCameraAfterGesture = () => {
    window.setTimeout(() => {
        savedCamera = readCamera();
        cameraGestureActive = false;
    }, 75);
};
plot.addEventListener('pointerup', saveCameraAfterGesture);
plot.addEventListener('pointercancel', saveCameraAfterGesture);

plot.on('plotly_relayout', (update) => {
    if (
        !restoringCamera
        && Object.keys(update).some((key) => key.startsWith('scene.camera'))
    ) {
        savedCamera = readCamera();
    }
});

// Preserve rotations immediately while replay frames continue to update data.
plot.on('plotly_relayouting', (update) => {
    if (Object.keys(update).some((key) => key.startsWith('scene.camera'))) {
        savedCamera = readCamera();
    }
});

plot.on('plotly_afterplot', () => {
    window.requestAnimationFrame(() => {
        if (restoringCamera || cameraGestureActive) {
            return;
        }
        const currentCamera = readCamera();
        if (JSON.stringify(currentCamera) !== JSON.stringify(savedCamera)) {
            restoringCamera = true;
            Plotly.relayout(plot, {'scene.camera': savedCamera}).finally(() => {
                restoringCamera = false;
            });
        }
    });
});
"""
    components.html(
        figure.to_html(
            full_html=False,
            include_plotlyjs=True,
            post_script=camera_sync_script,
            config={"scrollZoom": True, "displaylogo": False},
            auto_play=False,
            div_id="ground-station-replay",
        ),
        height=760,
        scrolling=False,
    )


def _telemetry_at_time(
    result: dict[str, Any],
    selected_time_s: float,
) -> dict[str, float]:
    """Interpolate telemetry arrays at the exact playback slider timestamp."""
    time_s = result["time_s"]

    def sample(values: np.ndarray) -> float:
        return float(np.interp(selected_time_s, time_s, values))

    attitude_deg = result["attitude_deg"]
    return {
        "altitude_m": sample(result["altitude_m"]),
        "velocity_m_s": sample(result["speed_m_s"]),
        "acceleration_g": sample(result["acceleration_g"]),
        "mach": sample(result["mach"]),
        "pitch_deg": sample(attitude_deg[:, 0]),
        "yaw_deg": sample(attitude_deg[:, 1]),
        "roll_deg": sample(attitude_deg[:, 2]),
    }


def _set_playback_from_plot(
    plot_state: Any,
    *,
    plot_key: str,
    playback_key: str,
    flight_time_s: float,
) -> None:
    selection = getattr(plot_state, "selection", None)
    points = (
        getattr(selection, "points", [])
        if selection is not None
        else []
    )
    if not points:
        return
    point = points[0]
    customdata = point.get("customdata")
    selected_time = (
        customdata[0]
        if isinstance(customdata, (list, tuple, np.ndarray))
        else customdata
    )
    if selected_time is None:
        selected_time = point.get("x")
    if selected_time is None or not np.isfinite(float(selected_time)):
        return
    selection_signature = (
        plot_key,
        point.get("curve_number"),
        point.get("point_index"),
        float(selected_time),
    )
    if (
        st.session_state.get("ground_station_last_plot_selection")
        == selection_signature
    ):
        return
    st.session_state["ground_station_last_plot_selection"] = (
        selection_signature
    )
    st.session_state[playback_key] = float(
        np.clip(float(selected_time), 0.0, flight_time_s)
    )


def _render_ground_station_inspector(
    result: dict[str, Any],
    flight_time_s: float,
    playback_key: str,
) -> None:
    selected_time_s = st.slider(
        "Telemetry inspection time (s)",
        min_value=0.0,
        max_value=flight_time_s,
        step=max(flight_time_s / 1000.0, 0.001),
        key=playback_key,
        help=(
            "Inspect any time in the flight; telemetry and the body-axis "
            "inspector update to this frame."
        ),
    )
    frame = _telemetry_at_time(result, selected_time_s)
    telemetry_columns = st.columns(4)
    telemetry_columns[0].metric(
        "Altitude",
        f"{frame['altitude_m']:,.1f} m",
    )
    telemetry_columns[1].metric(
        "Velocity",
        f"{frame['velocity_m_s']:,.1f} m/s",
    )
    telemetry_columns[2].metric(
        "Acceleration",
        f"{frame['acceleration_g']:,.2f} G",
    )
    telemetry_columns[3].metric("Mach", f"{frame['mach']:.3f}")
    st.caption(
        f"Frame at {selected_time_s:.3f} s · "
        f"Pitch {frame['pitch_deg']:.1f}° · "
        f"Yaw {frame['yaw_deg']:.1f}° · "
        f"Roll {frame['roll_deg']:.1f}°"
    )


def _spatial_timeseries_charts(
    result: dict[str, Any],
) -> tuple[go.Figure, go.Figure, go.Figure]:
    chart_groups = (
        (
            "Position components",
            "Position (m)",
            result["position_m"],
            (("X", COLOR_BLUE), ("Y", COLOR_GREEN), ("Z", COLOR_AMBER)),
        ),
        (
            "Velocity components",
            "Velocity (m/s)",
            result["velocity_m_s"],
            (("Vx", COLOR_BLUE), ("Vy", COLOR_GREEN), ("Vz", COLOR_AMBER)),
        ),
        (
            "Vehicle attitude",
            "Angle (°)",
            result["attitude_deg"].T,
            (
                ("Pitch", COLOR_BLUE),
                ("Yaw", COLOR_GREEN),
                ("Roll", COLOR_AMBER),
            ),
        ),
    )
    charts: list[go.Figure] = []
    for title, y_title, values, component_styles in chart_groups:
        figure = go.Figure()
        for component, (name, color) in enumerate(component_styles):
            figure.add_trace(
                go.Scatter(
                    x=result["time_s"],
                    y=values[component],
                    name=name,
                    mode="lines+markers",
                    line={"color": color, "width": 2},
                    marker={"size": 3},
                    customdata=result["time_s"],
                    hovertemplate=(
                        f"{name}: %{{y:.3f}}<br>"
                        "Time: %{customdata:.3f} s<extra></extra>"
                    ),
                )
            )
        figure.update_layout(
            template="plotly_dark",
            title=title,
            font={"family": "Arial, sans-serif", "size": 10},
            title_font={"size": 15},
            xaxis_title="Flight time (s)",
            yaxis_title=y_title,
            hovermode="x unified",
            legend={
                "orientation": "h",
                "yanchor": "top",
                "y": -0.2,
                "xanchor": "center",
                "x": 0.5,
                "font": {"size": 10},
                "itemclick": "toggleothers",
                "itemdoubleclick": "toggle",
            },
            margin={"t": 50, "b": 80, "l": 25, "r": 25},
            height=350,
        )
        charts.append(figure)
    return charts[0], charts[1], charts[2]


def _validation_metrics(
    result: dict[str, Any],
    telemetry: FlightTelemetry,
) -> TelemetryComparison:
    return compare_telemetry_to_simulation(
        result["time_s"],
        result["altitude_m"],
        telemetry,
    )


def _motor_signature(curve: ThrustCurve | None) -> tuple[Any, ...] | None:
    if curve is None:
        return None
    return (
        curve.designation,
        curve.burn_time_s,
        curve.total_impulse_ns,
        curve.propellant_mass_kg,
        tuple(curve.times_s.tolist()),
        tuple(curve.thrust_n.tolist()),
    )


def _dispersion_chart(
    dispersion: dict[str, np.ndarray],
    nominal_result: dict[str, Any] | None,
) -> go.Figure:
    landing_x = dispersion["landing_x_m"]
    landing_y = dispersion["landing_y_m"]
    center_x = float(np.mean(landing_x))
    center_y = float(np.mean(landing_y))
    radial_distance = np.hypot(
        landing_x - center_x,
        landing_y - center_y,
    )
    radius_95 = float(np.quantile(radial_distance, 0.95))
    angles = np.linspace(0.0, 2.0 * np.pi, 180)

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=landing_x,
            y=landing_y,
            mode="markers",
            name="Monte Carlo landings",
            marker={
                "size": 8,
                "color": dispersion["apogee_m"],
                "colorscale": "Viridis",
                "showscale": True,
                "colorbar": {"title": "Apogee (m)"},
                "line": {"color": "#D6DEE6", "width": 0.5},
            },
            customdata=np.column_stack(
                (
                    dispersion["apogee_m"],
                    dispersion["wind_speed_m_s"],
                    dispersion["thrust_scale"],
                )
            ),
            hovertemplate=(
                "Downrange: %{x:.1f} m<br>Crossrange: %{y:.1f} m<br>"
                "Apogee: %{customdata[0]:.1f} m<br>"
                "Wind: %{customdata[1]:.1f} m/s<br>"
                "Thrust scale: %{customdata[2]:.3f}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=center_x + radius_95 * np.cos(angles),
            y=center_y + radius_95 * np.sin(angles),
            mode="lines",
            name="95th-percentile radial footprint",
            line={"color": COLOR_AMBER, "dash": "dash", "width": 1.5},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[center_x],
            y=[center_y],
            mode="markers",
            name="Mean landing",
            marker={"symbol": "diamond", "size": 11, "color": COLOR_GREEN},
        )
    )
    if nominal_result is not None:
        figure.add_trace(
            go.Scatter(
                x=[nominal_result["position_m"][0, -1]],
                y=[nominal_result["position_m"][1, -1]],
                mode="markers",
                name="Nominal landing",
                marker={"symbol": "star", "size": 13, "color": COLOR_RED},
            )
        )
    figure.update_layout(
        template="plotly_dark",
        title="Landing dispersion footprint",
        xaxis_title="Downrange X (m)",
        yaxis_title="Crossrange Y (m)",
        yaxis={"scaleanchor": "x", "scaleratio": 1},
        hovermode="closest",
        font={"family": "Arial, sans-serif", "size": 10},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.25,
            "xanchor": "center",
            "x": 0.5,
            "itemclick": "toggleothers",
            "itemdoubleclick": "toggle",
        },
        margin={"t": 50, "b": 85, "l": 25, "r": 25},
        height=560,
    )
    return figure


st.title("6-DOF High-Power Rocket Simulation & Flight Analysis Engine")
st.markdown("### Developed by **Tegan Broody** | Ridley High School")
st.markdown("---")
st.caption(
    "Explore a numerically integrated rocket flight using the project’s 6-DOF "
    "dynamics, aerodynamic loads, and standard-atmosphere model."
)

st.sidebar.title("Vehicle configuration")
st.sidebar.caption("Set values with either the number field or its slider.")
st.sidebar.subheader("Mass & propulsion")
dry_mass_kg = _parameter_control(
    "Dry Mass (kg)",
    key="dry_mass",
    minimum=0.1,
    maximum=500.0,
    default=12.0,
    step=0.1,
    format_string="%.1f",
)
propellant_mass_kg = _parameter_control(
    "Propellant Mass (kg)",
    key="propellant_mass",
    minimum=0.0,
    maximum=500.0,
    default=3.0,
    step=0.1,
    format_string="%.1f",
)
burn_time_s = _parameter_control(
    "Motor Burn Time (s)",
    key="burn_time",
    minimum=0.1,
    maximum=120.0,
    default=5.0,
    step=0.1,
    format_string="%.1f",
)
peak_thrust_N = _parameter_control(
    "Peak Thrust (N)",
    key="peak_thrust",
    minimum=0.0,
    maximum=50_000.0,
    default=1_200.0,
    step=10.0,
    format_string="%.1f",
)
st.sidebar.caption(
    "Optionally replace the constant-thrust model with a RASP/RockSim motor curve."
)
uploaded_eng = st.sidebar.file_uploader(
    "Custom motor curve (.eng)",
    type=["eng"],
    help=(
        "Standard RASP format: motor header followed by time (s) and thrust (N) "
        "rows. Header propellant mass (g) is used for the simulation."
    ),
)
thrust_curve: ThrustCurve | None = None
if uploaded_eng is not None:
    try:
        thrust_curve = parse_eng_motor(
            uploaded_eng.getvalue().decode("utf-8-sig")
        )
        st.sidebar.success(
            f"{thrust_curve.designation}: {thrust_curve.burn_time_s:.2f} s, "
            f"{thrust_curve.total_impulse_ns:.1f} N·s"
        )
        st.sidebar.caption(
            f"Motor header propellant mass: "
            f"{thrust_curve.propellant_mass_kg * 1000:.1f} g. "
            "This value and the curve burn time override the manual inputs."
        )
    except (UnicodeDecodeError, ValueError) as error:
        st.sidebar.error(f"Could not read .eng motor file: {error}")

st.sidebar.subheader("Aerodynamics")
diameter_m = _parameter_control(
    "Rocket Diameter (m)",
    key="diameter",
    minimum=0.01,
    maximum=2.0,
    default=0.1,
    step=0.01,
    format_string="%.2f",
)
drag_coefficient = _parameter_control(
    "Drag Coefficient (Cd)",
    key="drag_coefficient",
    minimum=0.0,
    maximum=2.0,
    default=0.5,
    step=0.01,
    format_string="%.2f",
)
st.sidebar.subheader("Launch rail")
launch_elevation_deg = _parameter_control(
    "Launch Elevation (°)",
    key="launch_elevation",
    minimum=5.0,
    maximum=90.0,
    default=90.0,
    step=1.0,
    format_string="%.0f",
)
launch_azimuth_deg = _parameter_control(
    "Launch Azimuth (°)",
    key="launch_azimuth",
    minimum=0.0,
    maximum=359.0,
    default=0.0,
    step=1.0,
    format_string="%.0f",
)
wind_speed_m_s = _parameter_control(
    "Crosswind Speed (m/s)",
    key="crosswind_speed",
    minimum=0.0,
    maximum=50.0,
    default=0.0,
    step=0.5,
    format_string="%.1f",
)
st.sidebar.caption(
    "Crosswind is applied perpendicular to the selected launch azimuth; the "
    "standard-atmosphere jet wind remains enabled."
)
st.sidebar.subheader("Empirical validation")
uploaded_csv = st.sidebar.file_uploader(
    "Upload real flight telemetry",
    type=["csv"],
    help=(
        "CSV must contain Time (seconds) and Altitude (metres) columns. "
        "Velocity (m/s), X/Downrange (m), and Y/Crossrange (m) are optional. "
        "Time and altitude are rebased to the first valid sample."
    ),
)
st.sidebar.subheader("Monte Carlo analysis")
monte_carlo_enabled = st.sidebar.checkbox(
    "Enable landing dispersion analysis",
    value=False,
    help=(
        "Uses a fast vectorized point-mass approximation with randomized "
        "crosswind, motor thrust, and launch-angle deviations."
    ),
)
monte_carlo_iterations = (
    st.sidebar.slider(
        "Monte Carlo iterations",
        min_value=50,
        max_value=100,
        value=50,
        step=5,
        help="Each sample is advanced together using a lightweight fixed-step model.",
    )
    if monte_carlo_enabled
    else 50
)
run_monte_carlo = (
    st.sidebar.button("Run Monte Carlo Dispersion", width="stretch")
    if monte_carlo_enabled
    else False
)
telemetry: FlightTelemetry | None = None
if uploaded_csv is not None:
    try:
        telemetry = parse_flight_csv(uploaded_csv.getvalue())
        st.sidebar.success(
            f"Loaded {len(telemetry.time_s)} valid telemetry samples."
        )
        if (
            telemetry.downrange_m is None
            or telemetry.crossrange_m is None
        ):
            st.sidebar.caption(
                "No complete X/Y columns found; the 3D overlay will show altitude only."
            )
    except ValueError as error:
        st.sidebar.error(f"Could not read telemetry CSV: {error}")

specifications = {
    "dry_mass_kg": dry_mass_kg,
    "propellant_mass_kg": propellant_mass_kg,
    "burn_time_s": burn_time_s,
    "peak_thrust_N": peak_thrust_N,
    "diameter_m": diameter_m,
    "drag_coefficient": drag_coefficient,
    "launch_elevation_deg": launch_elevation_deg,
    "launch_azimuth_deg": launch_azimuth_deg,
    "wind_speed_m_s": wind_speed_m_s,
}
if thrust_curve is not None and thrust_curve.propellant_mass_kg is not None:
    specifications["propellant_mass_kg"] = thrust_curve.propellant_mass_kg
    specifications["burn_time_s"] = thrust_curve.burn_time_s
    specifications["peak_thrust_N"] = float(np.max(thrust_curve.thrust_n))

if st.sidebar.button(
    "Run Simulation",
    type="primary",
    width="stretch",
):
    st.session_state.pop("simulation_result", None)
    st.session_state.pop("monte_carlo_result", None)
    try:
        st.session_state["simulation_result"] = _run_simulation(
            specifications,
            thrust_curve,
        )
    except (
        ValueError,
        RuntimeError,
        FloatingPointError,
        np.linalg.LinAlgError,
    ) as error:
        st.error(f"Simulation could not be completed: {error}")

current_motor_signature = _motor_signature(thrust_curve)
if run_monte_carlo:
    try:
        with st.spinner("Computing dispersion ellipse..."):
            dispersion_result = run_monte_carlo_dispersion(
                _make_rocket(specifications, thrust_curve),
                iterations=monte_carlo_iterations,
                launch_elevation_deg=launch_elevation_deg,
                launch_azimuth_deg=launch_azimuth_deg,
                wind_speed_m_s=wind_speed_m_s,
            )
        st.session_state["monte_carlo_result"] = dispersion_result
        st.session_state["monte_carlo_specs"] = specifications.copy()
        st.session_state["monte_carlo_motor_signature"] = (
            current_motor_signature
        )
    except (
        ValueError,
        RuntimeError,
        FloatingPointError,
        np.linalg.LinAlgError,
    ) as error:
        st.error(f"Monte Carlo analysis could not be completed: {error}")

result = st.session_state.get("simulation_result")
dispersion_result = st.session_state.get("monte_carlo_result")
if (
    st.session_state.get("monte_carlo_specs") != specifications
    or st.session_state.get("monte_carlo_motor_signature")
    != current_motor_signature
):
    dispersion_result = None

st.subheader("Flight summary")
metric_columns = st.columns(6)
if result is None:
    for column, label in zip(
        metric_columns,
        (
            "Max Apogee",
            "Max Velocity",
            "Max Mach",
            "Max Q",
            "Peak Acceleration",
            "Total Flight Time",
        ),
    ):
        column.metric(label, "—")
    st.info(
        "Configure the rocket in the sidebar and run a simulation to view results."
    )
else:
    metrics = result["metrics"]
    metric_columns[0].metric(
        "Max Apogee",
        f"{metrics['apogee_m']:,.1f} m",
        delta=f"{metrics['apogee_m'] * 3.280839895:,.1f} ft",
        delta_color="off",
    )
    metric_columns[1].metric(
        "Max Velocity",
        f"{metrics['max_speed_m_s']:,.1f} m/s",
        delta="flight speed",
        delta_color="off",
    )
    metric_columns[2].metric(
        "Maximum Mach",
        f"{metrics['max_mach']:.2f}",
        delta=(
            "Transonic band reached"
            if metrics["transonic_band_reached"]
            else "Below Mach 0.8"
        ),
        delta_color="off",
    )
    metric_columns[3].metric(
        "Maximum Dynamic Pressure",
        f"{metrics['max_dynamic_pressure_pa']:,.0f} Pa",
    )
    metric_columns[4].metric(
        "Peak Acceleration",
        f"{metrics['peak_acceleration_g']:,.2f} G",
    )
    metric_columns[5].metric(
        "Total Flight Time",
        f"{metrics['flight_time_s']:,.2f} s",
    )
    st.caption("Transonic band is defined here as Mach 0.8–1.2.")

    st.subheader("Flight events")
    event_columns = st.columns(3)
    event_columns[0].metric(
        "Motor burnout",
        (
            f"{result['events']['burnout_s']:.3f} s"
            if result["events"]["burnout_s"] is not None
            else "Not reached"
        ),
    )
    event_columns[1].metric(
        "Apogee arrival",
        (
            f"{result['events']['apogee_s']:.3f} s"
            if result["events"]["apogee_s"] is not None
            else "Not detected"
        ),
    )
    event_columns[2].metric(
        "Impact / flight time",
        f"{metrics['flight_time_s']:.3f} s",
    )

    st.subheader("Longitudinal stability estimate")
    stability = result["stability"]
    stability_columns = st.columns(4)
    stability_columns[0].metric(
        "Static margin at launch",
        f"{stability['launch_margin_calibers']:.2f} calibers",
    )
    stability_columns[1].metric(
        "Static margin at burnout",
        f"{stability['burnout_margin_calibers']:.2f} calibers",
    )
    stability_columns[2].metric(
        "CG at launch",
        f"{stability['launch_cg_m']:.3f} m from nose",
    )
    stability_columns[3].metric(
        "CG at burnout",
        f"{stability['burnout_cg_m']:.3f} m from nose",
    )
    minimum_static_margin = min(
        stability["launch_margin_calibers"],
        stability["burnout_margin_calibers"],
    )
    if minimum_static_margin < 1.0:
        st.warning(
            f"Static margin is below the 1.0-caliber reference threshold "
            f"(minimum: {minimum_static_margin:.2f} calibers)."
        )
    else:
        st.success(
            f"Estimated static margin remains above the 1.0-caliber reference "
            f"threshold (minimum: {minimum_static_margin:.2f} calibers)."
        )
    st.caption(
        "Estimate assumes a uniform dry structure with CG at 0.45 body lengths, "
        "propellant CG at 0.80 body lengths, and CP at 0.67 body lengths. "
        "Validate CP and mass distribution with vehicle-specific data."
    )

    st.subheader("Chronological flight phases")
    burnout_time = result["events"]["burnout_s"]
    apogee_time = result["events"]["apogee_s"]
    impact_time = result["events"]["flight_time_s"]
    phase_rows = [
        {
            "Milestone": "Liftoff / rail clearance",
            "Time (s)": 0.0,
            "Details": (
                "Simulation starts in free flight; rail constraint is not modeled."
            ),
        },
        {
            "Milestone": "Motor burnout",
            "Time (s)": burnout_time,
            "Details": (
                "Propellant depletion event."
                if burnout_time is not None
                else "Burnout not reached before impact."
            ),
        },
        {
            "Milestone": "Apogee",
            "Time (s)": apogee_time,
            "Details": (
                f"Peak altitude {result['events']['apogee_altitude_m']:,.1f} m."
                if apogee_time is not None
                else "Apogee event not detected."
            ),
        },
        {
            "Milestone": "Descent",
            "Time (s)": apogee_time,
            "Details": (
                f"Begins at apogee; ground impact at {impact_time:.3f} s."
                if apogee_time is not None
                else f"Flight ends at ground impact at {impact_time:.3f} s."
            ),
        },
    ]
    st.dataframe(phase_rows, hide_index=True, width="stretch")

(
    trajectory_tab,
    spatial_tab,
    ground_station_tab,
    loads_tab,
    stability_tab,
    dispersion_tab,
    configuration_tab,
    validation_tab,
) = st.tabs(
    [
        "Trajectory",
        "3D Position & Drift",
        "Ground Station",
        "Flight loads",
        "Stability",
        "Monte Carlo dispersion",
        "Vehicle configuration",
        "Empirical Validation (Sim vs. Real)",
    ]
)

with trajectory_tab:
    if result is not None:
        trajectory_plot_key = "trajectory_inspection_plot"
        trajectory_selection = st.plotly_chart(
            _trajectory_chart(result, telemetry),
            width="stretch",
            key=trajectory_plot_key,
            on_select="rerun",
            selection_mode="points",
        )
        _set_playback_from_plot(
            trajectory_selection,
            plot_key=trajectory_plot_key,
            playback_key="ground_station_playback_time_s",
            flight_time_s=float(result["events"]["flight_time_s"]),
        )
    else:
        st.info(
            "The altitude and velocity profile will appear here after a simulation."
        )

with spatial_tab:
    if result is not None:
        st.caption(
            "Position and velocity use a launch-aligned inertial frame; attitude "
            "uses ZYX Euler angles with yaw relative to the launch azimuth. "
            "For readability, narrow axes are expanded visually; displayed "
            "coordinates and hover values remain in metres."
        )
        spatial_plot_key = "spatial_inspection_plot"
        spatial_selection = st.plotly_chart(
            _position_3d_chart(
                result,
                telemetry,
                equal_proportional_scale=True,
            ),
            width="stretch",
            key=spatial_plot_key,
            on_select="rerun",
            selection_mode="points",
        )
        _set_playback_from_plot(
            spatial_selection,
            plot_key=spatial_plot_key,
            playback_key="ground_station_playback_time_s",
            flight_time_s=float(result["events"]["flight_time_s"]),
        )
        position_chart, velocity_chart, attitude_chart = (
            _spatial_timeseries_charts(result)
        )
        for chart_key, chart in (
            ("position_timeseries_inspection", position_chart),
            ("velocity_timeseries_inspection", velocity_chart),
            ("attitude_timeseries_inspection", attitude_chart),
        ):
            chart_selection = st.plotly_chart(
                chart,
                width="stretch",
                key=chart_key,
                on_select="rerun",
                selection_mode="points",
            )
            _set_playback_from_plot(
                chart_selection,
                plot_key=chart_key,
                playback_key="ground_station_playback_time_s",
                flight_time_s=float(result["events"]["flight_time_s"]),
            )
    else:
        st.info(
            "The 3D flight path and spatial component profiles will appear here "
            "after a simulation."
        )

with ground_station_tab:
    if result is None:
        st.info(
            "Run a simulation to scrub through the flight and inspect live telemetry."
        )
    else:
        flight_time_s = float(result["events"]["flight_time_s"])
        playback_key = "ground_station_playback_time_s"
        st.subheader("Flight replay")
        st.caption(
            "The enlarged rocket model follows the X/Y/Z path with attached red, "
            "green, and blue body-axis vectors (coordinates remain in metres). "
            "Narrow axes are expanded visually for readability; drag to rotate the "
            "view at any time, including during replay. "
            "Attitude is estimated; simplified aerodynamics, rotational damping, "
            "rail restraint, and recovery systems are not fully modeled."
        )
        playback_speed = st.selectbox(
            "Playback speed",
            options=(0.5, 1.0, 2.0, 5.0),
            index=1,
            format_func=lambda speed: f"{speed:g}×",
            key="ground_station_playback_speed",
        )
        _render_ground_station_animation(
            _position_3d_chart(
                result,
                telemetry,
                selected_time_s=float(
                    st.session_state.get(playback_key, 0.0)
                ),
                animated=True,
                playback_speed=float(playback_speed),
                equal_proportional_scale=True,
            )
        )
        _render_ground_station_inspector(
            result,
            flight_time_s,
            playback_key,
        )

with loads_tab:
    if result is not None:
        st.plotly_chart(_loads_chart(result), width="stretch")
    else:
        st.info(
            "Acceleration and dynamic pressure profiles will appear here after a simulation."
        )

with stability_tab:
    if result is not None:
        st.caption(
            "CG/CP locations and static margin are engineering estimates based on "
            "the simplified geometry assumptions shown in the flight summary."
        )
        st.plotly_chart(_stability_chart(result), width="stretch")
    else:
        st.info(
            "Run a simulation to view the estimated CG and static-margin history."
        )

with dispersion_tab:
    if not monte_carlo_enabled:
        st.info(
            "Enable Monte Carlo analysis in the sidebar to calculate a landing footprint."
        )
    elif dispersion_result is None:
        st.info(
            "Start Monte Carlo Analysis from the sidebar to calculate the landing footprint."
        )
    else:
        st.caption(
            f"{len(dispersion_result['landing_x_m'])} trajectories; sampled wind "
            "magnitude/direction, motor thrust, and launch angles using the "
            "fast point-mass approximation. The dashed circle "
            "encloses the 95th percentile of radial landing distance from the mean."
        )
        dispersion_columns = st.columns(4)
        landing_radial_spread_m = np.hypot(
            dispersion_result["landing_x_m"]
            - np.mean(dispersion_result["landing_x_m"]),
            dispersion_result["landing_y_m"]
            - np.mean(dispersion_result["landing_y_m"]),
        )
        dispersion_columns[0].metric(
            "Mean downrange",
            f"{np.mean(dispersion_result['landing_x_m']):,.1f} m",
        )
        dispersion_columns[1].metric(
            "Mean crossrange",
            f"{np.mean(dispersion_result['landing_y_m']):,.1f} m",
        )
        dispersion_columns[2].metric(
            "95th-percentile radial spread",
            f"{np.quantile(landing_radial_spread_m, 0.95):,.1f} m",
        )
        dispersion_columns[3].metric(
            "Apogee range",
            f"{np.min(dispersion_result['apogee_m']):,.0f}–"
            f"{np.max(dispersion_result['apogee_m']):,.0f} m",
        )
        st.plotly_chart(
            _dispersion_chart(dispersion_result, result),
            width="stretch",
        )

with configuration_tab:
    if result is not None:
        configuration = {
            "vehicle": result["specs"],
            "simulation": {
                "model": "6-DOF rigid-body dynamics",
                "integrator": "SciPy RK45",
                "thrust_model": (
                    "RASP/RockSim piecewise-linear motor curve"
                    if thrust_curve is not None
                    else "constant at configured thrust until burnout"
                ),
                "motor_designation": (
                    thrust_curve.designation
                    if thrust_curve is not None
                    else None
                ),
                "motor_total_impulse_Ns": (
                    thrust_curve.total_impulse_ns
                    if thrust_curve is not None
                    else None
                ),
                "wind_model": (
                    "altitude-dependent wind with configurable crosswind"
                ),
                "crosswind_speed_m_s": result["specs"]["wind_speed_m_s"],
                "display_frame": "inertial axes rotated to launch azimuth",
                "launch_elevation_deg": result["specs"]["launch_elevation_deg"],
                "launch_azimuth_deg": result["specs"]["launch_azimuth_deg"],
                "flight_time_s": result["metrics"]["flight_time_s"],
            },
        }
        st.code(json.dumps(configuration, indent=2), language="json")
    else:
        st.info(
            "The configuration summary will appear here after a simulation."
        )

with validation_tab:
    if telemetry is None:
        st.info(
            "Upload a CSV in the sidebar to compare real flight telemetry with the simulation."
        )
    elif result is None:
        st.info(
            "Run a simulation to calculate real-versus-simulated flight errors."
        )
    else:
        try:
            comparison = _validation_metrics(result, telemetry)
            st.subheader("Apogee comparison")
            validation_columns = st.columns(4)
            validation_columns[0].metric(
                "Simulated apogee",
                f"{comparison.simulated_apogee_m:,.1f} m",
            )
            validation_columns[1].metric(
                "Real apogee",
                f"{comparison.real_apogee_m:,.1f} m",
            )
            validation_columns[2].metric(
                "Real − simulated",
                f"{comparison.apogee_delta_m:+,.1f} m",
                delta=(
                    f"{comparison.apogee_delta_percent:+.2f}%"
                    if comparison.apogee_delta_percent is not None
                    else "percentage unavailable"
                ),
                delta_color="off",
            )
            validation_columns[3].metric(
                "Altitude RMSE",
                f"{comparison.altitude_rmse_m:,.1f} m",
            )
            st.caption(
                f"Altitude MAE: {comparison.altitude_mae_m:,.1f} m · "
                f"{comparison.compared_samples} overlapping telemetry samples. "
                "Apogee percent delta is relative to simulated apogee; altitude errors "
                "are evaluated at real-flight sample times over the shared interval."
            )
            st.plotly_chart(
                _trajectory_chart(result, telemetry),
                width="stretch",
            )
        except ValueError as error:
            st.warning(f"Could not compare these flights: {error}")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Developed by Tegan Broody | Ridley High School | Aerospace Engineering Portfolio"
)
