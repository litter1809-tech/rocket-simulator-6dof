import numpy as np
import pytest

import src.dispersion as dispersion_module
from src.dispersion import run_monte_carlo_dispersion
from src.vehicle import Rocket


def test_monte_carlo_vectorizes_samples_without_calling_full_solver(monkeypatch) -> None:
    rocket = Rocket(
        dry_mass=12.0,
        propellant_mass=3.0,
        burn_time=5.0,
        average_thrust=1200.0,
        reference_area=np.pi * 0.05**2,
        drag_coefficient=0.5,
        moment_of_inertia=8.0,
    )

    def fail_if_full_solver_used(*_args, **_kwargs):
        raise AssertionError("Monte Carlo must not call the full RK45 solver.")

    monkeypatch.setattr(
        dispersion_module,
        "simulate_rocket",
        fail_if_full_solver_used,
        raising=False,
    )
    dispersion = run_monte_carlo_dispersion(
        rocket,
        iterations=50,
        launch_elevation_deg=90.0,
        launch_azimuth_deg=0.0,
        wind_speed_m_s=3.0,
    )

    assert dispersion["landing_x_m"].shape == (50,)
    assert dispersion["landing_y_m"].shape == (50,)
    assert np.all(np.isfinite(dispersion["landing_x_m"]))
    assert np.all(np.isfinite(dispersion["landing_y_m"]))
    assert np.all(np.isfinite(dispersion["apogee_m"]))
    assert np.all(np.isfinite(dispersion["flight_time_s"]))
    assert np.std(dispersion["thrust_scale"]) > 0.0
    assert np.std(dispersion["launch_elevation_deg"]) > 0.0
    assert np.std(dispersion["launch_azimuth_deg"]) > 0.0

    repeat = run_monte_carlo_dispersion(
        rocket,
        iterations=50,
        launch_elevation_deg=90.0,
        launch_azimuth_deg=0.0,
        wind_speed_m_s=3.0,
    )
    np.testing.assert_array_equal(
        dispersion["landing_x_m"],
        repeat["landing_x_m"],
    )
    np.testing.assert_array_equal(
        dispersion["landing_y_m"],
        repeat["landing_y_m"],
    )


def test_monte_carlo_rejects_iteration_counts_outside_sidebar_range() -> None:
    rocket = Rocket(
        dry_mass=1.0,
        propellant_mass=0.1,
        burn_time=2.0,
        average_thrust=50.0,
        reference_area=0.001,
        drag_coefficient=0.5,
        moment_of_inertia=0.5,
    )
    with pytest.raises(ValueError, match="between 50 and 100"):
        run_monte_carlo_dispersion(
            rocket,
            iterations=49,
            launch_elevation_deg=90.0,
            launch_azimuth_deg=0.0,
            wind_speed_m_s=3.0,
        )
