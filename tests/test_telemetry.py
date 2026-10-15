import numpy as np
import pytest

from src.telemetry import compare_telemetry_to_simulation, parse_flight_csv


def test_parse_rebases_sorts_deduplicates_and_reads_optional_columns() -> None:
    telemetry = parse_flight_csv(
        b"\xef\xbb\xbfTime (seconds),Altitude (m),Velocity (m/s),X (m),Y (m)\n"
        b"2,130,4,20,8\n"
        b"0,100,0,0,0\n"
        b"1,110,2,10,4\n"
        b"1,112,4,12,6\n"
        b"bad,150,5,25,10\n"
    )

    np.testing.assert_allclose(telemetry.time_s, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(telemetry.altitude_m, [0.0, 11.0, 30.0])
    np.testing.assert_allclose(telemetry.velocity_m_s, [0.0, 3.0, 4.0])
    np.testing.assert_allclose(telemetry.downrange_m, [0.0, 11.0, 20.0])
    np.testing.assert_allclose(telemetry.crossrange_m, [0.0, 5.0, 8.0])


def test_parse_derives_vertical_velocity_and_allows_altimeter_only_csv() -> None:
    telemetry = parse_flight_csv(
        "time,altitude\n"
        "5,50\n"
        "6,60\n"
        "7,50\n"
    )

    np.testing.assert_allclose(telemetry.time_s, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(telemetry.altitude_m, [0.0, 10.0, 0.0])
    np.testing.assert_allclose(telemetry.velocity_m_s, [10.0, 0.0, -10.0])
    assert telemetry.downrange_m is None
    assert telemetry.crossrange_m is None


def test_blank_optional_measurement_does_not_drop_altimeter_sample() -> None:
    telemetry = parse_flight_csv(
        "Time,Altitude,Velocity\n"
        "0,0,0\n"
        "1,10,\n"
        "2,0,-10\n"
    )

    np.testing.assert_allclose(telemetry.time_s, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(telemetry.altitude_m, [0.0, 10.0, 0.0])
    np.testing.assert_allclose(
    telemetry.velocity_m_s,
    [0.0, 0.0, -10.0],
    )

@pytest.mark.parametrize(
    ("csv_content", "message"),
    [
        ("", "header"),
        ("time,velocity\n0,1\n1,2\n", "Time"),
        ("time,altitude\n0,1\nbad,2\n", "two valid"),
        ("time,altitude\n1,1\n1,2\n", "distinct"),
    ],
)
def test_parse_rejects_missing_or_insufficient_samples(
    csv_content: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_flight_csv(csv_content)


def test_compare_telemetry_interpolates_simulated_altitude() -> None:
    telemetry = parse_flight_csv(
        "Time,Altitude\n0,0\n0.5,5\n1.5,5\n2,0\n"
    )

    comparison = compare_telemetry_to_simulation(
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 10.0, 0.0]),
        telemetry,
    )

    assert comparison.simulated_apogee_m == 10.0
    assert comparison.real_apogee_m == 5.0
    assert comparison.apogee_delta_m == -5.0
    assert comparison.apogee_delta_percent == -50.0
    assert comparison.altitude_rmse_m == 0.0
    assert comparison.compared_samples == 4


def test_compare_rejects_non_overlapping_flights() -> None:
    telemetry = parse_flight_csv("time,altitude\n10,0\n11,5\n")

    with pytest.raises(ValueError, match="no overlapping"):
        compare_telemetry_to_simulation(
            np.array([2.0, 3.0]),
            np.array([0.0, 1.0]),
            telemetry,
        )
