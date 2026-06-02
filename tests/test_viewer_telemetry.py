import numpy as np

from targetgeo.viewer.telemetry import load_telemetry, build_state


def test_load_telemetry_parses_rows(tmp_path):
    csv = tmp_path / "tele.csv"
    csv.write_text(
        "frame,lat,lon,alt_m,pitch,yaw,roll\n"
        "0,37.5,127.0,80.0,-30.0,45.0,0.0\n"
        "10,37.6,127.1,81.0,-20.0,40.0,1.0\n"
    )
    table = load_telemetry(str(csv))
    assert set(table.keys()) == {0, 10}
    assert table[0]["lat"] == 37.5
    assert table[10]["pyr"] == (-20.0, 40.0, 1.0)


def test_build_state_makes_gps_drone_state():
    row = {"lat": 37.5, "lon": 127.0, "alt_m": 80.0, "pyr": (-30.0, 45.0, 0.0)}
    K = np.eye(3)
    state = build_state(row, K)
    assert state.camera_lat == 37.5
    assert state.camera_pyr_deg == (-30.0, 45.0, 0.0)
    assert state.K is K
