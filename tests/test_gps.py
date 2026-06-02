"""WGS84-exact GPS conversion via pymap3d."""
import math

import pymap3d as pm

from targetgeo.gps import offset_to_target_gps


def test_zero_offset_returns_camera_gps():
    lat, lon, alt = offset_to_target_gps(
        east_m=0.0, north_m=0.0, up_m=0.0,
        cam_lat=37.5, cam_lon=127.0, cam_alt_m=100.0,
    )
    assert abs(lat - 37.5) < 1e-9
    assert abs(lon - 127.0) < 1e-9
    assert abs(alt - 100.0) < 1e-6


def test_east_offset_moves_longitude_east():
    cam_lat = 37.5
    cam_lon = 127.0
    cam_alt = 100.0
    lat, lon, alt = offset_to_target_gps(
        east_m=111.0, north_m=0.0, up_m=0.0,
        cam_lat=cam_lat, cam_lon=cam_lon, cam_alt_m=cam_alt,
    )
    # 111 m east at lat 37.5° ≈ 0.00126° lon (≈ 111 m / (R*cos(lat)) * 180/π)
    assert lon > cam_lon
    assert abs(lat - cam_lat) < 1e-4  # mostly unchanged
    assert abs(alt - cam_alt) < 1e-3  # ellipsoid curvature causes small alt change


def test_north_offset_moves_latitude_north():
    lat, lon, alt = offset_to_target_gps(
        east_m=0.0, north_m=111.0, up_m=0.0,
        cam_lat=37.5, cam_lon=127.0, cam_alt_m=100.0,
    )
    assert lat > 37.5
    assert abs(lon - 127.0) < 1e-4


def test_matches_pymap3d_direct():
    """Verify our wrapper just delegates to pm.enu2geodetic correctly."""
    args = (123.4, -56.7, 8.9, 37.5, 127.0, 100.0)
    expected = pm.enu2geodetic(*args)
    actual = offset_to_target_gps(
        east_m=args[0], north_m=args[1], up_m=args[2],
        cam_lat=args[3], cam_lon=args[4], cam_alt_m=args[5],
    )
    for a, e in zip(actual, expected):
        assert abs(a - e) < 1e-12


def test_round_trip_via_pymap3d():
    """Apply offset then reverse — should recover origin within numerical precision."""
    cam = (37.5, 127.0, 100.0)
    east, north, up = 250.0, -180.0, 30.0
    lat, lon, alt = offset_to_target_gps(east, north, up, *cam)
    e2, n2, u2 = pm.geodetic2enu(lat, lon, alt, *cam)
    assert abs(e2 - east) < 1e-6
    assert abs(n2 - north) < 1e-6
    assert abs(u2 - up) < 1e-6
