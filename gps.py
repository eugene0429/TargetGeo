"""WGS84-exact GPS conversion using pymap3d.

Camera-as-reference design: each frame's camera GPS is the local ENU origin.
No fixed reference point. No plane approximation.
"""

from __future__ import annotations

import pymap3d as pm


def offset_to_target_gps(
    east_m: float, north_m: float, up_m: float,
    cam_lat: float, cam_lon: float, cam_alt_m: float,
) -> tuple[float, float, float]:
    """Camera GPS + ENU offset → target GPS (WGS84 ellipsoid exact).

    Args:
        east_m, north_m, up_m: target offset from camera, in local ENU meters.
        cam_lat, cam_lon, cam_alt_m: camera absolute WGS84 position.

    Returns:
        (target_lat, target_lon, target_alt_m).
    """
    return pm.enu2geodetic(east_m, north_m, up_m, cam_lat, cam_lon, cam_alt_m)
