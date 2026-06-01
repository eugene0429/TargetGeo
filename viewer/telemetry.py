"""Optional per-frame telemetry loader.

CSV schema (header required):
    frame,lat,lon,alt_m,pitch,yaw,roll

Returns a dict keyed by integer frame index. K is supplied separately by the
caller (it is a fixed calibration, not per-frame telemetry).
"""

from __future__ import annotations

import csv

from seg_pose.pose_types import DroneStateGps


def load_telemetry(path: str) -> dict[int, dict]:
    table: dict[int, dict] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(float(row["frame"]))
            table[idx] = {
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "alt_m": float(row["alt_m"]),
                "pyr": (float(row["pitch"]), float(row["yaw"]), float(row["roll"])),
            }
    return table


def build_state(row: dict, K) -> DroneStateGps:
    """Construct a DroneStateGps from a telemetry row + intrinsics K."""
    return DroneStateGps(
        camera_lat=row["lat"], camera_lon=row["lon"], camera_alt_m=row["alt_m"],
        camera_pyr_deg=row["pyr"], K=K,
    )
