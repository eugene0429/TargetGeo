"""Verify vendored disambiguate.py works."""
import numpy as np

from targetgeo.disambiguate import disambiguate_visibility, DisambiguationResult


def test_disambiguate_picks_more_negative_nz_when_no_world_up():
    c1 = np.array([0.0, 0.0, 10.0]); n1 = np.array([0.0, 0.0, -1.0])
    c2 = np.array([0.0, 0.0, 10.0]); n2 = np.array([0.0, 0.0, +0.5])
    res = disambiguate_visibility([(c1, n1), (c2, n2)])
    assert isinstance(res, DisambiguationResult)
    assert res.chosen_idx == 0  # more negative nz wins
    assert res.method in ("visibility", "fallback")


def test_disambiguate_world_up_axis_picks_aligned_normal():
    # Flat target: true normal = world up. World up in cam frame = (0, -1, 0)
    # (typical when drone looks down).
    world_up = np.array([0.0, -1.0, 0.0])
    c1 = np.array([0.0, 0.0, 10.0]); n1 = np.array([0.0, -1.0, 0.0])  # aligned (score=1.0)
    c2 = np.array([0.0, 0.0, 10.0]); n2 = np.array([0.707, 0.707, 0.0])  # oblique (score~0.707)
    res = disambiguate_visibility([(c1, n1), (c2, n2)], world_up_cv=world_up)
    assert res.chosen_idx == 0
    assert res.method == "world_up_axis"
