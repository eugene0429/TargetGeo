import numpy as np
import pytest

from targetgeo.viewer.intrinsics import build_K


def test_build_K_from_explicit_fx_fy():
    K = build_K(1920, 1080, fx=1500.0, fy=1500.0, cx=960.0, cy=540.0)
    assert K.shape == (3, 3)
    assert K[0, 0] == 1500.0 and K[1, 1] == 1500.0
    assert K[0, 2] == 960.0 and K[1, 2] == 540.0
    assert K[2, 2] == 1.0


def test_build_K_explicit_defaults_principal_point_to_center():
    K = build_K(640, 480, fx=600.0, fy=600.0)
    assert K[0, 2] == 320.0 and K[1, 2] == 240.0


def test_build_K_from_hfov():
    # hfov=90 deg, width=1920 -> f = (1920/2)/tan(45deg) = 960
    K = build_K(1920, 1080, hfov_deg=90.0)
    assert K[0, 0] == pytest.approx(960.0, rel=1e-6)
    assert K[1, 1] == pytest.approx(960.0, rel=1e-6)
    assert K[0, 2] == 960.0 and K[1, 2] == 540.0


def test_build_K_defaults_to_60deg_when_nothing_given():
    K = build_K(1000, 1000)
    expected_f = (1000 / 2.0) / np.tan(np.deg2rad(60.0) / 2.0)
    assert K[0, 0] == pytest.approx(expected_f, rel=1e-6)
