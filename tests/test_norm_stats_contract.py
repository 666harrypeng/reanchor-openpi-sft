"""Norm-stat post-processing for the 8-D absolute-pose contract (CPU, no openpi).

π0.5 normalizes with 1 % / 99 % quantiles. The expert's orientation is a constant top-down
quaternion except for each episode's first row (the planner's hold at the current pose), so qw
and qz have q01 == q99 == 0 and those rows would be scaled by ~1e5 (the smoke run of 2026-11-03
logged a loss of 6e6). Unit-quaternion components already live in [-1, 1]: they use the identity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from reanchor_sft import contract
from reanchor_sft.training import identity_quaternion_norm, normalized_extremes


@dataclass
class _Stats:  # the four fields of openpi's NormStats
    mean: np.ndarray
    std: np.ndarray
    q01: np.ndarray
    q99: np.ndarray


def _degenerate_stats() -> _Stats:
    mean = np.array([0.35, -0.11, 0.28, 0.00004, 0.976, 0.029, 0.002, -0.31])
    std = np.array([0.19, 0.23, 0.065, 0.00097, 0.026, 0.21, 0.027, 0.95])
    q01 = np.array([0.0, -0.40, 0.148, -4e-5, 0.896, -0.344, 0.0, -1.0])
    q99 = np.array([0.592, 0.381, 0.361, -1.6e-6, 0.99998, 0.434, 0.0, 0.9996])
    return _Stats(mean, std, q01, q99)


def _quantile_normalize(x, s):
    return (x - s.q01) / (s.q99 - s.q01 + 1e-6) * 2.0 - 1.0


def test_quaternion_dims_normalize_by_the_identity():
    fixed = identity_quaternion_norm(_degenerate_stats())
    q = contract.QUATERNION_DIMS
    assert q == slice(3, 7)
    np.testing.assert_array_equal(fixed.q01[q], -1.0)
    np.testing.assert_array_equal(fixed.q99[q], 1.0)
    np.testing.assert_array_equal(fixed.mean[q], 0.0)
    np.testing.assert_array_equal(fixed.std[q], 1.0)
    hold_row = np.array([0.3, 0.0, 0.25, 0.0007, 0.9228, 0.0483, 0.3822, 1.0])
    normalized = _quantile_normalize(hold_row, fixed)
    np.testing.assert_allclose(normalized[q], hold_row[q], atol=1e-5)


def test_position_and_gripper_dims_are_untouched():
    before = _degenerate_stats()
    fixed = identity_quaternion_norm(before)
    for i in (0, 1, 2, 7):
        assert (fixed.mean[i], fixed.std[i], fixed.q01[i], fixed.q99[i]) == (
            before.mean[i],
            before.std[i],
            before.q01[i],
            before.q99[i],
        )
    assert before.q01[6] == before.q99[6] == 0.0  # the input is not modified in place


def test_normalized_extremes_reject_the_blow_up():
    lo = np.array([0.0, -0.4, 0.148, -0.0143, 0.886, -0.443, 0.0, -1.0])
    hi = np.array([0.5998, 0.395, 0.418, 0.0334, 1.0, 0.464, 0.4193, 1.0])
    with pytest.raises(ValueError, match="qz"):
        normalized_extremes(_degenerate_stats(), lo, hi, names=contract.STATE_NAMES)
    extremes = normalized_extremes(
        identity_quaternion_norm(_degenerate_stats()), lo, hi, names=contract.STATE_NAMES
    )
    # z: (0.418 - 0.148) / 0.213 * 2 - 1
    assert extremes["max_abs"] == pytest.approx(1.54, abs=0.01)
    assert extremes["qz"] == pytest.approx((0.0, 0.4193), abs=1e-4)
