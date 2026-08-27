"""
Tests for statistical conductance and resistance generators.
"""

import numpy as np
import pytest

from network_rheology.distributions import generate_conductances, generate_resistances


def test_lognormal_moments():
    """Verify lognormal generator closely matches theoretical mean and standard deviation."""
    target_mean = 10.0
    target_std = 2.0
    size = 200_000
    samples = generate_conductances(mean=target_mean, std=target_std, size=size, dist="lognormal", rng=42)

    sample_mean = np.mean(samples)
    sample_std = np.std(samples)

    import math
    assert math.isclose(sample_mean, target_mean, rel_tol=0.02)
    assert math.isclose(sample_std, target_std, rel_tol=0.05)
    assert np.all(samples > 0)


def test_hotspot_injection():
    """Verify hotspot fraction correctly replaces designated proportion of edges with hotspot conductances."""
    size = 1000
    hotspot_frac = 0.1  # 10%
    hotspot_val = 1e6

    samples = generate_conductances(
        mean=1.0,
        std=0.1,
        size=size,
        dist="constant",
        hotspot_frac=hotspot_frac,
        hotspot_val=hotspot_val,
        rng=42,
    )

    hotspot_count = np.sum(np.isclose(samples, hotspot_val))
    assert hotspot_count == 100


def test_truncated_normal():
    """Verify truncated normal clips negative values at min_val."""
    samples = generate_conductances(mean=1.0, std=5.0, size=5000, dist="truncated_normal", min_val=1e-6, rng=42)
    assert np.all(samples >= 1e-6)
