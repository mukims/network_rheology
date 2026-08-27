"""
Statistical distributions for conductances, resistances, and contact properties.
"""

from typing import Optional, Union
import numpy as np


def generate_conductances(
    mean: float,
    std: float,
    size: int,
    dist: str = "lognormal",
    min_val: float = 1e-12,
    hotspot_frac: float = 0.0,
    hotspot_val: float = 1e4,
    rng: Optional[Union[np.random.Generator, int]] = None,
) -> np.ndarray:
    """
    Generate non-negative random conductance values according to a specified distribution.

    Parameters
    ----------
    mean : float
        Target mean of the distribution (must be > 0).
    std : float
        Target standard deviation / disorder parameter (>= 0).
    size : int
        Number of random values to sample.
    dist : str, optional
        Distribution type:
        - "lognormal": Log-normal distribution with exact mean and std (recommended for physical disorder).
        - "truncated_normal": Gaussian distribution clipped at min_val.
        - "folded_normal": Absolute value of Gaussian |N(mean, std)| (legacy mode).
        - "uniform": Uniform distribution on [max(min_val, mean - std*sqrt(3)), mean + std*sqrt(3)].
        - "constant": Deterministic conductances equal to mean.
    min_val : float, optional
        Minimum conductance floor to prevent singular zero conductances, default 1e-12.
    hotspot_frac : float, optional
        Fraction of edges designated as high-conductance hotspots, default 0.0.
    hotspot_val : float, optional
        Conductance value for hotspot edges, default 1e4.
    rng : Optional[Union[np.random.Generator, int]], optional
        NumPy random Generator instance or integer seed.

    Returns
    -------
    np.ndarray
        Array of strictly positive conductance values of length `size`.
    """
    if isinstance(rng, (int, np.integer)):
        gen = np.random.default_rng(rng)
    elif isinstance(rng, np.random.Generator):
        gen = rng
    else:
        gen = np.random.default_rng()

    if size <= 0:
        return np.empty(0, dtype=np.float64)

    if std <= 0.0 or dist == "constant":
        values = np.full(size, max(mean, min_val), dtype=np.float64)
    elif dist == "lognormal":
        # Match moments of lognormal: mean = exp(mu + sigma^2 / 2), var = (exp(sigma^2) - 1) * mean^2
        variance = std**2
        sigma_sq = np.log(1.0 + variance / (mean**2))
        sigma = np.sqrt(sigma_sq)
        mu = np.log(mean) - 0.5 * sigma_sq
        values = gen.lognormal(mean=mu, sigma=sigma, size=size)
    elif dist == "truncated_normal":
        values = gen.normal(loc=mean, scale=std, size=size)
        values = np.clip(values, a_min=min_val, a_max=None)
    elif dist == "folded_normal":
        values = np.abs(gen.normal(loc=mean, scale=std, size=size))
        values = np.clip(values, a_min=min_val, a_max=None)
    elif dist == "uniform":
        half_width = std * np.sqrt(3.0)
        low = max(min_val, mean - half_width)
        high = mean + half_width
        values = gen.uniform(low=low, high=high, size=size)
    else:
        raise ValueError(f"Unknown distribution '{dist}'. Supported: 'lognormal', 'truncated_normal', 'folded_normal', 'uniform', 'constant'.")

    # Apply hotspot fraction if requested
    if hotspot_frac > 0.0 and size > 0:
        num_hotspots = int(np.round(hotspot_frac * size))
        if num_hotspots > 0:
            hotspot_indices = gen.choice(size, num_hotspots, replace=False)
            values[hotspot_indices] = hotspot_val

    return np.ascontiguousarray(values, dtype=np.float64)


def generate_resistances(
    mean_r: float,
    std_r: float,
    size: int,
    dist: str = "lognormal",
    min_r: float = 1e-12,
    hotspot_frac: float = 0.0,
    hotspot_r: float = 1e-4,
    rng: Optional[Union[np.random.Generator, int]] = None,
) -> np.ndarray:
    """
    Generate non-negative random resistance values according to a specified distribution.

    Parameters
    ----------
    mean_r : float
        Target mean resistance (Ohm).
    std_r : float
        Standard deviation of resistance.
    size : int
        Number of values to sample.
    dist : str, optional
        Distribution type ('lognormal', 'truncated_normal', 'folded_normal', 'uniform', 'constant').
    min_r : float, optional
        Minimum resistance floor.
    hotspot_frac : float, optional
        Fraction of edges designated as low-resistance hotspots.
    hotspot_r : float, optional
        Resistance value for hotspot links (Ohm).
    rng : Optional[Union[np.random.Generator, int]], optional
        NumPy random Generator instance or integer seed.

    Returns
    -------
    np.ndarray
        Array of resistance values of length `size`.
    """
    return generate_conductances(
        mean=mean_r,
        std=std_r,
        size=size,
        dist=dist,
        min_val=min_r,
        hotspot_frac=hotspot_frac,
        hotspot_val=hotspot_r,
        rng=rng,
    )
