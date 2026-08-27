"""
Parameter optimization and impedance curve fitting for electrical impedance spectroscopy.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import minimize


def compute_impedance_loss(
    sim_z: np.ndarray,
    target_z: np.ndarray,
    weights: Tuple[float, float] = (1.0, 1.0),
) -> float:
    """
    Calculate normalized error between simulated and target complex impedance spectra.

    Parameters
    ----------
    sim_z : np.ndarray
        Simulated complex impedance array.
    target_z : np.ndarray
        Target / experimental complex impedance array.
    weights : Tuple[float, float], optional
        Weights (weight_mag, weight_phase) for magnitude and phase loss terms.

    Returns
    -------
    float
        Total loss value.
    """
    sim_mag = np.abs(sim_z)
    target_mag = np.abs(target_z)

    # Avoid log(0)
    eps = 1e-12
    log_mag_loss = np.mean((np.log10(np.maximum(sim_mag, eps)) - np.log10(np.maximum(target_mag, eps))) ** 2)

    sim_phase = np.angle(sim_z)
    target_phase = np.angle(target_z)
    phase_loss = np.mean((sim_phase - target_phase) ** 2)

    return float(weights[0] * log_mag_loss + weights[1] * phase_loss)


def fit_equivalent_circuit_rc(
    frequencies: np.ndarray,
    target_z: np.ndarray,
    initial_guess: Tuple[float, float, float] = (1e3, 1e4, 1e-9),
    bounds: Optional[List[Tuple[float, float]]] = None,
) -> Dict[str, Union[float, np.ndarray, bool]]:
    """
    Fit an equivalent circuit model (e.g. series nanosheet resistance R_s in series with
    parallel junction resistance R_j and capacitance C_j: Z = R_s + R_j / (1 + i*w*R_j*C_j))
    to experimental impedance data.

    Parameters
    ----------
    frequencies : np.ndarray
        Array of frequencies (Hz).
    target_z : np.ndarray
        Array of target complex impedance values (Ohm).
    initial_guess : Tuple[float, float, float], optional
        Initial parameters [R_s, R_j, C_j].
    bounds : Optional[List[Tuple[float, float]]], optional
        Parameter bounds [(R_s_min, R_s_max), (R_j_min, R_j_max), (C_j_min, C_j_max)].

    Returns
    -------
    Dict[str, Union[float, np.ndarray, bool]]
        Dictionary containing optimized parameters and fitted spectrum.
    """
    freqs = np.asarray(frequencies, dtype=np.float64)
    target = np.asarray(target_z, dtype=np.complex128)
    omega = 2.0 * np.pi * freqs

    if bounds is None:
        bounds = [(1e-3, 1e9), (1e-3, 1e9), (1e-15, 1e-3)]

    # Optimize in log10 space for numerical stability across orders of magnitude
    log_init = np.log10(np.maximum(initial_guess, 1e-15))
    log_bounds = [(np.log10(b[0]), np.log10(b[1])) for b in bounds]

    def objective(log_params):
        r_s = 10.0 ** log_params[0]
        r_j = 10.0 ** log_params[1]
        c_j = 10.0 ** log_params[2]

        # Model: Z = R_s + R_j / (1 + i*w*R_j*C_j)
        denom = 1.0 + 1j * omega * r_j * c_j
        z_model = r_s + r_j / denom
        return compute_impedance_loss(z_model, target)

    res = minimize(
        objective,
        x0=log_init,
        bounds=log_bounds,
        method="L-BFGS-B",
    )

    opt_rs = float(10.0 ** res.x[0])
    opt_rj = float(10.0 ** res.x[1])
    opt_cj = float(10.0 ** res.x[2])

    fitted_z = opt_rs + opt_rj / (1.0 + 1j * omega * opt_rj * opt_cj)

    return {
        "success": bool(res.success),
        "R_sheet": opt_rs,
        "R_junction": opt_rj,
        "C_junction": opt_cj,
        "loss": float(res.fun),
        "fitted_Z": fitted_z,
        "message": str(res.message),
    }
