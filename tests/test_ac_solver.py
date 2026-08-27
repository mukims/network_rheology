"""
Tests for AC complex impedance and electrical impedance spectroscopy (EIS) solver.
"""

import math
import networkx as nx
import numpy as np
import pytest

from network_rheology.solvers.ac_solver import (
    solve_complex_impedance,
    solve_complex_impedance_spectrum,
)


def test_parallel_rc_circuit():
    """
    Test 2-node parallel RC circuit:
    Admittance Y(w) = G + i*w*C = 1/R + i*w*C
    Impedance Z(w) = 1 / Y(w) = R / (1 + i*w*R*C)
    """
    R = 1000.0  # 1 kOhm
    C = 1e-6    # 1 uF
    G_val = 1.0 / R
    tau = R * C # 1 ms, characteristic angular frequency w_c = 1/tau = 1000 rad/s -> f_c = 1000 / (2*pi) ~ 159.155 Hz

    G = nx.Graph()
    G.add_edge(0, 1)

    f_c = 1.0 / (2.0 * np.pi * R * C)

    # 1. Low frequency limit: f -> 0, Z -> R
    z_low = solve_complex_impedance(
        graph=G,
        node_a=0,
        node_b=1,
        frequency=1e-3,
        conductances=np.array([G_val]),
        capacitances=np.array([C]),
    )
    assert math.isclose(z_low.real, R, rel_tol=1e-3)
    assert math.isclose(z_low.imag, 0.0, abs_tol=1.0)

    # 2. At cutoff frequency f_c: Z(w_c) = R / (1 + i) = R * (1 - i) / 2 = R/2 - i*R/2
    z_cutoff = solve_complex_impedance(
        graph=G,
        node_a=0,
        node_b=1,
        frequency=f_c,
        conductances=np.array([G_val]),
        capacitances=np.array([C]),
    )
    assert math.isclose(z_cutoff.real, R / 2.0, rel_tol=1e-5)
    assert math.isclose(z_cutoff.imag, -R / 2.0, rel_tol=1e-5)

    # 3. High frequency limit: f -> inf, |Z| -> 0
    z_high = solve_complex_impedance(
        graph=G,
        node_a=0,
        node_b=1,
        frequency=1e6,
        conductances=np.array([G_val]),
        capacitances=np.array([C]),
    )
    assert math.isclose(abs(z_high), 0.0, abs_tol=1.0)


def test_impedance_spectrum_sweep():
    """Test full AC impedance spectrum across frequencies."""
    G = nx.Graph()
    G.add_edge(0, 1)
    G.add_edge(1, 2)

    freqs = np.logspace(0, 5, 20)
    conds = np.array([1e-3, 1e-3])  # two 1 kOhm in series
    caps = np.array([1e-8, 1e-8])   # 10 nF

    spec = solve_complex_impedance_spectrum(
        graph=G,
        node_a=0,
        node_b=2,
        frequencies=freqs,
        conductances=conds,
        capacitances=caps,
    )

    assert len(spec["Z_complex"]) == 20
    assert len(spec["phase_angle_deg"]) == 20
    # Phase should be negative (capacitive reactance)
    assert np.all(spec["Z_imag"] <= 1e-6)


def test_fit_equivalent_circuit_rc():
    """Test parameter fitting of equivalent RC circuit."""
    from network_rheology.fitting.impedance_fitting import fit_equivalent_circuit_rc

    true_rs = 50.0      # 50 Ohm
    true_rj = 5000.0    # 5 kOhm
    true_cj = 2e-9      # 2 nF

    freqs = np.logspace(1, 8, 30)
    omega = 2.0 * np.pi * freqs
    synthetic_z = true_rs + true_rj / (1.0 + 1j * omega * true_rj * true_cj)

    fit_res = fit_equivalent_circuit_rc(
        frequencies=freqs,
        target_z=synthetic_z,
        initial_guess=(10.0, 1000.0, 1e-8),
    )

    assert fit_res["success"]
    assert math.isclose(fit_res["R_sheet"], true_rs, rel_tol=0.05)
    assert math.isclose(fit_res["R_junction"], true_rj, rel_tol=0.05)
    assert math.isclose(fit_res["C_junction"], true_cj, rel_tol=0.05)

