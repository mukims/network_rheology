"""
AC Electrical Impedance Spectroscopy (EIS) solver for complex admittance and impedance spectra.
"""

from typing import Dict, Optional, Tuple, Union
import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def solve_complex_impedance(
    graph: nx.Graph,
    node_a: int,
    node_b: int,
    frequency: float,
    conductances: np.ndarray,
    capacitances: np.ndarray,
) -> complex:
    """
    Solve the two-point complex electrical impedance Z*(w) = Z' + i*Z'' between node_a and node_b
    at an angular frequency w = 2*pi*f.

    Parameters
    ----------
    graph : nx.Graph
        The resistor-capacitor network graph.
    node_a : int
        Source node (+1 A AC injection).
    node_b : int
        Drain / ground reference node (0 V).
    frequency : float
        AC frequency (Hz).
    conductances : np.ndarray
        Conductance g_ij for each edge (S = 1/Ohm).
    capacitances : np.ndarray
        Capacitance c_ij for each edge (Farads).

    Returns
    -------
    complex
        Complex impedance Z = Z_real + i*Z_imag (Ohm).
    """
    if node_a == node_b:
        return 0.0 + 0.0j

    if not nx.has_path(graph, node_a, node_b):
        return complex(float("inf"), float("inf"))

    omega = 2.0 * np.pi * float(frequency)
    num_nodes = max(graph.nodes()) + 1
    edges = list(graph.edges())

    g = np.asarray(conductances, dtype=np.float64)
    c = np.asarray(capacitances, dtype=np.float64)

    # Complex branch admittance: y_ij = g_ij + i*w*c_ij
    y = g + 1j * omega * c

    u_indices = np.array([e[0] for e in edges], dtype=np.int32)
    v_indices = np.array([e[1] for e in edges], dtype=np.int32)

    # Build complex admittance matrix Y(w) = G + i*w*C
    rows = np.concatenate([u_indices, v_indices, u_indices, v_indices])
    cols = np.concatenate([v_indices, u_indices, u_indices, v_indices])
    data = np.concatenate([-y, -y, y, y])
    Y = sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.complex128).tocsc()

    # Form reduced admittance matrix by grounding node_b
    nodes_kept = np.array([i for i in range(num_nodes) if i != node_b], dtype=np.int32)
    Y_red = Y[nodes_kept, :][:, nodes_kept].tocsc()

    mapped_a = node_a if node_a < node_b else node_a - 1
    rhs = np.zeros(num_nodes - 1, dtype=np.complex128)
    rhs[mapped_a] = 1.0 + 0.0j

    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=spla.MatrixRankWarning)
            v_red = spla.spsolve(Y_red, rhs)
        z_eff = complex(v_red[mapped_a])
        return z_eff
    except (spla.MatrixRankWarning, RuntimeError, ValueError):
        return complex(float("inf"), float("inf"))


def solve_complex_impedance_spectrum(
    graph: nx.Graph,
    node_a: int,
    node_b: int,
    frequencies: np.ndarray,
    conductances: np.ndarray,
    capacitances: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Compute full AC electrical impedance spectroscopy across a sweep of frequencies.

    Parameters
    ----------
    graph : nx.Graph
        NetworkX graph.
    node_a : int
        Source node index.
    node_b : int
        Ground reference node index.
    frequencies : np.ndarray
        Array of frequencies (Hz) (e.g. 10^-2 to 10^7 Hz).
    conductances : np.ndarray
        Conductance of each edge (S).
    capacitances : np.ndarray
        Capacitance of each edge (F).

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing:
        - "frequencies": Input frequency array (Hz).
        - "omega": Angular frequencies 2*pi*f (rad/s).
        - "Z_complex": Complex impedance array (Ohm).
        - "Z_real": Real part Z' (In-phase resistance, Ohm).
        - "Z_imag": Imaginary part Z'' (Out-of-phase reactance, Ohm).
        - "Z_magnitude": |Z| = sqrt(Z'^2 + Z''^2) (Ohm).
        - "phase_angle_rad": Phase angle theta = atan2(Z'', Z') (radians).
        - "phase_angle_deg": Phase angle in degrees.
    """
    freqs = np.asarray(frequencies, dtype=np.float64)
    z_list = []

    for f in freqs:
        z = solve_complex_impedance(
            graph=graph,
            node_a=node_a,
            node_b=node_b,
            frequency=f,
            conductances=conductances,
            capacitances=capacitances,
        )
        z_list.append(z)

    z_arr = np.array(z_list, dtype=np.complex128)
    z_real = np.real(z_arr)
    z_imag = np.imag(z_arr)
    z_mag = np.abs(z_arr)
    phase_rad = np.angle(z_arr)
    phase_deg = np.rad2deg(phase_rad)

    return {
        "frequencies": freqs,
        "omega": 2.0 * np.pi * freqs,
        "Z_complex": z_arr,
        "Z_real": z_real,
        "Z_imag": z_imag,
        "Z_magnitude": z_mag,
        "phase_angle_rad": phase_rad,
        "phase_angle_deg": phase_deg,
    }
