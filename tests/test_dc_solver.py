"""
Tests for DC resistance and electrical field solvers against analytical closed-form circuits.
"""

import math
import networkx as nx
import numpy as np
import pytest

from network_rheology.solvers.dc_solver import (
    compute_current_and_dissipation,
    solve_effective_resistance,
    solve_effective_resistance_fast,
)


def test_single_resistor():
    """Test 2-node single resistor of R = 5.0 Ohm."""
    G = nx.Graph()
    G.add_edge(0, 1, weight=1.0 / 5.0)  # g = 0.2 S

    r = solve_effective_resistance(G, node_a=0, node_b=1)
    assert math.isclose(r, 5.0, rel_tol=1e-6)

    # Fast solver
    r_fast = solve_effective_resistance_fast(
        num_nodes=2,
        u_indices=np.array([0]),
        v_indices=np.array([1]),
        conductances=np.array([0.2]),
        node_a=0,
        node_b=1,
    )
    assert math.isclose(r_fast, 5.0, rel_tol=1e-6)


def test_series_resistors():
    """Test 2 resistors in series: 0 --(1 Ohm)-- 1 --(2 Ohm)-- 2 -> R_02 = 3 Ohm."""
    G = nx.Graph()
    G.add_edge(0, 1, weight=1.0 / 1.0)
    G.add_edge(1, 2, weight=1.0 / 2.0)

    r_02 = solve_effective_resistance(G, node_a=0, node_b=2)
    assert math.isclose(r_02, 3.0, rel_tol=1e-6)


def test_parallel_resistors():
    """Test 2 resistors in parallel between node 0 and node 1: 1 Ohm || 1 Ohm = 0.5 Ohm."""
    # Since NetworkX Graph collapses multigraph edges, test via 3-node diamond: 0-1 and 0-2-1
    G = nx.Graph()
    G.add_edge(0, 1, weight=1.0)  # Path 1: 1 Ohm
    G.add_edge(0, 2, weight=2.0)  # Path 2 first half: 0.5 Ohm
    G.add_edge(2, 1, weight=2.0)  # Path 2 second half: 0.5 Ohm -> total 1 Ohm
    # Equivalent to 1 Ohm || 1 Ohm = 0.5 Ohm

    r_01 = solve_effective_resistance(G, node_a=0, node_b=1)
    assert math.isclose(r_01, 0.5, rel_tol=1e-6)


def test_triangle_circuit():
    """
    Test 3-node triangle with all 1 Ohm resistors.
    Path (0-2) in parallel with (0-1-2) -> 1 Ohm || 2 Ohm = 2/3 Ohm.
    """
    G = nx.Graph()
    G.add_edge(0, 1, weight=1.0)
    G.add_edge(1, 2, weight=1.0)
    G.add_edge(0, 2, weight=1.0)

    r_02 = solve_effective_resistance(G, node_a=0, node_b=2)
    assert math.isclose(r_02, 2.0 / 3.0, rel_tol=1e-6)

    # Fast solver
    r_fast = solve_effective_resistance_fast(
        num_nodes=3,
        u_indices=np.array([0, 1, 0]),
        v_indices=np.array([1, 2, 2]),
        conductances=np.array([1.0, 1.0, 1.0]),
        node_a=0,
        node_b=2,
    )
    assert math.isclose(r_fast, 2.0 / 3.0, rel_tol=1e-6)


def test_wheatstone_bridge():
    """
    Test 5-resistor Wheatstone bridge with arbitrary resistances:
    Nodes: 0 (input), 3 (output), 1, 2 (internal nodes).
    Edges: (0,1)=R1, (0,2)=R2, (1,2)=R5, (1,3)=R3, (2,3)=R4.
    """
    r1, r2, r3, r4, r5 = 1.0, 2.0, 3.0, 4.0, 5.0

    # Exact Delta-Wye transformation on delta (0, 1, 2):
    # R_N0 = R1*R2 / (R1+R2+R5) = 2/8 = 0.25
    # R_N1 = R1*R5 / (R1+R2+R5) = 5/8 = 0.625
    # R_N2 = R2*R5 / (R1+R2+R5) = 10/8 = 1.25
    # Branch 1 to node 3: 5/8 + 3 = 29/8
    # Branch 2 to node 3: 10/8 + 4 = 42/8
    # Parallel combination: (29/8 * 42/8) / (71/8) = 1218 / 568
    # Total R_03 = 2/8 + 1218/568 = 1360 / 568 = 170 / 71 ~ 2.394366197
    expected_r = 170.0 / 71.0

    G = nx.Graph()
    G.add_edge(0, 1, weight=1.0 / r1)
    G.add_edge(0, 2, weight=1.0 / r2)
    G.add_edge(1, 3, weight=1.0 / r3)
    G.add_edge(2, 3, weight=1.0 / r4)
    G.add_edge(1, 2, weight=1.0 / r5)

    r_03 = solve_effective_resistance(G, node_a=0, node_b=3)
    assert math.isclose(r_03, expected_r, rel_tol=1e-6)


def test_disconnected_components():
    """Test that disconnected nodes correctly return infinity."""
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3])
    G.add_edge(0, 1, weight=1.0)
    G.add_edge(2, 3, weight=1.0)

    # 0 and 3 are disconnected
    r_03 = solve_effective_resistance(G, node_a=0, node_b=3)
    assert math.isinf(r_03)

    r_fast = solve_effective_resistance_fast(
        num_nodes=4,
        u_indices=np.array([0, 2]),
        v_indices=np.array([1, 3]),
        conductances=np.array([1.0, 1.0]),
        node_a=0,
        node_b=3,
    )
    assert math.isinf(r_fast)


def test_joule_dissipation_conservation():
    """
    Test that total Joule dissipation P_total = sum(I^2 / g) equals R_eff * I_total^2 = R_eff * 1.0^2 = R_eff.
    """
    G = nx.Graph()
    G.add_edge(0, 1, weight=1.0 / 2.0)  # 2 Ohm
    G.add_edge(1, 2, weight=1.0 / 3.0)  # 3 Ohm
    G.add_edge(0, 2, weight=1.0 / 6.0)  # 6 Ohm (in parallel with 5 Ohm path -> 5 || 6 = 30/11 Ohm)

    fields = compute_current_and_dissipation(G, node_a=0, node_b=2)
    r_eff = fields["R_eff"]
    p_total = fields["P_total"]

    assert math.isclose(p_total, r_eff, rel_tol=1e-5)
    assert math.isclose(r_eff, 30.0 / 11.0, rel_tol=1e-5)
