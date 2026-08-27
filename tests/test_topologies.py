"""
Tests for graph topology generators (random graphs, 2D/3D brick lattices, regular grids).
"""

import networkx as nx
import numpy as np
import pytest

from network_rheology.topologies.brick_lattice import (
    create_brick_lattice_2d,
    create_brick_lattice_3d,
)
from network_rheology.topologies.grid_lattice import (
    create_1d_chain,
    create_2d_grid,
    create_3d_cubic_grid,
)
from network_rheology.topologies.random_graph import (
    create_random_edges_fast,
    create_random_network,
)


def test_random_graph_node_retention():
    """Verify that all nodes are retained in random graph even at very low edge density."""
    nodes = 100
    edges = 10  # Very sparse, many isolated nodes
    G = create_random_network(nodes=nodes, edges=edges, seed=42)

    assert G.number_of_nodes() == 100
    assert G.number_of_edges() == 10
    assert max(G.nodes()) == 99


def test_random_edges_fast_bounds():
    """Verify u, v arrays are within [0, nodes-1] and u < v."""
    u, v = create_random_edges_fast(nodes=50, edges=100, rng=123)
    assert len(u) == 100
    assert len(v) == 100
    assert np.all(u < v)
    assert np.all(u >= 0)
    assert np.all(v < 50)


def test_brick_lattice_2d():
    """Verify 2D flake layer structure."""
    layers = 4
    G_2d = create_brick_lattice_2d(layers=layers)
    assert G_2d.number_of_nodes() == 8 * layers  # 32 nodes
    assert nx.is_connected(G_2d)


def test_brick_lattice_3d():
    """Verify 3D stacked flake structure with OOP and IP electrodes."""
    layers = 4
    depth = 5

    # Out-of-Plane
    G_oop, ea_oop, eb_oop, mapping_oop = create_brick_lattice_3d(layers=layers, depth=depth, is_oop=True)
    # Bulk flakes: 8 * layers * depth = 32 * 5 = 160. Total nodes = 160 + 2 electrodes = 162
    assert G_oop.number_of_nodes() == 162
    assert nx.has_path(G_oop, ea_oop, eb_oop)

    # In-Plane
    G_ip, ea_ip, eb_ip, mapping_ip = create_brick_lattice_3d(layers=layers, depth=depth, is_oop=False)
    assert G_ip.number_of_nodes() == 162
    assert nx.has_path(G_ip, ea_ip, eb_ip)


def test_regular_grids():
    """Verify 1D, 2D, and 3D regular lattices."""
    G_1d = create_1d_chain(nodes=10)
    assert G_1d.number_of_nodes() == 10
    assert G_1d.number_of_edges() == 9

    G_2d = create_2d_grid(rows=4, cols=5)
    assert G_2d.number_of_nodes() == 20
    assert G_2d.number_of_edges() == 4 * (5 - 1) + 5 * (4 - 1)  # 16 + 15 = 31

    G_3d = create_3d_cubic_grid(nx_nodes=3, ny_nodes=3, nz_nodes=3)
    assert G_3d.number_of_nodes() == 27
