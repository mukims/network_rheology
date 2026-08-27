"""
Network topologies: Random graphs, 2D/3D brick-layered flake assemblies, and regular lattices.
"""

from network_rheology.topologies.random_graph import (
    create_random_network,
    create_random_edges_fast,
)
from network_rheology.topologies.brick_lattice import (
    create_brick_lattice_2d,
    create_brick_lattice_3d,
)
from network_rheology.topologies.grid_lattice import (
    create_1d_chain,
    create_2d_grid,
    create_3d_cubic_grid,
)

__all__ = [
    "create_random_network",
    "create_random_edges_fast",
    "create_brick_lattice_2d",
    "create_brick_lattice_3d",
    "create_1d_chain",
    "create_2d_grid",
    "create_3d_cubic_grid",
]
