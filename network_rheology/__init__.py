"""
Network Rheology: Simulation and analysis of electrical transport in disordered and structured resistor networks.
"""

from network_rheology.solvers.dc_solver import (
    solve_effective_resistance,
    solve_effective_resistance_fast,
    compute_current_and_dissipation,
)
from network_rheology.solvers.ac_solver import (
    solve_complex_impedance,
    solve_complex_impedance_spectrum,
)
from network_rheology.distributions import (
    generate_conductances,
    generate_resistances,
)
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

__version__ = "0.1.0"
__all__ = [
    "solve_effective_resistance",
    "solve_effective_resistance_fast",
    "compute_current_and_dissipation",
    "solve_complex_impedance",
    "solve_complex_impedance_spectrum",
    "generate_conductances",
    "generate_resistances",
    "create_random_network",
    "create_random_edges_fast",
    "create_brick_lattice_2d",
    "create_brick_lattice_3d",
    "create_1d_chain",
    "create_2d_grid",
    "create_3d_cubic_grid",
]
