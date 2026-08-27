"""
Solvers for DC resistance, electrical potential distribution, and AC complex impedance.
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

__all__ = [
    "solve_effective_resistance",
    "solve_effective_resistance_fast",
    "compute_current_and_dissipation",
    "solve_complex_impedance",
    "solve_complex_impedance_spectrum",
]
