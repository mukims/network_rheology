"""
3D Spatial visualization for nodal voltages, currents, and dissipation in stacked flake networks.
"""

from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np


def plot_3d_flake_currents(
    i_abs_nodes: np.ndarray,
    depth: int,
    save_path: Optional[str] = None,
    show: bool = True,
    title: str = "3D Current Distribution across Flake Assembly",
) -> plt.Figure:
    """
    Plot the 3D spatial current throughput across the staggered brick-layered flake assembly.

    Parameters
    ----------
    i_abs_nodes : np.ndarray
        Array of total absolute currents passing through each node (from `compute_current_and_dissipation`).
    depth : int
        Number of vertical unit cell layers.
    save_path : Optional[str], optional
        Path to save figure.
    show : bool, optional
        Whether to show interactively.
    title : str, optional
        Plot title.

    Returns
    -------
    plt.Figure
        Matplotlib 3D figure.
    """
    # Bulk flakes are 32 * depth (excluding the 2 external electrodes)
    n_bulk = 32 * depth
    vals_bulk = i_abs_nodes[:n_bulk]

    x_coords: List[float] = []
    y_coords: List[float] = []
    z_coords: List[float] = []

    for i in range(n_bulk):
        di = i % 32
        if di % 2 == 0:
            x = (di % 8) / 2.0
            y = int(di / 8)
            z = 2.0 * int(i / 32)
        else:
            x = ((di - 1) % 8) / 2.0 + 0.5
            y = int((di - 1) / 8) + 0.5
            z = 2.0 * int(i / 32) + 1.0

        x_coords.append(x)
        y_coords.append(y)
        z_coords.append(z)

    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(projection="3d")

    p = ax.scatter(
        x_coords,
        y_coords,
        z_coords,
        c=vals_bulk,
        cmap="inferno",
        s=80,
        edgecolors="black",
        linewidth=0.5,
        alpha=0.9,
    )
    cbar = fig.colorbar(p, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label("Current Throughput (A)", fontsize=11)

    ax.set_xlabel("X (In-plane width)", fontsize=11)
    ax.set_ylabel("Y (In-plane length)", fontsize=11)
    ax.set_zlabel("Z (Stack Depth Layer)", fontsize=11)
    ax.set_title(title, fontsize=13)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig
