"""
Publication-ready plotting tools for resistance sweeps, impedance spectra, and scaling analysis.
"""

from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_resistance_vs_edges(
    data: Union[pd.DataFrame, Dict[int, List[List[float]]], List[List[List[float]]]],
    save_path: Optional[str] = None,
    show: bool = True,
    title: Optional[str] = None,
    xlabel: str = "Edge Count (M)",
    ylabel: str = "Average Effective Resistance (Ohm)",
    include_error_bars: bool = True,
) -> plt.Figure:
    """
    Plot average resistance vs edge count across different disorder (std) levels.

    Parameters
    ----------
    data : Union[pd.DataFrame, Dict, List]
        Simulation results. Can be DataFrame with columns ['dev', 'edges', 'avg_resistance' (and optional 'sem')],
        or nested lists/dict from simulation runs.
    save_path : Optional[str], optional
        File path to save the generated figure.
    show : bool, optional
        Whether to display the plot interactively, default True.
    title : Optional[str], optional
        Plot title.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    include_error_bars : bool, optional
        Whether to plot error bands/bars if available.

    Returns
    -------
    plt.Figure
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    if isinstance(data, pd.DataFrame):
        for dev, group in data.groupby("dev"):
            group_sorted = group.sort_values("edges")
            edges = group_sorted["edges"].values
            res = group_sorted["avg_resistance"].values

            if "sem" in group_sorted.columns and include_error_bars:
                sem = group_sorted["sem"].values
                ax.plot(edges, res, marker="o", label=f"std = {dev}")
                ax.fill_between(edges, res - sem, res + sem, alpha=0.2)
            else:
                ax.plot(edges, res, marker="o", label=f"std = {dev}")
    elif isinstance(data, dict):
        for dev, dev_res in sorted(data.items()):
            if not dev_res:
                continue
            arr = np.array(dev_res)
            ax.plot(arr[:, 0], arr[:, 1], marker="o", label=f"std = {dev}")
    else:
        # List of results per disorder level
        for i, dev_res in enumerate(data):
            if not dev_res:
                continue
            arr = np.array(dev_res)
            ax.plot(arr[:, 0], arr[:, 1], marker="o", label=f"std = {i+1}")

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title("Random Resistor Network: Resistance vs. Connectivity", fontsize=13)

    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(title="Disorder (std)", frameon=True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_nyquist(
    frequencies: np.ndarray,
    z_complex: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True,
    title: str = "Nyquist Plot",
) -> plt.Figure:
    """
    Plot the complex impedance on a Nyquist diagram (-Z'' vs Z').
    """
    z_real = np.real(z_complex)
    z_imag = np.imag(z_complex)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    ax.plot(z_real, -z_imag, "b.-", linewidth=1.5, markersize=6, label="Impedance Z(w)")

    ax.set_xlabel(r"$Z' \ [\Omega]$ (Real / Resistance)", fontsize=12)
    ax.set_ylabel(r"$-Z'' \ [\Omega]$ (Negative Reactance)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_aspect("equal", "datalim")
    ax.legend(frameon=True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_bode(
    frequencies: np.ndarray,
    z_complex: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True,
    title: str = "Bode Plot",
) -> plt.Figure:
    """
    Plot magnitude |Z| and phase angle theta vs frequency on a 2-panel Bode diagram.
    """
    freqs = np.asarray(frequencies)
    z_mag = np.abs(z_complex)
    phase_deg = np.rad2deg(np.angle(z_complex))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, dpi=150)

    # Magnitude subplot
    ax1.loglog(freqs, z_mag, "b.-", linewidth=1.5, markersize=5)
    ax1.set_ylabel(r"$|Z| \ [\Omega]$", fontsize=12)
    ax1.set_title(title, fontsize=14)
    ax1.grid(True, which="both", linestyle="--", alpha=0.6)

    # Phase subplot
    ax2.semilogx(freqs, phase_deg, "r.-", linewidth=1.5, markersize=5)
    ax2.set_xlabel("Frequency (Hz)", fontsize=12)
    ax2.set_ylabel(r"Phase $\theta$ [deg]", fontsize=12)
    ax2.grid(True, which="both", linestyle="--", alpha=0.6)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_anisotropy_scaling(
    depths: np.ndarray,
    r_oop: np.ndarray,
    r_ip: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot anisotropy ratio R_oop / R_ip vs flake stack depth N with theoretical scaling lines.
    """
    depths = np.asarray(depths)
    ratio = np.asarray(r_oop) / np.asarray(r_ip)
    n_flakes = 32 * depths

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.scatter(n_flakes, ratio, color="navy", s=50, label="Simulation (R_oop / R_ip)", zorder=5)

    n_cont = np.linspace(n_flakes.min(), n_flakes.max(), 200)
    # Quadratic theoretical scaling
    c_fit = ratio[len(ratio) // 2] / (n_flakes[len(n_flakes) // 2] ** 2)
    ax.plot(n_cont, c_fit * n_cont**2, "r--", linewidth=1.5, label=r"Theory: $\propto N^2$ Scaling")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total Number of Flakes $N$", fontsize=12)
    ax.set_ylabel(r"Anisotropy Ratio $R_{\mathrm{oop}} / R_{\mathrm{ip}}$", fontsize=12)
    ax.set_title("3D Layered Flake Network: Anisotropic Transport Scaling", fontsize=13)
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend(frameon=True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig
