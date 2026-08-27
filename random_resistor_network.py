# -*- coding: utf-8 -*-
"""
Random Resistor Network Simulation.

Simulates effective electrical resistance in random resistor networks.
Implements the exact reduced Laplacian formulation for high-performance and exact Kirchhoff compliance.

@author: shardul
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from network_rheology.distributions import generate_conductances
from network_rheology.solvers.dc_solver import solve_effective_resistance_fast
from network_rheology.topologies.random_graph import create_random_edges_fast


def create_graph_from_sample(edges: int, seed: int, nodes: int = 400) -> nx.Graph:
    """
    Sample a random graph with `nodes` nodes and `edges` edges.
    Guarantees all `nodes` are preserved in the graph to prevent out-of-bounds indexing.
    """
    rng = np.random.default_rng(seed)
    max_edges = nodes * (nodes - 1) // 2
    if edges > max_edges:
        raise ValueError(f"Requested number of edges ({edges}) exceeds the maximum possible ({max_edges}).")

    u_idx, v_idx = create_random_edges_fast(nodes=nodes, edges=edges, rng=rng)
    G = nx.Graph()
    G.add_nodes_from(range(nodes))  # Ensure all nodes exist
    G.add_edges_from(zip(u_idx, v_idx))
    return G


def generate_random_values(mean: float, std: float, seed: int, num_edges: int) -> np.ndarray:
    """
    Generate random conductance values with log-normal or normal distribution.
    """
    rng = np.random.default_rng(seed)
    return generate_conductances(mean=mean, std=std, size=num_edges, dist="lognormal", rng=rng)


def matrix_interactions(w: float, meanc: float, stdc: float, edges: int, seed: int, nodes: int = 400):
    """
    Construct the true Kirchhoff graph Laplacian L = D - A and compute effective impedance / resistance.
    """
    rng = np.random.default_rng(seed)
    u_idx, v_idx = create_random_edges_fast(nodes=nodes, edges=edges, rng=rng)
    cap_values = generate_random_values(meanc, stdc, seed, edges)
    conductances = cap_values  # Conductance values

    # Build true Laplacian L = D - A
    rows = np.concatenate([u_idx, v_idx, u_idx, v_idx])
    cols = np.concatenate([v_idx, u_idx, u_idx, v_idx])
    data = np.concatenate([-conductances, -conductances, conductances, conductances])
    L = sp.coo_matrix((data, (rows, cols)), shape=(nodes, nodes)).tocsc()

    return L, nodes


def R(w: float, meanc: float, stdc: float, edges: int, seed: int, x: int, nodes: int = 400) -> Tuple[float, int]:
    """
    Compute the effective two-point resistance between node x and node nodes - 1.
    """
    rng = np.random.default_rng(seed)
    u_idx, v_idx = create_random_edges_fast(nodes=nodes, edges=edges, rng=rng)
    conductances = generate_conductances(mean=meanc, std=stdc, size=edges, dist="lognormal", rng=rng)

    res = solve_effective_resistance_fast(
        num_nodes=nodes,
        u_indices=u_idx,
        v_indices=v_idx,
        conductances=conductances,
        node_a=x,
        node_b=nodes - 1,
    )
    return res, nodes


def parse_args():
    parser = argparse.ArgumentParser(description="Random resistor network simulation.")
    parser.add_argument("--nodes", type=int, default=400, help="Number of nodes in the network.")
    parser.add_argument("--edges-start", type=int, default=1000, help="Starting edge count (inclusive).")
    parser.add_argument("--edges-stop", type=int, default=3000, help="Stopping edge count (exclusive).")
    parser.add_argument("--edges-step", type=int, default=200, help="Step size for edge count.")
    parser.add_argument("--dev-start", type=int, default=1, help="Starting disorder (std) value (inclusive).")
    parser.add_argument("--dev-stop", type=int, default=10, help="Stopping disorder (std) value (exclusive).")
    parser.add_argument("--seeds", type=int, default=100, help="Number of random seeds per edge count.")
    parser.add_argument("--mean", type=float, default=10.0, help="Mean of the distribution.")
    parser.add_argument("--dist", type=str, default="lognormal", choices=["lognormal", "truncated_normal", "folded_normal", "uniform", "constant"])
    parser.add_argument("--x-node", type=int, default=0, help="Node index for resistance calculation.")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel worker processes.")
    parser.add_argument("--save-csv", type=str, default="", help="Path to save results as CSV (optional).")
    parser.add_argument("--save-plot", type=str, default="", help="Path to save plot (optional).")
    parser.add_argument("--output-dir", type=str, default="", help="Create a timestamped output folder and save CSV/plot inside it.")
    parser.add_argument("--save-meta", type=str, default="", help="Path to save run metadata as JSON (optional).")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot.")
    return parser.parse_args()


def _eval_single_point(args_tuple):
    seed, mean, dev, edges, x_node, nodes, dist = args_tuple
    rng = np.random.default_rng(seed)
    u_idx, v_idx = create_random_edges_fast(nodes=nodes, edges=edges, rng=rng)
    conductances = generate_conductances(mean=mean, std=dev, size=edges, dist=dist, rng=rng)
    return solve_effective_resistance_fast(
        num_nodes=nodes,
        u_indices=u_idx,
        v_indices=v_idx,
        conductances=conductances,
        node_a=x_node,
        node_b=nodes - 1,
    )


def run_simulation(args):
    """
    Run parameter sweep with parallel multi-core execution.
    """
    num_workers = args.workers if getattr(args, "workers", None) else os.cpu_count() or 1
    dist = getattr(args, "dist", "lognormal")
    results = []

    print(f"Starting simulation on {num_workers} worker processes...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for dev in range(args.dev_start, args.dev_stop):
            print(f"Disorder (std) = {dev}")
            dev_results = []
            for edges in range(args.edges_start, args.edges_stop, args.edges_step):
                tasks = [
                    (seed, args.mean, float(dev), edges, args.x_node, args.nodes, dist)
                    for seed in range(args.seeds)
                ]
                futures = [executor.submit(_eval_single_point, t) for t in tasks]
                resistances = []
                for f in futures:
                    res = f.result()
                    if np.isfinite(res):
                        resistances.append(res)
                if resistances:
                    dev_results.append([edges, float(np.mean(resistances))])
            results.append(dev_results)
    return results


def save_csv(results, args):
    if not args.save_csv:
        return
    with open(args.save_csv, "w", encoding="utf-8") as f:
        f.write("dev,edges,avg_resistance\n")
        for dev, dev_res in zip(range(args.dev_start, args.dev_stop), results):
            for edges, avg_res in dev_res:
                f.write(f"{dev},{int(edges)},{float(avg_res):.6f}\n")


def prepare_output_paths(args):
    if not args.output_dir:
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / timestamp
    out_dir.mkdir(parents=True, exist_ok=False)

    if args.save_csv:
        csv_path = Path(args.save_csv)
        if not csv_path.is_absolute():
            args.save_csv = str(out_dir / csv_path)
    else:
        args.save_csv = str(out_dir / "results.csv")

    if args.save_plot:
        plot_path = Path(args.save_plot)
        if not plot_path.is_absolute():
            args.save_plot = str(out_dir / plot_path)
    else:
        args.save_plot = str(out_dir / "plot.png")

    if args.save_meta:
        meta_path = Path(args.save_meta)
        if not meta_path.is_absolute():
            args.save_meta = str(out_dir / meta_path)
    else:
        args.save_meta = str(out_dir / "metadata.json")


def save_metadata(args):
    if not args.save_meta:
        return
    with open(args.save_meta, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)


def plot_results(results, args):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    for dev, dev_res in zip(range(args.dev_start, args.dev_stop), results):
        if not dev_res:
            continue
        dev_arr = np.array(dev_res)
        ax.plot(dev_arr[:, 0], dev_arr[:, 1], marker="o", label=f"std = {dev}")

    ax.set_xlabel("Edge Count (M)", fontsize=12)
    ax.set_ylabel(r"Average Effective Resistance ($\Omega$)", fontsize=12)
    ax.set_title("Random Resistor Network: Resistance vs Connectivity", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(title="Disorder (std)", frameon=True)

    plt.tight_layout()
    if args.save_plot:
        plt.savefig(args.save_plot, dpi=200, bbox_inches="tight")
    if not args.no_show:
        plt.show()


def main():
    args = parse_args()
    prepare_output_paths(args)
    results = run_simulation(args)
    save_metadata(args)
    save_csv(results, args)
    plot_results(results, args)


if __name__ == "__main__":
    main()
