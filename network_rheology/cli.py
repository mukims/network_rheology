"""
Unified Command-Line Interface for Network Rheology simulations.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from network_rheology.distributions import generate_conductances
from network_rheology.solvers.ac_solver import solve_complex_impedance_spectrum
from network_rheology.solvers.dc_solver import (
    compute_current_and_dissipation,
    solve_effective_resistance,
    solve_effective_resistance_fast,
)
from network_rheology.topologies.brick_lattice import create_brick_lattice_3d
from network_rheology.topologies.random_graph import create_random_edges_fast
from network_rheology.visualization.plots import (
    plot_anisotropy_scaling,
    plot_bode,
    plot_nyquist,
    plot_resistance_vs_edges,
)


def _eval_single_random_seed(args_tuple: Tuple[int, int, int, float, float, str, int, int]) -> float:
    """Helper worker function to evaluate a single random seed."""
    seed, nodes, edges, mean_g, std_g, dist, node_a, node_b = args_tuple
    rng = np.random.default_rng(seed)

    u_idx, v_idx = create_random_edges_fast(nodes=nodes, edges=edges, rng=rng)
    conductances = generate_conductances(mean=mean_g, std=std_g, size=edges, dist=dist, rng=rng)

    return solve_effective_resistance_fast(
        num_nodes=nodes,
        u_indices=u_idx,
        v_indices=v_idx,
        conductances=conductances,
        node_a=node_a,
        node_b=node_b,
    )


def run_random_sweep(
    nodes: int = 400,
    edges_start: int = 1000,
    edges_stop: int = 3000,
    edges_step: int = 200,
    dev_start: int = 1,
    dev_stop: int = 10,
    seeds: int = 100,
    mean: float = 10.0,
    dist: str = "lognormal",
    x_node: int = 0,
    target_node: Optional[int] = None,
    workers: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run multi-core parallelized sweep over disorder and edge densities.
    """
    node_b = nodes - 1 if target_node is None else target_node
    edge_list = list(range(edges_start, edges_stop, edges_step))
    dev_list = list(range(dev_start, dev_stop))

    tasks = []
    for dev in dev_list:
        for edges in edge_list:
            for seed in range(seeds):
                tasks.append((seed, nodes, edges, mean, float(dev), dist, x_node, node_b))

    results_dict: Dict[Tuple[int, int], List[float]] = {(dev, e): [] for dev in dev_list for e in edge_list}

    num_workers = workers if workers and workers > 0 else os.cpu_count() or 1
    if verbose:
        print(f"Running {len(tasks)} simulations on {num_workers} parallel workers...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_eval_single_random_seed, t): t for t in tasks}
        for future in as_completed(futures):
            t = futures[future]
            dev = int(t[4])
            edges = t[2]
            try:
                res = future.result()
                results_dict[(dev, edges)].append(res)
            except Exception as e:
                if verbose:
                    print(f"Task failed for dev={dev}, edges={edges}: {e}")

    rows = []
    for dev in dev_list:
        for edges in edge_list:
            res_list = np.array(results_dict[(dev, edges)])
            # Filter finite values (percolating paths)
            finite_res = res_list[np.isfinite(res_list)]
            p_connected = len(finite_res) / len(res_list) if len(res_list) > 0 else 0.0

            if len(finite_res) > 0:
                avg_res = float(np.mean(finite_res))
                std_res = float(np.std(finite_res))
                sem_res = float(std_res / np.sqrt(len(finite_res)))
                median_res = float(np.median(finite_res))
                q25 = float(np.percentile(finite_res, 25))
                q75 = float(np.percentile(finite_res, 75))
            else:
                avg_res = float("inf")
                std_res = float("nan")
                sem_res = float("nan")
                median_res = float("inf")
                q25 = float("nan")
                q75 = float("nan")

            rows.append({
                "dev": dev,
                "edges": edges,
                "avg_resistance": avg_res,
                "std_resistance": std_res,
                "sem": sem_res,
                "median_resistance": median_res,
                "q25": q25,
                "q75": q75,
                "percolation_rate": p_connected,
                "valid_samples": len(finite_res),
            })

    return pd.DataFrame(rows)


def run_brick3d_sweep(
    layers: int = 4,
    depths: Optional[List[int]] = None,
    std: float = 0.0,
    seeds: int = 20,
    mean: float = 1.0,
    hotspot_frac: float = 0.0,
    hotspot_val: float = 1e4,
    ext_r: float = 1e-4,
    workers: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run 3D brick lattice sweep for In-Plane (IP) vs Out-of-Plane (OOP) anisotropy.
    """
    if depths is None:
        depths = [1, 2, 4, 8, 12, 16, 20]

    rows = []
    for d in depths:
        if verbose:
            print(f"Processing depth layer: {d}")
        oop_res_list = []
        ip_res_list = []

        for s in range(seeds):
            rng = np.random.default_rng(s)

            # OOP
            g_oop, ea_oop, eb_oop, _ = create_brick_lattice_3d(layers=layers, depth=d, is_oop=True, seed=s)
            edges_oop = list(g_oop.edges(data=True))
            g_vals_oop = []
            for u, v, data in edges_oop:
                if data.get("edge_type") == "electrode":
                    g_vals_oop.append(1.0 / ext_r)
                else:
                    g_vals_oop.append(
                        generate_conductances(
                            mean=mean,
                            std=std,
                            size=1,
                            dist="lognormal",
                            hotspot_frac=hotspot_frac,
                            hotspot_val=hotspot_val,
                            rng=rng,
                        )[0]
                    )

            r_oop = solve_effective_resistance(g_oop, ea_oop, eb_oop, conductances=np.array(g_vals_oop))
            oop_res_list.append(r_oop)

            # IP
            g_ip, ea_ip, eb_ip, _ = create_brick_lattice_3d(layers=layers, depth=d, is_oop=False, seed=s)
            edges_ip = list(g_ip.edges(data=True))
            g_vals_ip = []
            for u, v, data in edges_ip:
                if data.get("edge_type") == "electrode":
                    g_vals_ip.append(1.0 / ext_r)
                else:
                    g_vals_ip.append(
                        generate_conductances(
                            mean=mean,
                            std=std,
                            size=1,
                            dist="lognormal",
                            hotspot_frac=hotspot_frac,
                            hotspot_val=hotspot_val,
                            rng=rng,
                        )[0]
                    )

            r_ip = solve_effective_resistance(g_ip, ea_ip, eb_ip, conductances=np.array(g_vals_ip))
            ip_res_list.append(r_ip)

        mean_oop = float(np.mean(oop_res_list))
        mean_ip = float(np.mean(ip_res_list))
        ratio = mean_oop / mean_ip if mean_ip > 0 else float("inf")
        prefactor_c = ratio * (4.0 / (d**2)) if d > 0 else 0.0

        rows.append({
            "depth": d,
            "total_flakes": 32 * d,
            "R_OOP": mean_oop,
            "R_IP": mean_ip,
            "ratio_OOP_IP": ratio,
            "scaling_prefactor_c": prefactor_c,
        })

    return pd.DataFrame(rows)


def _prepare_output_dir(output_dir: Optional[str]) -> Optional[Path]:
    if not output_dir:
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(output_dir) / timestamp
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        prog="network-rheology",
        description="Network Rheology: Advanced simulation framework for resistor and impedance networks.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Subcommand: random
    p_rand = subparsers.add_parser("random", help="Simulate random resistor networks (G(N, M)).")
    p_rand.add_argument("--nodes", type=int, default=400, help="Number of nodes (default: 400).")
    p_rand.add_argument("--edges-start", type=int, default=1000, help="Starting edge count (default: 1000).")
    p_rand.add_argument("--edges-stop", type=int, default=3000, help="Stopping edge count (default: 3000).")
    p_rand.add_argument("--edges-step", type=int, default=200, help="Step size for edge count (default: 200).")
    p_rand.add_argument("--dev-start", type=int, default=1, help="Starting disorder std (default: 1).")
    p_rand.add_argument("--dev-stop", type=int, default=10, help="Stopping disorder std (default: 10).")
    p_rand.add_argument("--seeds", type=int, default=100, help="Number of random seeds per point (default: 100).")
    p_rand.add_argument("--mean", type=float, default=10.0, help="Mean conductance (default: 10.0).")
    p_rand.add_argument(
        "--dist",
        type=str,
        default="lognormal",
        choices=["lognormal", "truncated_normal", "folded_normal", "uniform", "constant"],
        help="Disorder distribution type (default: lognormal).",
    )
    p_rand.add_argument("--x-node", type=int, default=0, help="Probe node A index (default: 0).")
    p_rand.add_argument("--target-node", type=int, default=None, help="Probe node B index (default: nodes - 1).")
    p_rand.add_argument("--workers", type=int, default=None, help="Parallel CPU workers (default: all).")
    p_rand.add_argument("--save-csv", type=str, default="", help="Path to export CSV results.")
    p_rand.add_argument("--save-plot", type=str, default="", help="Path to export plot image.")
    p_rand.add_argument("--save-meta", type=str, default="", help="Path to export run metadata JSON.")
    p_rand.add_argument("--output-dir", type=str, default="", help="Save outputs inside a timestamped folder.")
    p_rand.add_argument("--no-show", action="store_true", help="Do not display plot interactively.")

    # Subcommand: brick3d
    p_brick = subparsers.add_parser("brick3d", help="Simulate 3D layered/staggered flake assemblies.")
    p_brick.add_argument("--layers", type=int, default=4, help="Horizontal unit cell count (default: 4).")
    p_brick.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8, 12, 16, 20], help="List of depth layers.")
    p_brick.add_argument("--std", type=float, default=0.0, help="Disorder std in contacts (default: 0.0).")
    p_brick.add_argument("--seeds", type=int, default=20, help="Random seeds per depth (default: 20).")
    p_brick.add_argument("--hotspot-frac", type=float, default=0.0, help="Fraction of hotspot contacts (default: 0.0).")
    p_brick.add_argument("--ext-r", type=float, default=1e-4, help="Electrode contact resistance (default: 1e-4).")
    p_brick.add_argument("--save-csv", type=str, default="", help="Path to export CSV results.")
    p_brick.add_argument("--save-plot", type=str, default="", help="Path to export plot image.")
    p_brick.add_argument("--output-dir", type=str, default="", help="Save outputs inside a timestamped folder.")
    p_brick.add_argument("--no-show", action="store_true", help="Do not display plot interactively.")

    # Subcommand: impedance
    p_imp = subparsers.add_parser("impedance", help="Simulate AC electrical impedance spectroscopy.")
    p_imp.add_argument("--f-min", type=float, default=1e-1, help="Minimum frequency in Hz (default: 0.1).")
    p_imp.add_argument("--f-max", type=float, default=1e7, help="Maximum frequency in Hz (default: 10^7).")
    p_imp.add_argument("--f-points", type=int, default=50, help="Number of logarithmic frequency points (default: 50).")
    p_imp.add_argument("--r-sheet", type=float, default=100.0, help="Nanosheet resistance in Ohm (default: 100).")
    p_imp.add_argument("--r-junction", type=float, default=1e4, help="Junction resistance in Ohm (default: 10^4).")
    p_imp.add_argument("--c-junction", type=float, default=1e-9, help="Junction capacitance in F (default: 1 nF).")
    p_imp.add_argument("--save-plot", type=str, default="", help="Path to export Bode/Nyquist plot image.")
    p_imp.add_argument("--output-dir", type=str, default="", help="Save outputs inside a timestamped folder.")
    p_imp.add_argument("--no-show", action="store_true", help="Do not display plot interactively.")

    args = parser.parse_args()

    if not args.command or args.command == "random":
        # Run random resistor network simulation
        out_dir = _prepare_output_dir(getattr(args, "output_dir", ""))
        df = run_random_sweep(
            nodes=getattr(args, "nodes", 400),
            edges_start=getattr(args, "edges_start", 1000),
            edges_stop=getattr(args, "edges_stop", 3000),
            edges_step=getattr(args, "edges_step", 200),
            dev_start=getattr(args, "dev_start", 1),
            dev_stop=getattr(args, "dev_stop", 10),
            seeds=getattr(args, "seeds", 100),
            mean=getattr(args, "mean", 10.0),
            dist=getattr(args, "dist", "lognormal"),
            x_node=getattr(args, "x_node", 0),
            target_node=getattr(args, "target_node", None),
            workers=getattr(args, "workers", None),
            verbose=True,
        )

        csv_path = args.save_csv if getattr(args, "save_csv", "") else (str(out_dir / "results.csv") if out_dir else "")
        plot_path = args.save_plot if getattr(args, "save_plot", "") else (str(out_dir / "plot.png") if out_dir else "")
        meta_path = args.save_meta if getattr(args, "save_meta", "") else (str(out_dir / "metadata.json") if out_dir else "")

        if csv_path:
            df.to_csv(csv_path, index=False)
            print(f"Saved CSV results to {csv_path}")

        if meta_path:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(vars(args), f, indent=2)
            print(f"Saved metadata to {meta_path}")

        plot_resistance_vs_edges(
            df,
            save_path=plot_path if plot_path else None,
            show=not getattr(args, "no_show", False),
        )

    elif args.command == "brick3d":
        out_dir = _prepare_output_dir(args.output_dir)
        df = run_brick3d_sweep(
            layers=args.layers,
            depths=args.depths,
            std=args.std,
            seeds=args.seeds,
            hotspot_frac=args.hotspot_frac,
            ext_r=args.ext_r,
            verbose=True,
        )

        csv_path = args.save_csv if args.save_csv else (str(out_dir / "brick3d_results.csv") if out_dir else "")
        plot_path = args.save_plot if args.save_plot else (str(out_dir / "scaling_plot.png") if out_dir else "")

        if csv_path:
            df.to_csv(csv_path, index=False)
            print(f"Saved CSV results to {csv_path}")

        plot_anisotropy_scaling(
            depths=df["depth"].values,
            r_oop=df["R_OOP"].values,
            r_ip=df["R_IP"].values,
            save_path=plot_path if plot_path else None,
            show=not args.no_show,
        )

    elif args.command == "impedance":
        out_dir = _prepare_output_dir(args.output_dir)
        frequencies = np.logspace(np.log10(args.f_min), np.log10(args.f_max), args.f_points)

        # Simple 4-node RC test circuit
        G = create_brick_lattice_3d(layers=2, depth=2, is_oop=True)[0]
        n_edges = G.number_of_edges()
        conds = np.full(n_edges, 1.0 / args.r_junction)
        caps = np.full(n_edges, args.c_junction)

        # Connect electrodes
        nodes = list(G.nodes())
        res_eis = solve_complex_impedance_spectrum(
            graph=G,
            node_a=nodes[-2],
            node_b=nodes[-1],
            frequencies=frequencies,
            conductances=conds,
            capacitances=caps,
        )

        plot_path = args.save_plot if args.save_plot else (str(out_dir / "bode_plot.png") if out_dir else "")
        plot_bode(
            frequencies=frequencies,
            z_complex=res_eis["Z_complex"],
            save_path=plot_path if plot_path else None,
            show=not args.no_show,
        )


if __name__ == "__main__":
    main()
