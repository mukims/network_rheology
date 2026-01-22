# -*- coding: utf-8 -*-
"""
Created on Thu May  9 15:04:19 2024

@author: shardul
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sc
import networkx as nx
from numpy.random import choice
import scipy.sparse.linalg as spla

def create_graph_from_sample(edges, seed, nodes=400):
    np.random.seed(seed)
    max_edges = nodes * (nodes - 1) // 2
    if edges > max_edges:
        raise ValueError(f"Requested number of edges ({edges}) exceeds the maximum possible ({max_edges}).")
    possible_edges = [(i, j) for i in range(nodes) for j in range(i + 1, nodes)]
    sample_indices = choice(len(possible_edges), edges, replace=False)
    sampled_edges = [possible_edges[i] for i in sample_indices]
    G = nx.Graph()
    G.add_edges_from(sampled_edges)
    return G
#nx.draw(create_graph_from_sample(1600, 23))
# Function to generate random values with a normal distribution
def generate_random_values(mean, std, seed, num_edges):
    np.random.seed(seed)
    return np.abs(np.random.normal(mean, std, num_edges))

# Function to compute the interaction matrix
def matrix_interactions(w, meanc, stdc, edges, seed, nodes=400):
    grid = create_graph_from_sample(edges, seed, nodes=nodes)
    num_edges = grid.number_of_edges()
    nodes = grid.number_of_nodes()
    cap_values = generate_random_values(meanc, stdc, seed, num_edges)
    
    adjM = sc.lil_matrix((nodes, nodes), dtype=np.complex128)
    edges2 = list(grid.edges())
    interactions = 1 / cap_values
    
    for edge_idx, (i, j) in enumerate(edges2):
        interaction_value = interactions[edge_idx]
        adjM[i, j] = interaction_value
        adjM[j, i] = interaction_value  # Ensure the matrix is symmetric
    
    adjM.setdiag(adjM.sum(axis=1).A1 - 0.0001)
    
    MI = spla.inv(adjM.tocsc())  # Using inv for CSC matrices

    return MI, nodes



#plt.figure(figsize=(12, 12))
#G1=create_graph_from_sample(1000, 1)
#print(matrix_interactions(0, 10, 1, 1200, 1))
#nx.draw(G1,pos=nx.spring_layout(G1),node_size=3,alpha = 0.5)
#nx.draw_networkx_nodes(G1,pos=nx.spring_layout(G1), nodelist=[0,399], node_color='red', node_size=10)
#plt.show()


def R(w, meanc, stdc, edges, seed, x, nodes=400):
    MI, nodes = matrix_interactions(w, meanc, stdc, edges, seed, nodes=nodes)
    
    if x >= nodes:
        raise IndexError(f"Node index out of bounds: x={x}, nodes={nodes}")
    
    return MI[x, x] + MI[nodes - 1, nodes - 1] - MI[x, nodes - 1] - MI[nodes - 1, x], nodes


def parse_args():
    parser = argparse.ArgumentParser(description="Random resistor network simulation.")
    parser.add_argument("--nodes", type=int, default=400, help="Number of nodes in the network.")
    parser.add_argument("--edges-start", type=int, default=1000, help="Starting edge count (inclusive).")
    parser.add_argument("--edges-stop", type=int, default=3000, help="Stopping edge count (exclusive).")
    parser.add_argument("--edges-step", type=int, default=200, help="Step size for edge count.")
    parser.add_argument("--dev-start", type=int, default=1, help="Starting disorder (std) value (inclusive).")
    parser.add_argument("--dev-stop", type=int, default=10, help="Stopping disorder (std) value (exclusive).")
    parser.add_argument("--seeds", type=int, default=100, help="Number of random seeds per edge count.")
    parser.add_argument("--mean", type=float, default=10.0, help="Mean of the normal distribution.")
    parser.add_argument("--x-node", type=int, default=0, help="Node index for resistance calculation.")
    parser.add_argument("--save-csv", type=str, default="", help="Path to save results as CSV (optional).")
    parser.add_argument("--save-plot", type=str, default="", help="Path to save plot (optional).")
    parser.add_argument("--output-dir", type=str, default="", help="Create a timestamped output folder and save CSV/plot inside it.")
    parser.add_argument("--save-meta", type=str, default="", help="Path to save run metadata as JSON (optional).")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot.")
    return parser.parse_args()


def run_simulation(args):
    results = []
    for dev in range(args.dev_start, args.dev_stop):
        print(f"Disorder (std) = {dev}")
        dev_results = []
        for edges in range(args.edges_start, args.edges_stop, args.edges_step):
            print(f"  Processing edges: {edges}")
            resistances = []
            for seed in range(args.seeds):
                try:
                    resistance, _ = R(0, args.mean, dev, edges, seed, args.x_node, nodes=args.nodes)
                    resistances.append(np.abs(resistance))
                except IndexError as e:
                    print(f"  Skipping seed {seed} due to index error: {e}")
            if resistances:
                dev_results.append([edges, np.mean(resistances)])
        results.append(dev_results)
    return results


def save_csv(results, args):
    if not args.save_csv:
        return
    with open(args.save_csv, "w", encoding="utf-8") as f:
        f.write("dev,edges,avg_resistance\n")
        for dev, dev_res in zip(range(args.dev_start, args.dev_stop), results):
            for edges, avg_res in dev_res:
                f.write(f"{dev},{int(edges)},{float(avg_res)}\n")


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
    for dev, dev_res in zip(range(args.dev_start, args.dev_stop), results):
        if not dev_res:
            continue
        dev_arr = np.array(dev_res)
        plt.plot(dev_arr[:, 0], dev_arr[:, 1], label=f"{dev}")

    plt.xlabel("size")
    plt.ylabel("Average Resistance")
    plt.legend(title="std")

    if args.save_plot:
        plt.savefig(args.save_plot, dpi=150, bbox_inches="tight")
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
