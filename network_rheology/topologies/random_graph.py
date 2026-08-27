"""
Fast generation of random resistor network graphs (Erdos-Renyi G(N, M)).
"""

from typing import Optional, Tuple, Union
import networkx as nx
import numpy as np


def create_random_edges_fast(
    nodes: int = 400,
    edges: int = 1000,
    rng: Optional[Union[np.random.Generator, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized generation of random edge index arrays (u, v) without Python list allocation overhead.

    Parameters
    ----------
    nodes : int
        Number of nodes in the graph (default 400).
    edges : int
        Number of unique undirected edges to sample.
    rng : Optional[Union[np.random.Generator, int]], optional
        Random generator instance or integer seed.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (u_indices, v_indices) arrays of shape (edges,).
    """
    if isinstance(rng, (int, np.integer)):
        gen = np.random.default_rng(rng)
    elif isinstance(rng, np.random.Generator):
        gen = rng
    else:
        gen = np.random.default_rng()

    max_possible_edges = nodes * (nodes - 1) // 2
    if edges > max_possible_edges:
        raise ValueError(
            f"Requested number of edges ({edges}) exceeds maximum possible for {nodes} nodes ({max_possible_edges})."
        )

    # Get upper triangular index pairs (all possible undirected edges)
    u_all, v_all = np.triu_indices(nodes, k=1)

    if edges == max_possible_edges:
        return u_all, v_all

    # Sample edge indices without replacement
    sampled_idx = gen.choice(max_possible_edges, size=edges, replace=False)
    return u_all[sampled_idx], v_all[sampled_idx]


def create_random_network(
    nodes: int = 400,
    edges: int = 1000,
    seed: Optional[int] = None,
) -> nx.Graph:
    """
    Generate a NetworkX graph with exactly `nodes` nodes (0 to nodes-1) and `edges` sampled uniformly at random.
    Guarantees all `nodes` are present in the graph structure even if isolated.

    Parameters
    ----------
    nodes : int
        Number of nodes (default 400).
    edges : int
        Number of edges to sample (default 1000).
    seed : Optional[int], optional
        Random seed for reproducibility.

    Returns
    -------
    nx.Graph
        NetworkX Graph with all `nodes` nodes and `edges` random edges.
    """
    u_idx, v_idx = create_random_edges_fast(nodes=nodes, edges=edges, rng=seed)

    G = nx.Graph()
    G.add_nodes_from(range(nodes))  # Ensure all nodes exist
    G.add_edges_from(zip(u_idx, v_idx))
    return G
