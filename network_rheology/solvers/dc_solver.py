"""
High-performance DC resistance and potential/current solvers using sparse reduced Laplacian linear systems.
"""

from typing import Dict, Optional, Tuple, Union
import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def solve_effective_resistance_fast(
    num_nodes: int,
    u_indices: np.ndarray,
    v_indices: np.ndarray,
    conductances: np.ndarray,
    node_a: int,
    node_b: int,
    check_connectivity: bool = True,
) -> float:
    """
    Ultra-fast calculation of effective two-point resistance between node_a and node_b
    using vectorized reduced Laplacian linear solve.

    Parameters
    ----------
    num_nodes : int
        Total number of nodes in the network (nodes indexed 0 to num_nodes - 1).
    u_indices : np.ndarray
        Array of start node indices for each edge.
    v_indices : np.ndarray
        Array of end node indices for each edge.
    conductances : np.ndarray
        Array of conductance values g_ij for each edge.
    node_a : int
        Source node index (injection of +1 A).
    node_b : int
        Drain / ground reference node index (extraction of -1 A).
    check_connectivity : bool, optional
        Whether to check graph connectivity between node_a and node_b, default True.

    Returns
    -------
    float
        Effective two-point resistance R_ab (Ohm). Returns float('inf') if disconnected.
    """
    if node_a == node_b:
        return 0.0

    if node_a >= num_nodes or node_b >= num_nodes or node_a < 0 or node_b < 0:
        raise IndexError(f"Node index out of bounds: node_a={node_a}, node_b={node_b}, num_nodes={num_nodes}")

    if len(u_indices) == 0:
        return float("inf")

    u = np.asarray(u_indices, dtype=np.int32)
    v = np.asarray(v_indices, dtype=np.int32)
    g = np.asarray(conductances, dtype=np.float64)

    if check_connectivity:
        # Quick BFS / graph connectivity check
        G_quick = nx.Graph()
        G_quick.add_nodes_from(range(num_nodes))
        G_quick.add_edges_from(zip(u, v))
        if not nx.has_path(G_quick, node_a, node_b):
            return float("inf")

    # Build the sparse Laplacian matrix L = D - A directly in COO format
    # Off-diagonals: -g for (u, v) and (v, u)
    # Diagonals: +g for (u, u) and (v, v)
    rows = np.concatenate([u, v, u, v])
    cols = np.concatenate([v, u, u, v])
    data = np.concatenate([-g, -g, g, g])

    # Construct the sparse Laplacian
    L = sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes)).tocsc()

    # Form reduced Laplacian by grounding node_b (remove row and column node_b)
    # Index mapping for remaining nodes:
    nodes_kept = np.array([i for i in range(num_nodes) if i != node_b], dtype=np.int32)
    L_red = L[nodes_kept, :][:, nodes_kept].tocsc()

    # Build external current injection vector: +1.0 A at node_a
    rhs = np.zeros(num_nodes - 1, dtype=np.float64)
    mapped_a = node_a if node_a < node_b else node_a - 1
    rhs[mapped_a] = 1.0

    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=spla.MatrixRankWarning)
            v_sol = spla.spsolve(L_red, rhs)
        r_eff = float(v_sol[mapped_a])
        return max(0.0, r_eff) if np.isfinite(r_eff) else float("inf")
    except (spla.MatrixRankWarning, RuntimeError, ValueError):
        return float("inf")


def solve_effective_resistance(
    graph: nx.Graph,
    node_a: int,
    node_b: int,
    conductances: Optional[Union[np.ndarray, Dict[Tuple[int, int], float]]] = None,
    weight_attr: str = "weight",
    return_fields: bool = False,
) -> Union[float, Tuple[float, np.ndarray, np.ndarray]]:
    """
    Solve effective two-point resistance on a NetworkX graph with full physical field reconstruction.

    Parameters
    ----------
    graph : nx.Graph
        NetworkX graph representing the resistor network.
    node_a : int
        Source node index where +1 A is injected.
    node_b : int
        Reference node index grounded at 0 V where -1 A is extracted.
    conductances : Optional[Union[np.ndarray, Dict]], optional
        Array or dictionary of conductance values. If None, reads from graph edge attribute `weight_attr`.
    weight_attr : str, optional
        Edge attribute name for conductances when conductances is None, default 'weight'.
    return_fields : bool, optional
        If True, returns (R_eff, potentials_array, edge_currents_array).

    Returns
    -------
    Union[float, Tuple[float, np.ndarray, np.ndarray]]
        Effective resistance, or tuple of (R_eff, V_nodes, I_edges).
    """
    if node_a == node_b:
        if return_fields:
            num_nodes = graph.number_of_nodes()
            return 0.0, np.zeros(num_nodes), np.zeros(graph.number_of_edges())
        return 0.0

    if not nx.has_path(graph, node_a, node_b):
        if return_fields:
            num_nodes = max(graph.nodes()) + 1 if graph.number_of_nodes() > 0 else 0
            return float("inf"), np.full(num_nodes, np.nan), np.full(graph.number_of_edges(), np.nan)
        return float("inf")

    num_nodes = max(graph.nodes()) + 1
    edges = list(graph.edges())
    num_edges = len(edges)

    if conductances is None:
        g = np.array([graph[u][v].get(weight_attr, 1.0) for u, v in edges], dtype=np.float64)
    elif isinstance(conductances, dict):
        g = np.array([conductances.get((u, v), conductances.get((v, u), 1.0)) for u, v in edges], dtype=np.float64)
    else:
        g = np.asarray(conductances, dtype=np.float64)
        if len(g) != num_edges:
            raise ValueError(f"Conductance array length ({len(g)}) does not match edge count ({num_edges}).")

    u_indices = np.array([e[0] for e in edges], dtype=np.int32)
    v_indices = np.array([e[1] for e in edges], dtype=np.int32)

    # Build Laplacian
    rows = np.concatenate([u_indices, v_indices, u_indices, v_indices])
    cols = np.concatenate([v_indices, u_indices, u_indices, v_indices])
    data = np.concatenate([-g, -g, g, g])
    L = sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes)).tocsc()

    # Ground node_b
    nodes_kept = np.array([i for i in range(num_nodes) if i != node_b], dtype=np.int32)
    L_red = L[nodes_kept, :][:, nodes_kept].tocsc()

    mapped_a = node_a if node_a < node_b else node_a - 1
    rhs = np.zeros(num_nodes - 1, dtype=np.float64)
    rhs[mapped_a] = 1.0

    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=spla.MatrixRankWarning)
            v_red = spla.spsolve(L_red, rhs)
        r_eff = max(0.0, float(v_red[mapped_a]))
        if not np.isfinite(r_eff):
            return float("inf") if not return_fields else (float("inf"), np.full(num_nodes, np.nan), np.full(num_edges, np.nan))
    except (spla.MatrixRankWarning, RuntimeError, ValueError):
        return float("inf") if not return_fields else (float("inf"), np.full(num_nodes, np.nan), np.full(num_edges, np.nan))

    if not return_fields:
        return r_eff

    # Reconstruct full node voltage vector V (with V[node_b] = 0.0)
    V_full = np.zeros(num_nodes, dtype=np.float64)
    V_full[nodes_kept] = v_red

    # Calculate branch currents I_ij = g_ij * (V_i - V_j)
    I_edges = g * (V_full[u_indices] - V_full[v_indices])

    return r_eff, V_full, I_edges


def compute_current_and_dissipation(
    graph: nx.Graph,
    node_a: int,
    node_b: int,
    conductances: Optional[Union[np.ndarray, Dict[Tuple[int, int], float]]] = None,
    weight_attr: str = "weight",
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Compute full electrical field metrics including node potentials, branch currents,
    and local Joule heating (power dissipation) for all edges in the network.

    Parameters
    ----------
    graph : nx.Graph
        The resistor network graph.
    node_a : int
        Source node index (+1 A injected).
    node_b : int
        Ground reference node index (-1 A extracted).
    conductances : Optional[Union[np.ndarray, Dict]], optional
        Conductances for each edge.
    weight_attr : str, optional
        Edge attribute name for conductance if conductances is None.

    Returns
    -------
    Dict[str, Union[float, np.ndarray]]
        Dictionary containing:
        - "R_eff": Effective resistance (Ohm).
        - "V_nodes": Voltage at each node (V).
        - "I_edges": Signed branch current through each edge (A).
        - "I_abs_nodes": Total absolute current passing through each node (A).
        - "P_joule_edges": Power dissipation per edge (W).
        - "P_total": Total power dissipation sum(I^2 / g) (W), equal to R_eff * I_total^2.
    """
    r_eff, V_nodes, I_edges = solve_effective_resistance(
        graph=graph,
        node_a=node_a,
        node_b=node_b,
        conductances=conductances,
        weight_attr=weight_attr,
        return_fields=True,
    )

    if np.isinf(r_eff) or np.isnan(r_eff):
        num_nodes = max(graph.nodes()) + 1 if graph.number_of_nodes() > 0 else 0
        num_edges = graph.number_of_edges()
        return {
            "R_eff": float("inf"),
            "V_nodes": np.full(num_nodes, np.nan),
            "I_edges": np.full(num_edges, np.nan),
            "I_abs_nodes": np.full(num_nodes, np.nan),
            "P_joule_edges": np.full(num_edges, np.nan),
            "P_total": float("inf"),
        }

    edges = list(graph.edges())
    num_nodes = len(V_nodes)
    u_idx = np.array([e[0] for e in edges], dtype=np.int32)
    v_idx = np.array([e[1] for e in edges], dtype=np.int32)

    if conductances is None:
        g = np.array([graph[u][v].get(weight_attr, 1.0) for u, v in edges], dtype=np.float64)
    elif isinstance(conductances, dict):
        g = np.array([conductances.get((u, v), conductances.get((v, u), 1.0)) for u, v in edges], dtype=np.float64)
    else:
        g = np.asarray(conductances, dtype=np.float64)

    # Local Joule dissipation P_ij = I_ij^2 / g_ij = g_ij * (V_i - V_j)^2
    P_joule_edges = g * (V_nodes[u_idx] - V_nodes[v_idx]) ** 2
    P_total = float(np.sum(P_joule_edges))

    # Calculate absolute nodal throughput current
    I_abs_nodes = np.zeros(num_nodes, dtype=np.float64)
    np.add.at(I_abs_nodes, u_idx, np.abs(I_edges))
    np.add.at(I_abs_nodes, v_idx, np.abs(I_edges))

    return {
        "R_eff": r_eff,
        "V_nodes": V_nodes,
        "I_edges": I_edges,
        "I_abs_nodes": I_abs_nodes,
        "P_joule_edges": P_joule_edges,
        "P_total": P_total,
    }
