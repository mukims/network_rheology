"""
2D and 3D brick-and-mortar / overlapping flake network topologies for 2D material composites.
"""

from typing import Dict, List, Optional, Tuple
import networkx as nx
import numpy as np


def _get_2d_flake_edges(layers: int) -> List[Tuple[int, int]]:
    """
    Generate intra-layer and inter-flake edge tuples for a 2D staggered flake layer.
    """
    edges = []
    # E0: Horizontal connections within rows
    for j in range(0, 8 * layers, 8):
        for i in range(7):
            edges.append((i + j, i + 1 + j))

    # E1: Staggered cross-connections between adjacent rows
    for j in range(0, 8 * (layers - 1), 8):
        for i in range(1, 7, 2):
            edges.append((j + i, j + 7 + i))
            edges.append((j + i, j + 7 + 2 + i))

    # E2: Periodic / boundary connectors
    for j in range(0, 8 * (layers - 1), 8):
        edges.append((j + 7, j + 14))

    return edges


def create_brick_lattice_2d(layers: int = 4, seed: Optional[int] = None) -> nx.Graph:
    """
    Create a 2D network of overlapping/staggered flakes (representing a 2D material sheet).

    Parameters
    ----------
    layers : int
        Number of unit cell blocks horizontally (default 4, creating 8*layers = 32 nodes).
    seed : Optional[int], optional
        Random seed (if randomized links are used).

    Returns
    -------
    nx.Graph
        2D flake network graph with 8*layers nodes.
    """
    if seed is not None:
        np.random.seed(seed)

    G_2d = nx.Graph()
    G_2d.add_nodes_from(range(8 * layers))
    edges = _get_2d_flake_edges(layers)
    G_2d.add_edges_from(edges)
    return G_2d


def create_brick_lattice_3d(
    layers: int = 4,
    depth: int = 10,
    is_oop: bool = True,
    seed: Optional[int] = None,
) -> Tuple[nx.Graph, int, int, Dict[Tuple[int, int], int]]:
    """
    Create a 3D brick-layered network of stacked 2D flake planes, with vertical inter-plane
    contacts and boundary electrode nodes for Out-of-Plane (OOP) or In-Plane (IP) transport.

    Parameters
    ----------
    layers : int
        Horizontal size parameter (number of blocks, each having 8 flakes, default 4).
    depth : int
        Number of stacked vertical layers / unit cells (default 10).
    is_oop : bool
        If True, attaches external electrodes for Out-of-Plane (OOP: top to bottom) transport.
        If False, attaches external electrodes for In-Plane (IP: left to right) transport.
    seed : Optional[int], optional
        Random seed.

    Returns
    -------
    Tuple[nx.Graph, int, int, Dict[Tuple[int, int], int]]
        - G_3d: The complete 3D network graph including bulk nodes and 2 external electrode nodes.
        - electrode_a: Node index for external electrode 1 (+1 A injection).
        - electrode_b: Node index for external electrode 2 (ground extraction).
        - node_mapping: Dictionary mapping (flake_index_in_2d, z_layer) to 3D node index.
    """
    G_2d = create_brick_lattice_2d(layers=layers, seed=seed)
    G_3d = nx.Graph()

    node_count = 0
    node_mapping: Dict[Tuple[int, int], int] = {}

    # Add bulk nodes across all vertical depth layers
    for z in range(depth):
        for node_2d in G_2d.nodes():
            G_3d.add_node(node_count, layer_2d=node_2d, z=z)
            node_mapping[(node_2d, z)] = node_count
            node_count += 1

    # In-plane edges within each vertical layer
    for edge in G_2d.edges():
        for z in range(depth):
            G_3d.add_edge(node_mapping[(edge[0], z)], node_mapping[(edge[1], z)], edge_type="in_plane")

    # Vertical inter-layer contacts: staggered odd flakes in layer z connect to even flakes in layer z+1
    for z in range(depth - 1):
        for node_2d in G_2d.nodes():
            if node_2d % 2 == 1:
                for neighbor in G_2d.neighbors(node_2d):
                    if neighbor % 2 == 0:
                        G_3d.add_edge(
                            node_mapping[(node_2d, z)],
                            node_mapping[(neighbor, z + 1)],
                            edge_type="vertical",
                        )

    # Attach two external electrode contact nodes
    electrode_a = node_count
    G_3d.add_node(electrode_a, is_electrode=True, name="Electrode_A")
    node_count += 1

    electrode_b = node_count
    G_3d.add_node(electrode_b, is_electrode=True, name="Electrode_B")

    if is_oop:
        # Out-of-Plane: Electrode A connects to bottom layer (z=0) even nodes
        #               Electrode B connects to top layer (z=depth-1) odd nodes
        for node_2d in G_2d.nodes():
            if node_2d % 2 == 0:
                G_3d.add_edge(node_mapping[(node_2d, 0)], electrode_a, edge_type="electrode")
            if node_2d % 2 == 1:
                G_3d.add_edge(node_mapping[(node_2d, depth - 1)], electrode_b, edge_type="electrode")
    else:
        # In-Plane: Electrode A connects to leftmost flakes (node % 8 == 0) across all z
        #           Electrode B connects to rightmost flakes (node % 8 == 7) across all z
        for z in range(depth):
            for node_2d in G_2d.nodes():
                if node_2d % 8 == 0:
                    G_3d.add_edge(node_mapping[(node_2d, z)], electrode_a, edge_type="electrode")
                elif node_2d % 8 == 7:
                    G_3d.add_edge(node_mapping[(node_2d, z)], electrode_b, edge_type="electrode")

    return G_3d, electrode_a, electrode_b, node_mapping
