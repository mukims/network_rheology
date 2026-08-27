"""
Regular grid topologies (1D chain, 2D square lattice, 3D cubic lattice) for validation.
"""

import networkx as nx


def create_1d_chain(nodes: int = 10, pbc: bool = False) -> nx.Graph:
    """
    Create a 1D chain resistor network.
    """
    return nx.cycle_graph(nodes) if pbc else nx.path_graph(nodes)


def create_2d_grid(rows: int = 5, cols: int = 5, pbc: bool = False) -> nx.Graph:
    """
    Create a 2D square lattice resistor network.
    Nodes are indexed 0 to rows*cols - 1 with (r, c) -> r*cols + c.
    """
    G = nx.grid_2d_graph(rows, cols, periodic=pbc)
    # Convert node labels (r, c) to integer indices
    mapping = {(r, c): r * cols + c for r in range(rows) for c in range(cols)}
    return nx.relabel_nodes(G, mapping)


def create_3d_cubic_grid(nx_nodes: int = 4, ny_nodes: int = 4, nz_nodes: int = 4, pbc: bool = False) -> nx.Graph:
    """
    Create a 3D cubic lattice resistor network.
    """
    G = nx.grid_graph(dim=[nx_nodes, ny_nodes, nz_nodes], periodic=pbc)
    mapping = {
        (x, y, z): (x * ny_nodes * nz_nodes) + (y * nz_nodes) + z
        for x in range(nx_nodes)
        for y in range(ny_nodes)
        for z in range(nz_nodes)
    }
    return nx.relabel_nodes(G, mapping)
