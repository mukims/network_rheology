# Network Rheology

A high-performance computational physics and materials science framework for modeling **electrical transport, effective resistance, current density fields, and AC Electrical Impedance Spectroscopy (EIS)** in disordered and structured resistor networks.

---

## Table of Contents
- [1. Overview](#1-overview)
- [2. Physical & Mathematical Foundations](#2-physical--mathematical-foundations)
  - [Kirchhoff's Current Law & The Graph Laplacian](#kirchhoffs-current-law--the-graph-laplacian)
  - [The Reduced Laplacian Formulation](#the-reduced-laplacian-formulation)
  - [AC Electrical Impedance Spectroscopy (EIS)](#ac-electrical-impedance-spectroscopy-eis)
  - [Joule Power Dissipation & Branch Currents](#joule-power-dissipation--branch-currents)
- [3. Concrete Physical Applications & Examples](#3-concrete-physical-applications--examples)
  - [Physical Example 1: 2D Material Nanosheet Assemblies (Graphene / MXenes)](#physical-example-1-2d-material-nanosheet-assemblies-graphene--mxenes)
  - [Physical Example 2: Percolation in Disordered Nanowire Networks & Carbon Nanotubes](#physical-example-2-percolation-in-disordered-nanowire-networks--carbon-nanotubes)
  - [Physical Example 3: Frequency-Dependent Impedance in Capacitive Junctions](#physical-example-3-frequency-dependent-impedance-in-capacitive-junctions)
- [4. Repository Architecture](#4-repository-architecture)
- [5. Installation](#5-installation)
- [6. Command-Line Interface (CLI) Guide](#6-command-line-interface-cli-guide)
  - [A. Random Resistor Network Sweeps (`random`)](#a-random-resistor-network-sweeps-random)
  - [B. 3D Flake Assembly Anisotropy (`brick3d`)](#b-3d-flake-assembly-anisotropy-brick3d)
  - [C. AC Impedance Spectroscopy (`impedance`)](#c-ac-impedance-spectroscopy-impedance)
  - [D. Standalone Accelerated Script (`random_resistor_network.py`)](#d-standalone-accelerated-script-random_resistor_networkpy)
- [7. Python API Tutorial & Examples](#7-python-api-tutorial--examples)
  - [Example A: Computing Resistance and Potential Fields](#example-a-computing-resistance-and-potential-fields)
  - [Example B: 3D Flake Anisotropy & Current Visualization](#example-b-3d-flake-anisotropy--current-visualization)
  - [Example C: AC Impedance Spectroscopy & Parameter Fitting](#example-c-ac-impedance-spectroscopy--parameter-fitting)
- [8. Testing & Verification](#8-testing--verification)
- [9. References & Further Reading](#9-references--further-reading)

---

## 1. Overview

Disordered and structured conductive networks are central to many emerging technologies in energy storage, printed electronics, flexible sensors, and 2D material composites (e.g. graphene, MXenes, transition metal dichalcogenides, and silver nanowires). 

In such systems, macroscopic electrical properties (resistance, impedance, conductivity, and percolation) emerge from microscopic interactions across thousands of inter-flake and inter-particle junctions.

**Network Rheology** provides:
* **Mathematically Exact Solvers**: Formulated directly via the reduced graph Laplacian without artificial grounding damping.
* **Extreme Performance**: $\mathcal{O}(N^{1.5})$ sparse linear algebra replaces $\mathcal{O}(N^3)$ dense inversion, with multi-core parallelization for sweeps over thousands of realizations in seconds.
* **Diverse Microstructural Topologies**: Erdős–Rényi random graphs, 2D/3D staggered "brick-and-mortar" flake assemblies, and 1D/2D/3D regular lattices.
* **Full Field Diagnostics**: Node potentials $V_i$, branch currents $I_{ij}$, and local Joule heating $P_{ij}$.
* **AC Impedance & Inversion**: Complex admittance spectra $Y(\omega) = G + i\omega C$, Nyquist and Bode plots, and parameter fitting tools.

---

## 2. Physical & Mathematical Foundations

### Kirchhoff's Current Law & The Graph Laplacian
Consider an electrical network represented by a graph $G = (V, E)$ with $N$ nodes and $M$ edges. Each edge $(i, j) \in E$ has a conductance $g_{ij} = 1/R_{ij} > 0$.

By Kirchhoff's Current Law (KCL), the net current flowing into node $i$ from all neighboring nodes $j \in \mathcal{N}(i)$ must equal the externally injected current $I_i^{\text{ext}}$:
$$\sum_{j \in \mathcal{N}(i)} g_{ij} (V_i - V_j) = I_i^{\text{ext}}$$

In matrix notation, this forms the linear system:
$$L V = I^{\text{ext}}$$
where $L$ is the **Kirchhoff Graph Laplacian Matrix** ($N \times N$):
$$L_{ij} = \begin{cases} \sum_{k \neq i} g_{ik}, & \text{if } i = j \text{ (diagonal degree)} \\ -g_{ij}, & \text{if } i \neq j \text{ and } (i, j) \in E \\ 0, & \text{otherwise} \end{cases}$$

### The Reduced Laplacian Formulation
Since the rows and columns of $L$ sum to zero ($\sum_j L_{ij} = 0$), $L$ has a zero eigenvalue corresponding to the constant potential eigenvector $\mathbf{1}$. To solve for physical potentials without introducing artificial ground leakage, we set a reference drain node $b$ as ground ($V_b = 0$).

Deleting row $b$ and column $b$ yields the **Reduced Laplacian** $L_{\text{red}}$, which is symmetric, strictly positive-definite, and non-singular for any connected component:
$$L_{\text{red}} V_{\text{red}} = I_{\text{red}}^{\text{ext}}$$

Injecting a test current of $+1\,\text{A}$ at source node $a$ and extracting $-1\,\text{A}$ at ground node $b$ ($I_a = +1, I_b = -1$) yields the exact two-point effective resistance directly:
$$R_{ab} = V_a - V_b = V_a \quad (\text{since } V_b = 0)$$

### AC Electrical Impedance Spectroscopy (EIS)
When junctions possess both conductive pathways (resistance $R_{ij} = 1/g_{ij}$) and dielectric/interfacial capacitance ($c_{ij}$), the system responds to alternating current (AC) at frequency $f$ ($\omega = 2\pi f$).

The complex branch admittance is:
$$y_{ij}(\omega) = g_{ij} + i \omega c_{ij}$$

The complex nodal admittance matrix $Y(\omega) = G + i\omega C$ is constructed as:
$$Y_{ij}(\omega) = -y_{ij}(\omega) \quad (i \neq j), \qquad Y_{ii}(\omega) = \sum_{k \neq i} y_{ik}(\omega)$$

Solving the reduced complex linear system $Y_{\text{red}}(\omega) V_{\text{red}}(\omega) = I_{\text{red}}^{\text{ext}}$ yields the complex two-point impedance:
$$Z_{ab}(\omega) = Z'(\omega) + i Z''(\omega) = V_a(\omega)$$
* **Real Part (In-phase / Resistance)**: $Z'(\omega) = \text{Re}(Z_{ab}(\omega))$
* **Imaginary Part (Out-of-phase / Reactance)**: $Z''(\omega) = \text{Im}(Z_{ab}(\omega))$
* **Magnitude**: $|Z(\omega)| = \sqrt{Z'^2 + Z''^2}$
* **Phase Angle**: $\theta(\omega) = \arctan\left(\frac{Z''(\omega)}{Z'(\omega)}\right)$

### Joule Power Dissipation & Branch Currents
Once node voltages $V$ are computed:
* **Branch Current** through contact $(i, j)$:
  $$I_{ij} = g_{ij} (V_i - V_j)$$
* **Local Joule Heating (Power Dissipation)**:
  $$P_{ij} = I_{ij}^2 R_{ij} = g_{ij} (V_i - V_j)^2$$
* **Global Energy Conservation**:
  $$\sum_{(i,j) \in E} P_{ij} = R_{ab} \cdot (I^{\text{ext}})^2 = R_{ab}$$

---

## 3. Concrete Physical Applications & Examples

### Physical Example 1: 2D Material Nanosheet Assemblies (Graphene / MXenes)

```
        Electrode A (+1 A) [Top Surface]
=================================================
 [ Flake Layer 4 ]  --- (Overlap Contact) ---
-------------------------------------------------
 [ Flake Layer 3 ]  --- (Overlap Contact) ---
-------------------------------------------------
 [ Flake Layer 2 ]  --- (Overlap Contact) ---
-------------------------------------------------
 [ Flake Layer 1 ]  --- (Overlap Contact) ---
=================================================
        Electrode B (0 V)   [Bottom Surface]
```

* **Physical Context**: Solution-processed thin films and membranes of 2D materials (graphene, MXene $\text{Ti}_3\text{C}_2\text{T}_x$, $\text{MoS}_2$) consist of horizontally staggered, overlapping flakes forming a "brick-and-mortar" structure.
* **Anisotropic Transport ($R_{\text{oop}}$ vs $R_{\text{ip}}$)**:
  - **In-Plane (IP)**: Current travels along overlapping continuous horizontal networks.
  - **Out-of-Plane (OOP)**: Current must cross vertical van der Waals / tunneling junctions across every layer.
* **Scaling Law**: As the film thickness (number of flake layers $N$) increases, the OOP-to-IP resistance ratio scales quadratically:
  $$\frac{R_{\text{oop}}}{R_{\text{ip}}} \propto N^2$$
* **Hotspots & Local Bottlenecks**: A small fraction of low-resistance junctions ("hotspots", e.g., pinholes or direct metallic contacts) dramatically lowers the macroscopic OOP resistance and produces localized current crowding.

---

### Physical Example 2: Percolation in Disordered Nanowire Networks & Carbon Nanotubes

```
  (Source Node 0)                                 (Drain Node N-1)
       [0] ---- [2]           [5] ---- [7]             [N-1]
         \     /  \          /           \            /
          [1] ---- [3] ---- [4] --------- [6] ------ [8]
               (Conductive Percolating Backbone)
```

* **Physical Context**: Transparent conductive electrodes, flexible touchscreens, and conductive polymer composites use random networks of silver nanowires (AgNWs) or carbon nanotubes (CNTs).
* **Percolation Threshold ($p_c$)**: Below a critical edge density $M_c$, the network is disconnected ($R_{\text{eff}} = \infty$). Above $M_c$, a giant connected component forms and resistance scales as:
  $$R_{\text{eff}} \propto (M - M_c)^{-t}$$
  where $t$ is the universal transport critical exponent ($t \approx 1.3$ in 2D, $t \approx 2.0$ in 3D).
* **Contact Resistance Disorder**: Junction conductances vary across orders of magnitude due to contact pressure, oxide barriers, and organic surfactant residues, described physically by a **Log-Normal distribution** ($\ln g \sim \mathcal{N}(\mu, \sigma^2)$).

---

### Physical Example 3: Frequency-Dependent Impedance in Capacitive Junctions

```
                      +----[ R_junction ]----+
                      |                      |
--[ R_nanosheet ]-----+                      +--
                      |                      |
                      +----[   C_gap    ]----+
```

* **Physical Context**: In AC impedance spectroscopy of colloidal nanosheets, battery electrodes, or biological tissue rheology, flake-to-flake junctions act as parallel resistor-capacitor ($R \parallel C$) elements in series with the intrinsic flake sheet resistance ($R_s$).
* **Spectroscopic Response**:
  - **Low Frequencies ($f \to 0$)**: Capacitors block current ($Z_C \to \infty$); transport is dominated by junction resistance ($Z \approx R_s + R_j$).
  - **High Frequencies ($f \to \infty$)**: Capacitors short-circuit ($Z_C \to 0$); transport is limited only by intrinsic sheet resistance ($Z \approx R_s$).
  - **Characteristic Cutoff Frequency**: $f_c = \frac{1}{2\pi R_j C_j}$, marking the semicircle peak on the Nyquist plot.

---

## 4. Repository Architecture

```
network_rheology/
├── pyproject.toml                         # Package configuration & dependencies
├── requirements.txt                       # Core requirements (numpy, scipy, matplotlib, networkx, pandas, pytest)
├── README.md                              # Comprehensive guide and documentation
├── random_resistor_network.py             # High-speed CLI simulation script (backward-compatible)
│
├── network_rheology/                      # Core Python Package
│   ├── __init__.py                        # Top-level API exports
│   ├── cli.py                             # Unified multi-command CLI
│   ├── distributions.py                   # Log-normal, truncated normal, uniform, hotspot generators
│   │
│   ├── solvers/
│   │   ├── __init__.py
│   │   ├── dc_solver.py                   # Reduced Laplacian solver, KCL, branch currents, Joule heating
│   │   └── ac_solver.py                   # Complex admittance Y(w) solver for AC impedance (EIS)
│   │
│   ├── topologies/
│   │   ├── __init__.py
│   │   ├── random_graph.py                # Vectorized Erdos-Renyi G(N, M) generator
│   │   ├── brick_lattice.py               # 2D & 3D brick-and-mortar / flake network generator
│   │   └── grid_lattice.py                # 1D chain, 2D square, 3D cubic lattice generators
│   │
│   ├── fitting/
│   │   ├── __init__.py
│   │   └── impedance_fitting.py           # Equivalent circuit parameter optimization (Rs, Rj, Cj)
│   │
│   └── visualization/
│       ├── __init__.py
│       ├── plots.py                       # Resistance sweeps, error bands, Nyquist, Bode, anisotropy plots
│       └── field_viewer.py                # 3D spatial current density and voltage viewer
│
├── tests/                                 # Automated Test Suite (21 unit & integration tests)
│   ├── __init__.py
│   ├── test_dc_solver.py                  # Series, parallel, triangle, Wheatstone bridge benchmarks
│   ├── test_ac_solver.py                  # Analytical RC circuit & parameter fitting benchmarks
│   ├── test_topologies.py                 # Graph node retention & flake connectivity tests
│   ├── test_distributions.py              # Statistical moment verification tests
│   └── test_cli.py                        # CLI end-to-end integration tests
│
└── notebooks/                             # Exploratory Jupyter Notebooks
    ├── Resistor_network_annotated_final.ipynb
    ├── 3D_brick_layer.ipynb
    └── random_network.ipynb
```

---

## 5. Installation

```bash
# Clone the repository
git clone https://github.com/mukims/network_rheology.git
cd network_rheology

# Install dependencies
pip install -r requirements.txt

# (Optional) Install the package in editable development mode
pip install -e .
```

---

## 6. Command-Line Interface (CLI) Guide

### A. Random Resistor Network Sweeps (`random`)

Simulates effective resistance over edge density sweeps and disorder levels with multi-core parallel processing:

```bash
python -m network_rheology.cli random \
  --nodes 400 \
  --edges-start 1000 \
  --edges-stop 3000 \
  --edges-step 200 \
  --dev-start 1 \
  --dev-stop 10 \
  --seeds 100 \
  --mean 10.0 \
  --dist lognormal \
  --output-dir outputs \
  --no-show
```

**Key Arguments:**
* `--nodes`: Number of nodes in graph (default: `400`).
* `--edges-start/stop/step`: Sweep range for edge connectivity.
* `--dev-start/stop`: Sweep range for disorder standard deviation (`std`).
* `--dist`: Conductance distribution (`lognormal`, `truncated_normal`, `uniform`, `constant`).
* `--output-dir`: Creates a timestamped directory containing `results.csv`, `plot.png`, and `metadata.json`.

---

### B. 3D Flake Assembly Anisotropy (`brick3d`)

Simulates 3D stacked flake networks to calculate In-Plane ($R_{\text{ip}}$) vs Out-of-Plane ($R_{\text{oop}}$) transport and anisotropy scaling prefactors:

```bash
python -m network_rheology.cli brick3d \
  --layers 4 \
  --depths 1 2 4 8 12 16 20 \
  --std 0.5 \
  --seeds 20 \
  --hotspot-frac 0.05 \
  --save-plot anisotropy_scaling.png \
  --save-csv anisotropy_results.csv \
  --no-show
```

---

### C. AC Impedance Spectroscopy (`impedance`)

Generates frequency sweeps for AC complex impedance and outputs Bode/Nyquist diagrams:

```bash
python -m network_rheology.cli impedance \
  --f-min 0.1 \
  --f-max 1e7 \
  --f-points 50 \
  --r-sheet 50.0 \
  --r-junction 5000.0 \
  --c-junction 1e-9 \
  --save-plot impedance_bode.png \
  --no-show
```

---

### D. Standalone Accelerated Script (`random_resistor_network.py`)

A drop-in, backward-compatible script powered by the fast solver engine:

```bash
python random_resistor_network.py --nodes 400 --edges-start 1000 --edges-stop 3000 --edges-step 200 --seeds 100
```

---

## 7. Python API Tutorial & Examples

### Example A: Computing Resistance and Potential Fields

```python
import networkx as nx
from network_rheology.solvers.dc_solver import solve_effective_resistance, compute_current_and_dissipation

# Create a 5-node ring circuit with 2 Ohm resistors
G = nx.cycle_graph(5)
conductance = 1.0 / 2.0  # 0.5 S

# Compute effective resistance between node 0 and node 2
# Path 1: 0-1-2 (4 Ohm), Path 2: 0-4-3-2 (6 Ohm) -> R_eff = (4 * 6) / 10 = 2.4 Ohm
r_eff = solve_effective_resistance(G, node_a=0, node_b=2, conductances=[conductance]*5)
print(f"Effective Resistance: {r_eff:.2f} Ohm")  # 2.40 Ohm

# Compute full potential and dissipation field
field = compute_current_and_dissipation(G, node_a=0, node_b=2, conductances=[conductance]*5)
print("Node Potentials (V):", field["V_nodes"])
print("Branch Currents (A):", field["I_edges"])
print("Total Joule Power (W):", field["P_total"])  # Exactly equals R_eff * 1.0^2
```

---

### Example B: 3D Flake Anisotropy & Current Visualization

```python
import numpy as np
from network_rheology.topologies.brick_lattice import create_brick_lattice_3d
from network_rheology.solvers.dc_solver import solve_effective_resistance, compute_current_and_dissipation
from network_rheology.visualization.field_viewer import plot_3d_flake_currents

depth = 10
# Generate Out-of-Plane (OOP) and In-Plane (IP) flake networks
G_oop, ea_oop, eb_oop, _ = create_brick_lattice_3d(layers=4, depth=depth, is_oop=True)
G_ip, ea_ip, eb_ip, _ = create_brick_lattice_3d(layers=4, depth=depth, is_oop=False)

r_oop = solve_effective_resistance(G_oop, ea_oop, eb_oop)
r_ip = solve_effective_resistance(G_ip, ea_ip, eb_ip)
print(f"Anisotropy Ratio (R_OOP / R_IP): {r_oop / r_ip:.2f}")

# Compute and visualize 3D current distribution across the flake stack
field_oop = compute_current_and_dissipation(G_oop, ea_oop, eb_oop)
fig = plot_3d_flake_currents(field_oop["I_abs_nodes"], depth=depth, show=False)
fig.savefig("current_distribution_3d.png")
```

---

### Example C: AC Impedance Spectroscopy & Parameter Fitting

```python
import numpy as np
import matplotlib.pyplot as plt
from network_rheology.fitting.impedance_fitting import fit_equivalent_circuit_rc
from network_rheology.visualization.plots import plot_nyquist, plot_bode

# Generate synthetic experimental impedance data
freqs = np.logspace(0, 7, 50)
omega = 2.0 * np.pi * freqs
true_rs = 45.0      # 45 Ohm sheet resistance
true_rj = 12000.0   # 12 kOhm junction resistance
true_cj = 1.5e-9    # 1.5 nF junction capacitance

z_synthetic = true_rs + true_rj / (1.0 + 1j * omega * true_rj * true_cj)

# Fit equivalent circuit parameters to experimental curve
fit_results = fit_equivalent_circuit_rc(frequencies=freqs, target_z=z_synthetic)
print("Fit Success:", fit_results["success"])
print(f"Extracted R_sheet:    {fit_results['R_sheet']:.2f} Ohm (True: {true_rs})")
print(f"Extracted R_junction: {fit_results['R_junction']:.2f} Ohm (True: {true_rj})")
print(f"Extracted C_junction: {fit_results['C_junction']*1e9:.2f} nF (True: {true_cj*1e9})")

# Generate publication-quality plots
plot_nyquist(freqs, z_synthetic, save_path="nyquist_fit.png", show=False)
plot_bode(freqs, z_synthetic, save_path="bode_fit.png", show=False)
```

---
