# Network Rheology

Simulates effective resistance in random resistor networks built on a 400-node graph. The script samples random edges, assigns random conductances from a normal distribution, constructs a symmetric interaction matrix, inverts it, and computes the effective resistance between a chosen node and the last node. It then averages resistance over multiple random seeds and plots resistance versus edge count for different disorder levels.

## Contents
- `random_resistor_network.py`: Main simulation and plotting script.
- `*.ipynb`: Exploratory notebooks.
- `output_*.png`: Generated plots (ignored by default in git).

## Requirements
Python 3.9+ recommended.

```bash
pip install -r requirements.txt
```

## Usage
Run the main script:

```bash
python random_resistor_network.py
```

This will:
- Sweep edge counts (1000 to 2800, step 200)
- Sweep disorder values (`dev` from 1 to 9)
- Average over 100 random seeds per edge count
- Plot average resistance vs. edge count

### CLI options
You can customize the run with flags:

```bash
python random_resistor_network.py --nodes 400 --edges-start 1000 --edges-stop 3000 --edges-step 200 \
  --dev-start 1 --dev-stop 10 --seeds 100 --mean 10 --x-node 0
```

Save a plot without showing it (useful for headless runs):

```bash
python random_resistor_network.py --save-plot results.png --no-show
```

Save numeric results to CSV:

```bash
python random_resistor_network.py --save-csv results.csv --no-show
```

Write both CSV and plot into a timestamped output folder:

```bash
python random_resistor_network.py --output-dir outputs --no-show
```

Save run metadata (all CLI args) to JSON:

```bash
python random_resistor_network.py --save-meta metadata.json --no-show
```

## Notes
- The network size is fixed at 400 nodes in the current script.
- Runtime can be significant due to repeated sparse matrix inversions.
- You can adjust parameters (node count, edge counts, number of seeds) inside `random_resistor_network.py`.
