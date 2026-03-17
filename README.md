<h1 align="center">
dK-series <br/>  
<i>A family of reference models for unweighted graphs</i>
</h1>

<p align="center">
<a href="https://github.com/kazuibasou/dk-series/blob/main/LICENSE" target="_blank">
<img alt="License: MIT" src="https://img.shields.io/github/license/kazuibasou/dk-series">
</a>

<a href="https://doi.org/10.1109/ACCESS.2024.3459830" target="_blank">
<img alt="DOI" src="https://img.shields.io/badge/DOI-10.1109%2FACCESS.2024.3459830-blue.svg">
</a>

</p>


# dK-series

dK-series [1, 2] is a family of randomization methods (a.k.a. reference models and null models) for unweighted graphs (or pairwise networks).
The dK-series produces randomized networks that preserve up to the individual node's degree, node's degree correlation, and node's clustering coefficient of the given unweighted network, depending on the parameter value *d* = 0, 1, 1.5, 2, or 2.5.
In general, when *d* = 0 or 1, the code runs fast. When *d* = 1.5 or 2, it takes longer. When *d* = 2.5, it takes even longer than when *d* = 2.

We provide a Python implementation of the dK-series.

If you use this code, please cite:

- [Takumi Sakiyama, Kazuki Nakajima, Masaki Aida. Efficient intervention in the spread of misinformation in social networks. *IEEE Access*. Vol. 12, pp. 133489–133498 (2024).](https://doi.org/10.1109/ACCESS.2024.3459830)

# What's new

- March 2026: We released a Python package `dk_series`. The package supports all *d* values (0, 1, 1.5, 2, 2.5) and provides exact simple graph sampling for *d* = 1 [[Del Genio et al. *PLOS ONE* (2010)]](https://doi.org/10.1371/journal.pone.0010012) and *d* = 2 [[Bassler et al. *New Journal of Physics* (2015)]](https://doi.org/10.1088/1367-2630/17/8/083052). The rewiring for *d* = 1.5 and 2.5 is accelerated with [Numba](https://numba.pydata.org/).

---

## Requirements

- Python 3.8+
- NumPy
- (Optional) NetworkX — for `to_networkx()`
- (Optional) igraph — for `to_igraph()`
- (Optional) Numba — for accelerated d=1.5 and d=2.5 rewiring

## Installation

Install in editable mode from the repository root:

```bash
pip install -e .
```

Or add the repository root to your Python path:

```python
import sys
sys.path.insert(0, "/path/to/dk-series")
import dk_series
```

## Quick start

```python
import dk_series

# 1. Load a network
edges, node_map = dk_series.read_network("data/soc-karate.txt")

# 2. Randomize (d=2: preserves joint degree distribution)
rand_edges = dk_series.randomize(edges, d=2, seed=42)

# 3. Compare statistics with the original
dk_series.compare(edges, rand_edges)
```

---

## API Reference

### `read_network(filepath)`

Read an edge-list file and return internal edge arrays.

```python
edges, node_map = dk_series.read_network("data/soc-karate.txt")
# edges   : ndarray (M, 2) — edge array using internal indices (0..N-1)
# node_map: ndarray (N,)   — node_map[i] = original node ID
```

Input file format (one edge per line as `u v`):

```
0 1
0 4
1 2
2 3
3 4
```

Lines starting with `#` and blank lines are ignored. Node IDs do not need to start at 0 or be consecutive.

---

### `write_network(filepath, edges, node_map)`

Write a randomized network to a file using the original node IDs.

```python
dk_series.write_network("outputs/rand_karate.txt", rand_edges, node_map)
```

---

### `randomize(edges, d, seed=None, num=1, rewiring_coeff=500, simple=True)`

Run dK-series randomization.

| Argument | Type | Description |
|---|---|---|
| `edges` | ndarray (M, 2) | Edge array returned by `read_network()` |
| `d` | float | Randomization parameter (0, 1, 1.5, 2, or 2.5) |
| `seed` | int or None | Random seed (recommended for reproducibility) |
| `num` | int | Number of networks to generate; returns an ndarray directly if 1, a list if >1 |
| `rewiring_coeff` | int | Rewiring attempt coefficient for d=1.5 and d=2.5 (attempts = rewiring_coeff × M) |
| `simple` | bool | If `True`, output is guaranteed to be a simple graph (no self-loops, no multi-edges). Applies to d=1 and d=2. If `False` (default), uses a faster configuration-model approach that may produce self-loops or multi-edges. |

Statistics preserved by each value of `d`:

| d | Preserved statistics | Simple graph (`simple=True`) |
|---|---|---|
| 0 | Edge count only (fully random) | Yes — Erdős-Rényi G(N, M) |
| 1 | Degree distribution P(k) | Yes |
| 1.5 | Degree distribution P(k) + clustering coefficient (approximate) | Yes |
| 2 | Degree distribution P(k) + joint degree distribution P(k,l) | Yes |
| 2.5 | Degree distribution P(k) + P(k,l) + clustering coefficient (approximate) | Yes |

For d=0, `simple=True` generates an Erdős-Rényi G(N, M) graph via rejection sampling.
For d=1, the exact sampling algorithm of Del Genio et al. (2010) [3] is used.
For d=2, the joint degree matrix is preserved exactly using the algorithm of Bassler et al. (2015) [4].
For d=1.5 and d=2.5, the rewiring step skips any candidate swap that would create a duplicate edge.

```python
# Default: faster, may contain self-loops or multi-edges
rand_edges = dk_series.randomize(edges, d=2, seed=42)

# Simple graph: no self-loops, no multi-edges
rand_edges = dk_series.randomize(edges, d=2, seed=42, simple=True)

# Generate multiple networks
rand_list = dk_series.randomize(edges, d=2, seed=0, num=10)
```

---

### `compare(orig_edges, rand_edges, verbose=True)`

Compare statistics between the original and a randomized network.

```python
result = dk_series.compare(edges, rand_edges)
# result['degree_dist_l1'] : L1 distance of degree distributions
# result['jdm_l1']         : normalized L1 distance of joint degree distributions
# result['ddcc_l1']        : normalized L1 distance of degree-dependent clustering coefficients
```

---

### `compare_multiple(orig_edges, rand_edges_list, verbose=True)`

Compare statistics across multiple randomized networks and report mean ± std.

```python
rand_list = dk_series.randomize(edges, d=2, seed=0, num=10)
summary, results = dk_series.compare_multiple(edges, rand_list)
# summary['degree_dist_l1'] = {'mean': ..., 'std': ...}
```

---

### `to_networkx(edges, node_map=None)`

Convert an edge array to a NetworkX graph.

```python
import networkx as nx
G = dk_series.to_networkx(rand_edges, node_map)
print(nx.average_clustering(G))
```

---

### `to_igraph(edges, N, node_map=None)`

Convert an edge array to an igraph graph.

```python
N = len(node_map)
g = dk_series.to_igraph(rand_edges, N=N, node_map=node_map)
print(g.transitivity_undirected())
```

---

## Tutorial notebook

See [`tutorial.ipynb`](tutorial.ipynb) for a step-by-step walkthrough.

---

## References

[1] Orsini, C., Dankulov, M., Colomer-de-Simón, P. et al. Quantifying randomness in real networks. Nat. Commun. 6, 8627 (2015). [<a href="https://doi.org/10.1038/ncomms9627">paper</a>]

[2] Mahadevan, P., Krioukov, D., Fall, K., & Vahdat, A. Systematic topology analysis and generation using degree correlations. SIGCOMM Comput. Commun. Rev., 36(4), 135–146 (2006). [<a href="https://doi.org/10.1145/1151659.1159930">paper</a>]

[3] Del Genio, C. I., Kim, H., Toroczkai, Z., & Bassler, K. E. Efficient and exact sampling of simple graphs with given arbitrary degree sequence. PLOS ONE, 5(4), e10012 (2010). [<a href="https://doi.org/10.1371/journal.pone.0010012">paper</a>]

[4] Bassler, K. E., Del Genio, C. I., Erdős, P. L., Miklós, I., & Toroczkai, Z. Exact sampling of graphs with prescribed degree correlations. New Journal of Physics, 17(8), 083052 (2015). [<a href="https://doi.org/10.1088/1367-2630/17/8/083052">paper</a>]

## License

This source code is released under the MIT License, see LICENSE.txt.

## Contact

- Kazuki Nakajima (https://kazuibasou.github.io/index_en.html)
- kazuibasou[at]gmail.com
