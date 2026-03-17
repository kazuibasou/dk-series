"""
dk_series: dK-series ネットワークランダム化の Python 実装

使用例
------
>>> import dk_series
>>> edges, node_map = dk_series.read_network("data/example-network.txt")
>>> rand_edges = dk_series.randomize(edges, d=2, seed=42)
>>> G = dk_series.to_networkx(rand_edges, node_map)
>>> g = dk_series.to_igraph(rand_edges, N=len(node_map), node_map=node_map)
"""
import numpy as np

from ._io import read_network, write_network
from ._convert import to_networkx, to_igraph
from ._core import randomize_d0, randomize_d1, randomize_d1_exact, randomize_d15, randomize_d2, randomize_d2_exact, randomize_d25, _compute_degrees
from ._validate import compare, compare_multiple


def randomize(edges, d=2, seed=None, num=1, rewiring_coeff=500, simple=False):
    """
    Run dK-series network randomization.

    Parameters
    ----------
    edges : ndarray, shape (M, 2)
        Edge array returned by read_network() (internal indices 0..N-1).
    d : float
        Randomization parameter. One of 0, 1, 1.5, 2, 2.5.
    seed : int or None
        Random seed for reproducibility.
    num : int
        Number of randomized networks to generate.
        Returns an ndarray directly if 1, a list of ndarray if >1.
    rewiring_coeff : int
        Rewiring attempt coefficient for d=1.5 and d=2.5.
        Number of attempts = rewiring_coeff * M. Default: 500.
    simple : bool
        If True (default), the output is guaranteed to be a simple graph
        (no self-loops, no multi-edges). Applies to d=1 and d=2.
        If False, uses a faster configuration-model approach that may
        produce self-loops or multi-edges.
        Has no effect for d=0, 1.5, and 2.5.

    Returns
    -------
    rand_edges : ndarray (M, 2) or list of ndarray
        Randomized edge array(s).
    """
    rng = np.random.default_rng(seed)
    N = int(edges.max()) + 1
    degrees = _compute_degrees(N, edges)

    results = []
    for i in range(num):
        print(f"--- Randomization {i+1}/{num} (d={d}) ---")
        if d == 0:
            result = randomize_d0(N, len(edges), rng, simple)
        elif d == 1:
            if simple:
                result, _ = randomize_d1_exact(N, edges, degrees, rng)
            else:
                result = randomize_d1(N, edges, degrees, rng)
        elif d == 1.5:
            result = randomize_d15(N, edges, degrees, rng, rewiring_coeff, simple)
        elif d == 2:
            if simple:
                result, _ = randomize_d2_exact(N, edges, degrees, rng)
            else:
                result = randomize_d2(N, edges, degrees, rng)
        elif d == 2.5:
            result = randomize_d25(N, edges, degrees, rng, rewiring_coeff, simple)
        else:
            raise ValueError(f"d must be 0, 1, 1.5, 2, or 2.5. Got: {d}")
        results.append(result)

    return results[0] if num == 1 else results


__all__ = [
    "read_network",
    "write_network",
    "randomize",
    "randomize_d1_exact",
    "randomize_d2_exact",
    "to_networkx",
    "to_igraph",
    "compare",
    "compare_multiple",
]
