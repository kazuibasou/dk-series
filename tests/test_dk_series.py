"""
Comprehensive tests for the dk_series package.

Test matrix:
  - d = 0, 1, 1.5, 2, 2.5
  - simple = True / False
  - Networks: soc-karate (small), soc-dolphins (medium),
              example-network (large), ca-netscience (large), ca-csphd (large)

Checks:
  - Edge count preserved
  - Simple graph properties (simple=True: no self-loops/multi-edges)
  - Degree distribution preserved (d >= 1)
  - Joint degree distribution preserved (d = 2, simple=True)
  - Reproducibility (same seed -> same result)
  - num > 1 returns list of correct length
  - write_network / read_network roundtrip
  - compare() and compare_multiple() return correct structure
  - to_networkx() and to_igraph() conversions
"""

import os
import tempfile

import numpy as np
import pytest

import dk_series


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def is_simple(edges):
    """Return (n_self_loops, n_multi_edges)."""
    self_loops = sum(1 for u, v in edges if u == v)
    seen = set()
    multi = 0
    for u, v in edges:
        key = (min(u, v), max(u, v))
        if key in seen:
            multi += 1
        seen.add(key)
    return self_loops, multi


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

@pytest.fixture(scope="module")
def karate():
    edges, node_map = dk_series.read_network(os.path.join(DATA_DIR, "soc-karate.txt"))
    return edges, node_map

@pytest.fixture(scope="module")
def dolphins():
    edges, node_map = dk_series.read_network(os.path.join(DATA_DIR, "soc-dolphins.txt"))
    return edges, node_map

@pytest.fixture(scope="module")
def example_net():
    edges, node_map = dk_series.read_network(os.path.join(DATA_DIR, "example-network.txt"))
    return edges, node_map

@pytest.fixture(scope="module")
def netscience():
    edges, node_map = dk_series.read_network(os.path.join(DATA_DIR, "ca-netscience.txt"))
    return edges, node_map

@pytest.fixture(scope="module")
def csphd():
    edges, node_map = dk_series.read_network(os.path.join(DATA_DIR, "ca-csphd.txt"))
    return edges, node_map


# -----------------------------------------------------------------------
# I/O tests
# -----------------------------------------------------------------------

class TestIO:
    def test_read_returns_correct_types(self, karate):
        edges, node_map = karate
        assert isinstance(edges, np.ndarray)
        assert isinstance(node_map, np.ndarray)
        assert edges.ndim == 2 and edges.shape[1] == 2
        assert edges.dtype == np.int64

    @pytest.mark.parametrize("network_name,expected_nodes,expected_edges", [
        ("karate",      34,   78),
        ("dolphins",    62,  159),
        ("example_net", None, 1996),
        ("netscience",  None,  914),
        ("csphd",       None, 1740),
    ])
    def test_read_edge_count(self, request, network_name, expected_nodes, expected_edges):
        edges, node_map = request.getfixturevalue(network_name)
        assert len(edges) == expected_edges
        if expected_nodes is not None:
            assert len(node_map) == expected_nodes

    @pytest.mark.parametrize("network_name", [
        "karate", "dolphins", "example_net", "netscience", "csphd"
    ])
    def test_original_is_simple(self, request, network_name):
        edges, _ = request.getfixturevalue(network_name)
        sl, me = is_simple(edges)
        assert sl == 0 and me == 0, \
            f"{network_name}: {sl} self-loops, {me} multi-edges in original"

    def test_write_read_roundtrip(self, karate):
        edges, node_map = karate
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            path = f.name
        try:
            dk_series.write_network(path, edges, node_map)
            edges2, node_map2 = dk_series.read_network(path)
            assert len(edges2) == len(edges)
            assert set(map(tuple, edges2.tolist())) == set(map(tuple, edges.tolist()))
            assert np.array_equal(np.sort(node_map), np.sort(node_map2))
        finally:
            os.unlink(path)


# -----------------------------------------------------------------------
# Randomization: edge count (all networks x all d x simple=True/False)
# -----------------------------------------------------------------------

@pytest.mark.parametrize("network_name", [
    "karate", "dolphins", "example_net", "netscience", "csphd"
])
@pytest.mark.parametrize("d", [0, 1, 1.5, 2, 2.5])
@pytest.mark.parametrize("simple", [False, True])
class TestEdgeCount:
    def test_edge_count_preserved(self, request, network_name, d, simple):
        edges, _ = request.getfixturevalue(network_name)
        rand = dk_series.randomize(edges, d=d, seed=0, simple=simple)
        assert len(rand) == len(edges)


# -----------------------------------------------------------------------
# Randomization: simple graph guarantee (simple=True, all networks)
# -----------------------------------------------------------------------

@pytest.mark.parametrize("network_name", [
    "karate", "dolphins", "example_net", "netscience", "csphd"
])
@pytest.mark.parametrize("d", [1, 1.5, 2, 2.5])
class TestSimpleTrue:
    def test_no_self_loops(self, request, network_name, d):
        edges, _ = request.getfixturevalue(network_name)
        rand = dk_series.randomize(edges, d=d, seed=0, simple=True)
        sl, _ = is_simple(rand)
        assert sl == 0, f"{network_name} d={d}: {sl} self-loops found"

    def test_no_multi_edges(self, request, network_name, d):
        edges, _ = request.getfixturevalue(network_name)
        rand = dk_series.randomize(edges, d=d, seed=0, simple=True)
        _, me = is_simple(rand)
        assert me == 0, f"{network_name} d={d}: {me} multi-edges found"


# -----------------------------------------------------------------------
# Randomization: degree distribution preserved (d >= 1, simple=True)
# -----------------------------------------------------------------------

@pytest.mark.parametrize("network_name", [
    "karate", "dolphins", "example_net", "netscience", "csphd"
])
@pytest.mark.parametrize("d", [1, 1.5, 2, 2.5])
class TestDegreeDistPreserved:
    def test_degree_dist_l1_zero(self, request, network_name, d):
        edges, _ = request.getfixturevalue(network_name)
        rand = dk_series.randomize(edges, d=d, seed=0, simple=True)
        result = dk_series.compare(edges, rand, verbose=False)
        assert result["degree_dist_l1"] == pytest.approx(0.0, abs=1e-9), \
            f"{network_name} d={d}: degree dist L1 = {result['degree_dist_l1']}"


# -----------------------------------------------------------------------
# Randomization: JDM preserved (d=2, simple=True, all networks)
# -----------------------------------------------------------------------

@pytest.mark.parametrize("network_name", [
    "karate", "dolphins", "example_net", "netscience", "csphd"
])
class TestJDMPreserved:
    def test_jdm_l1_zero_d2(self, request, network_name):
        edges, _ = request.getfixturevalue(network_name)
        rand = dk_series.randomize(edges, d=2, seed=0, simple=True)
        result = dk_series.compare(edges, rand, verbose=False)
        assert result["jdm_l1"] == pytest.approx(0.0, abs=1e-9), \
            f"{network_name} d=2: JDM L1 = {result['jdm_l1']}"


# -----------------------------------------------------------------------
# Reproducibility (karate のみ: seed の挙動確認)
# -----------------------------------------------------------------------

@pytest.mark.parametrize("d", [0, 1, 1.5, 2, 2.5])
@pytest.mark.parametrize("simple", [False, True])
class TestReproducibility:
    def test_same_seed_same_result(self, karate, d, simple):
        edges, _ = karate
        r1 = dk_series.randomize(edges, d=d, seed=42, simple=simple)
        r2 = dk_series.randomize(edges, d=d, seed=42, simple=simple)
        assert np.array_equal(r1, r2), f"d={d} simple={simple}: different results with same seed"

    def test_different_seed_different_result(self, karate, d, simple):
        edges, _ = karate
        r1 = dk_series.randomize(edges, d=d, seed=0, simple=simple)
        r2 = dk_series.randomize(edges, d=d, seed=1, simple=simple)
        assert not np.array_equal(r1, r2), f"d={d} simple={simple}: same result with different seeds"


# -----------------------------------------------------------------------
# num > 1
# -----------------------------------------------------------------------

class TestNumArg:
    @pytest.mark.parametrize("d", [0, 1, 2])
    def test_num_returns_list(self, karate, d):
        edges, _ = karate
        result = dk_series.randomize(edges, d=d, seed=0, num=5)
        assert isinstance(result, list)
        assert len(result) == 5

    @pytest.mark.parametrize("d", [0, 1, 2])
    def test_num_1_returns_array(self, karate, d):
        edges, _ = karate
        result = dk_series.randomize(edges, d=d, seed=0, num=1)
        assert isinstance(result, np.ndarray)

    def test_num_list_each_correct_edge_count(self, karate):
        edges, _ = karate
        rand_list = dk_series.randomize(edges, d=2, seed=0, num=5, simple=True)
        for r in rand_list:
            assert len(r) == len(edges)


# -----------------------------------------------------------------------
# compare() and compare_multiple()
# -----------------------------------------------------------------------

class TestCompare:
    def test_compare_returns_expected_keys(self, karate):
        edges, _ = karate
        rand = dk_series.randomize(edges, d=2, seed=0, simple=True)
        result = dk_series.compare(edges, rand, verbose=False)
        assert "degree_dist_l1" in result
        assert "jdm_l1" in result
        assert "ddcc_l1" in result

    def test_compare_values_nonnegative(self, karate):
        edges, _ = karate
        rand = dk_series.randomize(edges, d=2, seed=0, simple=True)
        result = dk_series.compare(edges, rand, verbose=False)
        for v in result.values():
            assert v >= 0

    def test_compare_self_is_zero(self, karate):
        edges, _ = karate
        result = dk_series.compare(edges, edges, verbose=False)
        assert result["degree_dist_l1"] == pytest.approx(0.0, abs=1e-9)
        assert result["jdm_l1"] == pytest.approx(0.0, abs=1e-9)
        assert result["ddcc_l1"] == pytest.approx(0.0, abs=1e-9)

    def test_compare_multiple_returns_summary_and_list(self, karate):
        edges, _ = karate
        rand_list = dk_series.randomize(edges, d=2, seed=0, num=5, simple=True)
        summary, results = dk_series.compare_multiple(edges, rand_list, verbose=False)
        assert len(results) == 5
        assert "degree_dist_l1" in summary
        assert "mean" in summary["degree_dist_l1"]
        assert "std" in summary["degree_dist_l1"]

    def test_compare_multiple_mean_zero_d2(self, karate):
        edges, _ = karate
        rand_list = dk_series.randomize(edges, d=2, seed=0, num=5, simple=True)
        summary, _ = dk_series.compare_multiple(edges, rand_list, verbose=False)
        assert summary["degree_dist_l1"]["mean"] == pytest.approx(0.0, abs=1e-9)
        assert summary["jdm_l1"]["mean"] == pytest.approx(0.0, abs=1e-9)


# -----------------------------------------------------------------------
# Conversions
# -----------------------------------------------------------------------

class TestConversions:
    def test_to_networkx_node_count(self, karate):
        edges, node_map = karate
        rand = dk_series.randomize(edges, d=2, seed=0, simple=True)
        G = dk_series.to_networkx(rand, node_map)
        import networkx as nx
        assert G.number_of_nodes() == len(node_map)

    def test_to_networkx_edge_count(self, karate):
        edges, node_map = karate
        rand = dk_series.randomize(edges, d=2, seed=0, simple=True)
        G = dk_series.to_networkx(rand, node_map)
        import networkx as nx
        assert G.number_of_edges() == len(rand)

    def test_to_networkx_uses_original_ids(self, karate):
        edges, node_map = karate
        rand = dk_series.randomize(edges, d=2, seed=0, simple=True)
        G = dk_series.to_networkx(rand, node_map)
        import networkx as nx
        assert set(G.nodes()) == set(node_map.tolist())

    def test_to_igraph_node_count(self, karate):
        edges, node_map = karate
        N = len(node_map)
        rand = dk_series.randomize(edges, d=2, seed=0, simple=True)
        g = dk_series.to_igraph(rand, N=N, node_map=node_map)
        assert g.vcount() == N

    def test_to_igraph_edge_count(self, karate):
        edges, node_map = karate
        N = len(node_map)
        rand = dk_series.randomize(edges, d=2, seed=0, simple=True)
        g = dk_series.to_igraph(rand, N=N, node_map=node_map)
        assert g.ecount() == len(rand)
