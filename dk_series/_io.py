"""ファイルI/O: エッジリスト形式のネットワークファイルを読み書きする"""
import numpy as np


def read_network(filepath):
    """
    エッジリストファイルを読み込む。

    Parameters
    ----------
    filepath : str
        各行に "u v" 形式のエッジリストファイルパス

    Returns
    -------
    edges : ndarray, shape (M, 2), dtype int64
        内部インデックス (0..N-1) でのエッジ配列
    node_map : ndarray, shape (N,), dtype int64
        node_map[i] = 元のノードID
    """
    edges_raw = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            u, v = int(parts[0]), int(parts[1])
            edges_raw.append((u, v))

    # ノードIDを 0..N-1 に再マッピング
    nodes = sorted(set(n for e in edges_raw for n in e))
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    node_map = np.array(nodes, dtype=np.int64)
    edges = np.array(
        [(node_to_idx[u], node_to_idx[v]) for u, v in edges_raw],
        dtype=np.int64
    )

    N = len(nodes)
    M = len(edges)
    print(f"Network loaded: {N} nodes, {M} edges")
    return edges, node_map


def write_network(filepath, edges, node_map):
    """
    エッジリストをファイルに書き出す（元のノードIDで出力）。

    Parameters
    ----------
    filepath : str
        出力ファイルパス
    edges : ndarray, shape (M, 2)
        内部インデックスのエッジ配列
    node_map : ndarray, shape (N,)
        node_map[i] = 元のノードID
    """
    with open(filepath, 'w') as f:
        for u, v in edges:
            orig_u = int(node_map[u])
            orig_v = int(node_map[v])
            f.write(f"{orig_u} {orig_v}\n")
