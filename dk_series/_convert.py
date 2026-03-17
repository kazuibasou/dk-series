"""NetworkX・igraph への変換ユーティリティ"""


def to_networkx(edges, node_map=None):
    """
    エッジ配列から NetworkX グラフを生成する。

    Parameters
    ----------
    edges : ndarray, shape (M, 2)
        エッジ配列
    node_map : ndarray or None
        node_map[i] = 元のノードID。指定した場合は元のIDでノードを作成する。

    Returns
    -------
    G : networkx.Graph
    """
    import networkx as nx
    G = nx.Graph()
    if node_map is not None:
        edge_list = [(int(node_map[u]), int(node_map[v])) for u, v in edges]
    else:
        edge_list = [(int(u), int(v)) for u, v in edges]
    G.add_edges_from(edge_list)
    return G


def to_igraph(edges, N, node_map=None):
    """
    エッジ配列から igraph グラフを生成する。

    Parameters
    ----------
    edges : ndarray, shape (M, 2)
        エッジ配列（内部インデックス 0..N-1）
    N : int
        ノード数
    node_map : ndarray or None
        node_map[i] = 元のノードID。指定した場合は vs['name'] 属性に格納。

    Returns
    -------
    g : igraph.Graph
    """
    import igraph as ig
    g = ig.Graph(n=N, edges=[(int(u), int(v)) for u, v in edges])
    if node_map is not None:
        g.vs['name'] = [int(node_map[i]) for i in range(N)]
    return g
