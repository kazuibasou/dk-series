"""
入力グラフと生成グラフの統計量を比較・検証するユーティリティ。

計算する距離:
  - 次数分布     P(k)   : L1 距離
  - 同時次数分布  P(k,l) : 正規化 L1 距離
  - DDCC         c(k)   : 正規化 L1 距離（C++ と同じ定義）
"""
import numpy as np
from ._core import _compute_degrees, _compute_ddcc


# -----------------------------------------------------------------------
# 統計量の計算
# -----------------------------------------------------------------------

def compute_degree_dist(N, degrees):
    """
    次数分布 P(k) を計算する。

    Returns
    -------
    ks : ndarray  次数の値
    pk : ndarray  P(k) = N_k / N
    """
    max_k = int(degrees.max())
    N_k = np.zeros(max_k + 1, dtype=np.int64)
    for k in degrees:
        N_k[int(k)] += 1
    ks = np.arange(max_k + 1)
    pk = N_k / N
    return ks, pk


def compute_jdm_normalized(edges, degrees):
    """
    正規化された同時次数分布 P(k,l) を計算する。

    Returns
    -------
    jdm_norm : dict[(k,l)] = P(k,l)   （有向、両方向カウント済み）
    total    : int  有向エッジ総数（= 2M）
    """
    jdm = {}
    for u, v in edges:
        k, l = int(degrees[u]), int(degrees[v])
        jdm[(k, l)] = jdm.get((k, l), 0) + 1
        jdm[(l, k)] = jdm.get((l, k), 0) + 1

    total = sum(jdm.values())
    jdm_norm = {kl: cnt / total for kl, cnt in jdm.items()} if total > 0 else jdm
    return jdm_norm, total


# -----------------------------------------------------------------------
# 距離の計算
# -----------------------------------------------------------------------

def _l1_degree_dist(pk_orig, pk_rand):
    """次数分布の L1 距離"""
    max_k = max(len(pk_orig), len(pk_rand))
    a = np.zeros(max_k)
    b = np.zeros(max_k)
    a[:len(pk_orig)] = pk_orig
    b[:len(pk_rand)] = pk_rand
    return float(np.sum(np.abs(a - b)))


def _l1_jdm(jdm_orig, jdm_rand):
    """同時次数分布の正規化 L1 距離"""
    all_keys = set(jdm_orig.keys()) | set(jdm_rand.keys())
    dist = 0.0
    for key in all_keys:
        dist += abs(jdm_orig.get(key, 0.0) - jdm_rand.get(key, 0.0))
    return float(dist)


def _l1_ddcc_normalized(ddcc_orig, ddcc_rand):
    """DDCC の正規化 L1 距離（C++ と同じ定義: Σ|diff| / Σ target）"""
    max_k = max(len(ddcc_orig), len(ddcc_rand))
    a = np.zeros(max_k)
    b = np.zeros(max_k)
    a[:len(ddcc_orig)] = ddcc_orig
    b[:len(ddcc_rand)] = ddcc_rand
    norm = float(np.sum(a))
    if norm == 0:
        return float('nan')
    return float(np.sum(np.abs(a - b))) / norm


# -----------------------------------------------------------------------
# 公開関数
# -----------------------------------------------------------------------

def compare(orig_edges, rand_edges, verbose=True):
    """
    元ネットワークとランダム化ネットワークの統計量を比較する。

    Parameters
    ----------
    orig_edges : ndarray (M, 2)
        元ネットワークのエッジ配列
    rand_edges : ndarray (M, 2)
        ランダム化後のエッジ配列
    verbose : bool
        True のとき結果を標準出力に表示する

    Returns
    -------
    result : dict
        'degree_dist_l1'  : 次数分布の L1 距離
        'jdm_l1'          : 同時次数分布の正規化 L1 距離
        'ddcc_l1'         : DDCC の正規化 L1 距離
    """
    N_orig = int(orig_edges.max()) + 1
    N_rand = int(rand_edges.max()) + 1
    N = max(N_orig, N_rand)

    deg_orig = _compute_degrees(N, orig_edges)
    deg_rand = _compute_degrees(N, rand_edges)
    max_k = int(max(deg_orig.max(), deg_rand.max()))

    # --- 次数分布 ---
    _, pk_orig = compute_degree_dist(N, deg_orig)
    _, pk_rand = compute_degree_dist(N, deg_rand)
    d_pk = _l1_degree_dist(pk_orig, pk_rand)

    # --- 同時次数分布 ---
    jdm_orig, _ = compute_jdm_normalized(orig_edges, deg_orig)
    jdm_rand, _ = compute_jdm_normalized(rand_edges, deg_rand)
    d_jdm = _l1_jdm(jdm_orig, jdm_rand)

    # --- DDCC ---
    ddcc_orig, N_k = _compute_ddcc(N, orig_edges, deg_orig, max_k)
    ddcc_rand, _   = _compute_ddcc(N, rand_edges, deg_rand, max_k)
    d_ddcc = _l1_ddcc_normalized(ddcc_orig, ddcc_rand)

    result = {
        'degree_dist_l1': d_pk,
        'jdm_l1':         d_jdm,
        'ddcc_l1':        d_ddcc,
    }

    if verbose:
        print("=" * 50)
        print("  Comparison results")
        print("=" * 50)
        print(f"  Degree dist.  P(k)   L1 : {d_pk:.6f}")
        print(f"  Joint deg.    P(k,l) L1 : {d_jdm:.6f}")
        print(f"  DDCC          c(k)   L1 : {d_ddcc:.6f}  (normalized)")
        print("=" * 50)

    return result


def compare_multiple(orig_edges, rand_edges_list, verbose=True):
    """
    複数のランダム化ネットワークとの平均距離を計算する。

    Parameters
    ----------
    orig_edges     : ndarray (M, 2)
    rand_edges_list : list of ndarray (M, 2)
    verbose        : bool

    Returns
    -------
    summary : dict  各指標の mean / std
    results : list of dict  個別の結果
    """
    results = [compare(orig_edges, e, verbose=False) for e in rand_edges_list]
    keys = results[0].keys()
    summary = {}
    for k in keys:
        vals = [r[k] for r in results]
        summary[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}

    if verbose:
        n = len(rand_edges_list)
        print("=" * 55)
        print(f"  Comparison results (mean ± std over {n} samples)")
        print("=" * 55)
        print(f"  Degree dist.  P(k)   L1 : "
              f"{summary['degree_dist_l1']['mean']:.6f} "
              f"± {summary['degree_dist_l1']['std']:.6f}")
        print(f"  Joint deg.    P(k,l) L1 : "
              f"{summary['jdm_l1']['mean']:.6f} "
              f"± {summary['jdm_l1']['std']:.6f}")
        print(f"  DDCC          c(k)   L1 : "
              f"{summary['ddcc_l1']['mean']:.6f} "
              f"± {summary['ddcc_l1']['std']:.6f}  (normalized)")
        print("=" * 55)

    return summary, results
