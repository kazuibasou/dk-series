"""
dK-series ランダム化アルゴリズム（Python + NumPy 実装）。
d=2.5 の内部ループのみ Numba を使用。
"""
import numpy as np
from ._numba_core import rewiring_loop_d25, rewiring_loop_d15


# -----------------------------------------------------------------------
# 共通ユーティリティ
# -----------------------------------------------------------------------

def _compute_degrees(N, edges):
    """各ノードの次数を計算する"""
    degrees = np.zeros(N, dtype=np.int64)
    for u, v in edges:
        degrees[u] += 1
        degrees[v] += 1
    return degrees


def _compute_jdm(N, edges, degrees):
    """
    結合次数行列 (JDM) を計算する。
    戻り値: dict[(k,l)] = エッジ数（有向、C++ と同様に両方向カウント）
    """
    jdm = {}
    for u, v in edges:
        k = int(degrees[u])
        l = int(degrees[v])
        jdm[(k, l)] = jdm.get((k, l), 0) + 1
        jdm[(l, k)] = jdm.get((l, k), 0) + 1
    return jdm


def _compute_ddcc(N, edges, degrees, max_k):
    """
    度数依存クラスタリング係数 (DDCC) を計算する。
    C++ の calc_degree_dependent_clustering_coefficient() と同等。

    Returns
    -------
    ddcc : ndarray (max_k+1,)
    N_k  : ndarray (max_k+1,)   各次数のノード数
    """
    k_size = max_k + 1
    ddcc = np.zeros(k_size, dtype=np.float64)
    N_k = np.zeros(k_size, dtype=np.int64)

    # 隣接リスト（set）を構築してクラスタリング係数を計算
    nlist = [[] for _ in range(N)]
    for u, v in edges:
        nlist[u].append(int(v))
        nlist[v].append(int(u))

    for v in range(N):
        k = int(degrees[v])
        N_k[k] += 1
        if k < 2:
            continue
        neighbors = nlist[v]
        nset = set(neighbors)
        cc = 0.0
        for i in range(len(neighbors) - 1):
            u = neighbors[i]
            for j in range(i + 1, len(neighbors)):
                w = neighbors[j]
                if v != u and v != w and u != w:
                    # C++ は std::count（多重辺も数える）
                    cnt = neighbors[i:i+1].count(w)  # 1 or 0 for simple
                    cc += 2 * nlist[u].count(w)
        cc /= k * (k - 1)
        ddcc[k] += cc

    for k in range(2, k_size):
        if N_k[k] > 0:
            ddcc[k] /= N_k[k]

    return ddcc, N_k


# -----------------------------------------------------------------------
# d = 0
# -----------------------------------------------------------------------

def randomize_d0(N, M, rng, simple=False):
    """
    完全ランダム: ノード数とエッジ数のみ保存。

    simple=False: 自己ループ・多重辺を許す（高速）
    simple=True : Erdős-Rényi G(N, M) モデル（rejection sampling）

    Returns
    -------
    edges : ndarray (M, 2)
    """
    if not simple:
        u = rng.integers(0, N, size=M)
        v = rng.integers(0, N, size=M)
        return np.stack([u, v], axis=1).astype(np.int64)

    # Erdős-Rényi G(N, M): rejection sampling で simple graph を生成
    edge_set = set()
    result = []
    while len(result) < M:
        u = int(rng.integers(0, N))
        v = int(rng.integers(0, N))
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key in edge_set:
            continue
        edge_set.add(key)
        result.append((u, v))
    return np.array(result, dtype=np.int64)


# -----------------------------------------------------------------------
# d = 1  Configuration Model（参考実装）
# -----------------------------------------------------------------------

def randomize_d1(N, edges, degrees, rng):
    """
    Configuration Model: 次数列を保存。
    スタブをランダムに結合するため、自己ループや多重辺が生じる可能性がある。
    simple graph が必要な場合は randomize_d1_exact() を使用すること。

    Returns
    -------
    edges : ndarray (M, 2)
    """
    stub_list = np.repeat(np.arange(N, dtype=np.int64), degrees)
    rng.shuffle(stub_list)
    M = len(stub_list) // 2
    return stub_list.reshape(-1, 2)[:M]


# -----------------------------------------------------------------------
# d = 1  Del Genio アルゴリズム（simple graph 保証）
# -----------------------------------------------------------------------

def _is_graphical(res_deg):
    """
    Erdős-Gallai 定理により次数列がグラフィカルかどうかを判定する。
    res_deg: numpy 配列（0 を含んでよい。正の値のみ検査に使用）

    完全ベクトル化実装: O(N log N)
    """
    d = np.sort(res_deg[res_deg > 0])[::-1].astype(np.int64)
    n = len(d)
    if n == 0:
        return True
    if d.sum() % 2 != 0:
        return False

    # 累積和と接尾辞和を事前計算
    cumsum = np.cumsum(d)
    suffix_sum = np.empty(n + 1, dtype=np.int64)
    suffix_sum[n] = 0
    suffix_sum[:-1] = np.cumsum(d[::-1])[::-1]

    # 各 k に対して min(k+1, d[i]) の境界インデックス j を一括計算
    # -d は非減少列。searchsorted で最初に -d[j] >= -(k+1)、
    # すなわち d[j] <= k+1 となる j を求める
    k = np.arange(n, dtype=np.int64)
    j_full = np.searchsorted(-d, -(k + 1), side='left')
    j = np.maximum(j_full, k + 1)

    # EG 条件: cumsum[k] <= k*(k+1) + (j-k-1)*(k+1) + suffix_sum[j]
    rhs = k * (k + 1) + (j - k - 1) * (k + 1) + suffix_sum[j]
    return bool(np.all(cumsum <= rhs))


def randomize_d1_exact(N, edges, degrees, rng):
    """
    Del Genio et al. (2010) アルゴリズムによる d=1 ランダム化。
    与えられた次数列を持つ simple graph（自己ループ・多重辺なし）を
    rejection-free で生成する。

    参考文献:
        C.I. Del Genio, H. Kim, Z. Toroczkai, K.E. Bassler,
        "Efficient and exact sampling of simple graphs with given
        arbitrary degree sequence", PLoS ONE 5(4): e10012 (2010)

    アルゴリズム概要:
        1. 最大残余次数のノードをハブとして選択
        2. ハブに接続可能なノード集合 A（allowed set）を Erdős-Gallai
           定理で構築（第 1 ステップは Corollary 1 により全候補が許可）
        3. A からランダムに選んで接続し、ハブのスタブが尽きるまで繰り返す
        4. 全ノードの次数が満たされるまで繰り返す

    Returns
    -------
    edges_out : ndarray (M, 2)
    log_weight : float
        サンプル重みの対数。不偏推定量の計算に使用する。
        log w = Σ_i [ Σ_j log|A_{i,j}| - log(d̄_i!) ]
        大規模ネットワークでは w 自体が非常に大きくなるため、
        対数のまま使用することを推奨する。
    """
    res_deg = degrees.copy().astype(np.int64)
    result_edges = []
    log_weight = 0.0

    active = set(v for v in range(N) if res_deg[v] > 0)

    while active:
        # 最大残余次数のノードをハブとして選択
        hub = max(active, key=lambda v: res_deg[v])
        d_hub_initial = int(res_deg[hub])

        # ハブのセッション全体で一度だけソート（ステップごとに再ソートしない）
        cand_arr = np.array([v for v in active if v != hub], dtype=np.int64)
        order = np.argsort(-res_deg[cand_arr])
        candidates = list(cand_arr[order])  # 次数降順、ステップごとに del で更新

        first_step = True

        while res_deg[hub] > 0:
            n_cand = len(candidates)

            if first_step:
                # 最初のステップ: Corollary 1 により全候補が allowed
                lo = n_cand
                first_step = False
            else:
                d_rem = int(res_deg[hub])
                if n_cand <= d_rem:
                    # 全候補が leftmost adjacency set に収まる（Theorem 6）
                    lo = n_cand
                else:
                    # 「最後の候補を先に確認」最適化:
                    # 最低次数候補が allowed なら Theorem FMT により全候補が allowed
                    # → ほとんどのケースで二分探索を省略できる
                    v_last = candidates[-1]
                    res_deg[hub] -= 1
                    res_deg[v_last] -= 1
                    ok_last = _is_graphical(res_deg)
                    res_deg[hub] += 1
                    res_deg[v_last] += 1

                    if ok_last:
                        lo = n_cand
                    else:
                        # 二分探索で境界を O(log N) で発見
                        # 最後は fail と確定済みなので hi = n_cand - 1
                        lo, hi = d_rem, n_cand - 1
                        while lo < hi:
                            mid = (lo + hi) // 2
                            v = candidates[mid]
                            res_deg[hub] -= 1
                            res_deg[v] -= 1
                            ok = _is_graphical(res_deg)
                            res_deg[hub] += 1
                            res_deg[v] += 1
                            if ok:
                                lo = mid + 1
                            else:
                                hi = mid

            # 重み更新: k_{i,j} = lo（allowed set のサイズ）
            log_weight += np.log(lo)

            # candidates[:lo] からランダムに選択し、リストから削除
            a_idx = int(rng.integers(0, lo))
            a = candidates[a_idx]
            del candidates[a_idx]  # ソート順を維持したまま削除（多重辺防止）

            # エッジ追加・残余次数更新
            result_edges.append([min(hub, a), max(hub, a)])
            res_deg[hub] -= 1
            res_deg[a] -= 1

            if res_deg[a] == 0:
                active.discard(a)

        active.discard(hub)
        # 重み更新の分母: d̄_hub!（NumPy で高速計算）
        if d_hub_initial > 1:
            log_weight -= np.sum(np.log(np.arange(1, d_hub_initial + 1)))

    edges_out = (
        np.array(result_edges, dtype=np.int64)
        if result_edges
        else np.empty((0, 2), dtype=np.int64)
    )
    return edges_out, log_weight


# -----------------------------------------------------------------------
# d = 2  Bassler et al. (2015) アルゴリズム（simple graph 保証）
# -----------------------------------------------------------------------

def _is_bigraphical(s, t):
    """
    Gale-Ryser 定理により二部グラフの次数列 (s, t) がグラフィカルか判定する。
    s: 左側ノードの次数列（numpy 配列、0含む可能性あり）
    t: 右側ノードの次数列（numpy 配列、0含む可能性あり）

    条件: s の各要素は t の上位 s_i 個の和以下
    """
    s = np.sort(s[s > 0])[::-1].astype(np.int64)
    t = np.sort(t[t > 0])[::-1].astype(np.int64)
    ns, nt = len(s), len(t)
    if ns == 0 or nt == 0:
        return (s.sum() == 0 and t.sum() == 0)
    if s.sum() != t.sum():
        return False

    # Gale-Ryser: Σ_{i=1}^k s_i <= Σ_j min(k, t_j)  for all k
    # t の接尾辞和を用いて高速化
    suffix_t = np.zeros(nt + 1, dtype=np.int64)
    suffix_t[:-1] = np.cumsum(t[::-1])[::-1]

    cumsum_s = np.cumsum(s)
    k = np.arange(1, ns + 1, dtype=np.int64)
    # Σ_j min(k, t_j): t は非増加なので binary search で境界を発見
    j = np.searchsorted(-t, -k, side='left')   # 最初に t[j] < k となるインデックス
    rhs = j * k + suffix_t[j]
    return bool(np.all(cumsum_s <= rhs))


def _triplet_graphical_bipartite(k_new, P_fixed, Q_fixed, eps, n_u, n_v):
    """
    Bassler et al. (2015) Theorem gratri: 部分的二部次数列 (P, Q, eps) の
    graphicality テスト。

    k_new を追加した後の P-side が balanced realization を持つかを確認する。
    P-side = 「現ノードの次数クラス」側、Q-side = 「相手クラス」側。
    _is_bigraphical の対称性により、どちら側のノードでも同じ関数で判定できる。

    Parameters
    ----------
    k_new    : 現ノードの割り当て値（P-side に追加）
    P_fixed  : P-side の既確定値リスト
    Q_fixed  : Q-side の既確定値リスト
    eps      : サブグラフの総エッジ数 = J_{alpha, beta}
    n_u      : P-side の総ノード数
    n_v      : Q-side の総ノード数
    """
    sum_P = sum(P_fixed) + k_new
    n_A = len(P_fixed) + 1   # P-side 確定数（現ノード含む）
    n_B = n_u - n_A           # P-side 残余ノード数
    eps_B = eps - sum_P       # P-side 残余スタブ数

    if eps_B < 0:
        return False
    if n_B == 0:
        if eps_B != 0:
            return False
    else:
        if eps_B > n_B * n_v:
            return False
        if (eps_B + n_B - 1) // n_B > n_v:   # ceil(eps_B / n_B) > cap
            return False

    sum_Q = sum(Q_fixed)
    n_H = len(Q_fixed)        # Q-side 確定数
    n_K = n_v - n_H           # Q-side 残余ノード数
    eps_K = eps - sum_Q       # Q-side 残余スタブ数

    if eps_K < 0:
        return False
    if n_K == 0:
        if eps_K != 0:
            return False
    else:
        if eps_K > n_K * n_u:
            return False
        if (eps_K + n_K - 1) // n_K > n_u:   # ceil(eps_K / n_K) > cap
            return False

    # Balanced completion for P-side remaining (B)
    if n_B > 0:
        f = eps_B // n_B
        c = eps_B % n_B
        B_bal = [f + 1] * c + [f] * (n_B - c)
    else:
        B_bal = []

    # Balanced completion for Q-side remaining (K)
    if n_K > 0:
        f = eps_K // n_K
        c = eps_K % n_K
        K_bal = [f + 1] * c + [f] * (n_K - c)
    else:
        K_bal = []

    U = np.array(list(P_fixed) + [k_new] + B_bal, dtype=np.int64)
    V = np.array(list(Q_fixed) + K_bal, dtype=np.int64)
    return _is_bigraphical(U, V)


def _triplet_graphical_unipartite(k_new, P_fixed, eps_total, n_nodes):
    """
    Bassler et al. (2015) Theorem gratri: 部分的一部次数列の graphicality テスト。

    Parameters
    ----------
    k_new     : 現ノードの割り当て値
    P_fixed   : 既確定ノードの次数リスト
    eps_total : 2 * J_{alpha,alpha} = jdm[(alpha,alpha)]（全次数の合計）
    n_nodes   : このクラスの総ノード数
    """
    sum_P = sum(P_fixed) + k_new
    n_A = len(P_fixed) + 1    # 確定ノード数（現ノード含む）
    n_B = n_nodes - n_A       # 残余ノード数
    remaining = eps_total - sum_P

    if remaining < 0:
        return False
    if n_B == 0:
        if remaining != 0:
            return False
        return _is_graphical(np.array(list(P_fixed) + [k_new], dtype=np.int64))

    if remaining > n_B * (n_nodes - 1):   # cap = n_nodes - 1 (no self-loops)
        return False
    if (remaining + n_B - 1) // n_B > n_nodes - 1:
        return False

    f = remaining // n_B
    c = remaining % n_B
    B_bal = [f + 1] * c + [f] * (n_B - c)

    full = np.array(list(P_fixed) + [k_new] + B_bal, dtype=np.int64)
    return _is_graphical(full)



def _randomize_bipartite_exact(nodes_u, nodes_v, deg_u, deg_v, rng):
    """
    Kim et al. (2012) に基づく二部グラフ exact sampler。
    nodes_u, nodes_v: ノードIDの配列
    deg_u[i]: nodes_u[i] の次数
    deg_v[j]: nodes_v[j] の次数

    GR チェックは「残余 U シーケンス全体」と「残余 candidates の V シーケンス」を
    使用する（node i のみの単一要素チェックでは不正確）。

    Returns
    -------
    edges : list of (u, v) tuples
    log_weight : float
    """
    n_u = len(nodes_u)
    n_v = len(nodes_v)

    # U ノードを次数降順でソート（Kim et al. アルゴリズムの前提）
    sort_order = np.argsort(-deg_u.astype(np.int64))
    nodes_u = nodes_u[sort_order]
    res_u = deg_u[sort_order].copy().astype(np.int64)
    res_v = deg_v.copy().astype(np.int64)
    edges = []
    log_weight = 0.0

    for i in range(n_u):
        if res_u[i] == 0:
            continue
        d_i_initial = int(res_u[i])

        # 候補: res_v > 0 の V ノードを res_v 降順でソート
        cand_idx = np.where(res_v > 0)[0]
        order = np.argsort(-res_v[cand_idx])
        candidates = list(cand_idx[order])

        first_step = True

        while res_u[i] > 0:
            n_cand = len(candidates)
            if n_cand == 0:
                break  # 通常は起きないはず

            if first_step:
                lo = n_cand
                first_step = False
            else:
                d_rem = int(res_u[i])
                if n_cand <= d_rem:
                    lo = n_cand
                else:
                    # 正しい GR チェック:
                    # node i を j に接続した後の残余シーケンスで判定
                    # U 側: [res_u[i]-1, res_u[i+1], ..., res_u[n_u-1]]
                    # V 側: candidates から j を除いた res_v 値
                    u_rem_tail = res_u[i + 1:]  # i より後の U ノードの残余次数

                    j_last = candidates[-1]
                    res_u[i] -= 1
                    res_v[j_last] -= 1
                    u_rem = np.concatenate([[res_u[i]], u_rem_tail])
                    v_rem = res_v[np.array(candidates[:-1], dtype=np.int64)]
                    ok_last = _is_bigraphical(u_rem, v_rem)
                    res_u[i] += 1
                    res_v[j_last] += 1

                    if ok_last:
                        lo = n_cand
                    else:
                        lo, hi = d_rem, n_cand - 1
                        while lo < hi:
                            mid = (lo + hi) // 2
                            j_mid = candidates[mid]
                            res_u[i] -= 1
                            res_v[j_mid] -= 1
                            u_rem = np.concatenate([[res_u[i]], u_rem_tail])
                            v_rem_cands = candidates[:mid] + candidates[mid + 1:]
                            v_rem = res_v[np.array(v_rem_cands, dtype=np.int64)]
                            ok = _is_bigraphical(u_rem, v_rem)
                            res_u[i] += 1
                            res_v[j_mid] += 1
                            if ok:
                                lo = mid + 1
                            else:
                                hi = mid

            log_weight += np.log(lo)
            a_idx = int(rng.integers(0, lo))
            j_chosen = candidates[a_idx]
            del candidates[a_idx]

            edges.append((nodes_u[i], nodes_v[j_chosen]))
            res_u[i] -= 1
            res_v[j_chosen] -= 1

        if d_i_initial > 1:
            log_weight -= np.sum(np.log(np.arange(1, d_i_initial + 1)))

    return edges, log_weight


def _sample_degree_spectra(degrees, jdm, nodes_of_class, rng):
    """
    Bassler et al. (2015) §4.1: degree spectra matrix をランダムサンプリングする。

    ノードを global index 順（0..N-1）に処理し、各ノードの各 beta クラスへの
    接続数 S_{beta, i} を以下の手順で決定する:

    1. 各 beta について Theorem gratri（triplet graphicality テスト）を用いて
       有効範囲 [m_beta, M_beta] を二分探索で求める。
    2. 残余予算制約 r = max(m, l-T)、R = min(M, l-t) を適用する。
    3. [r, R] から一様サンプルし、重み log(R-r+1) を累積する。

    この手順により、出力 DSM は構造的に bigraphical が保証される
    （修復ステップは不要）。

    Returns
    -------
    spectra    : dict node_id -> dict beta -> int
    log_weight : float  サンプル重みの対数（式 (8) の spectra weight）
    """
    N = len(degrees)
    degree_classes = sorted(set(int(d) for d in degrees if d > 0))
    spectra = {v: {} for v in range(N)}
    log_weight = 0.0

    for i in range(N):
        d_i = int(degrees[i])
        if d_i == 0:
            continue

        l = d_i  # ノード i の残余予算

        # J_{d_i, beta} > 0 の beta クラスを昇順に並べる（論文 α=1,2,...,Δ に対応）
        betas = sorted([b for b in degree_classes if jdm.get((d_i, b), 0) > 0])
        if not betas:
            continue

        # 各 beta の P_fixed / Q_fixed を事前計算（ノード i より前のノードに基づく）
        # P_fixed[beta]: V_{d_i} の確定済み spectra 値（v < i）
        # Q_fixed[beta]: V_beta の確定済み spectra 値（v < i）
        P_fixed_cache = {}
        Q_fixed_cache = {}
        for beta in betas:
            my_nodes = nodes_of_class[d_i]
            P_fixed_cache[beta] = [spectra[v].get(beta, 0) for v in my_nodes if v < i]
            if d_i != beta:
                other_nodes = nodes_of_class[beta]
                Q_fixed_cache[beta] = [spectra[u].get(d_i, 0) for u in other_nodes if u < i]

        def check_k(beta, k, l_curr):
            """k が S_{beta,i} として有効かを Theorem gratri で判定する。"""
            if k < 0:
                return False
            if d_i == beta:
                cap = len(nodes_of_class[d_i]) - 1
                if k > min(cap, l_curr):
                    return False
                return _triplet_graphical_unipartite(
                    k, P_fixed_cache[beta],
                    jdm.get((d_i, d_i), 0),
                    len(nodes_of_class[d_i])
                )
            else:
                cap = len(nodes_of_class[beta])
                if k > min(cap, l_curr):
                    return False
                return _triplet_graphical_bipartite(
                    k, P_fixed_cache[beta], Q_fixed_cache[beta],
                    jdm.get((d_i, beta), 0),
                    len(nodes_of_class[d_i]), len(nodes_of_class[beta])
                )

        def find_m_M(beta, l_curr):
            """
            Corollary cortri により有効範囲 [m, M] は連続。
            論文処方（§4.1）: k=0 から順次探索で m を求め、
            m を起点に二分探索で M を求める。
            """
            if d_i == beta:
                cap = len(nodes_of_class[d_i]) - 1
                eps_remaining = jdm.get((d_i, d_i), 0) - sum(P_fixed_cache[beta])
            else:
                cap = len(nodes_of_class[beta])
                eps_remaining = jdm.get((d_i, beta), 0) - sum(P_fixed_cache[beta])
            theo_max = min(cap, l_curr, eps_remaining)
            if theo_max < 0:
                return 0, 0

            # 論文処方: k=0 から順次スキャンして最初の True を m とする
            m = None
            for k in range(theo_max + 1):
                if check_k(beta, k, l_curr):
                    m = k
                    break
            if m is None:
                return 0, 0

            # m から theo_max の範囲は True*, False* 構造なので二分探索で M を求める
            lo, hi = m, theo_max
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if check_k(beta, mid, l_curr):
                    lo = mid
                else:
                    hi = mid - 1
            M = lo

            return m, M

        # 各 beta を昇順に処理（論文 §4.1 Step (iii)）
        for idx, beta in enumerate(betas):
            # Step (iii).a.1: 現 beta 以降の全 beta について m, M を計算
            m_M_all = {b: find_m_M(b, l) for b in betas[idx:]}

            m_b, M_b = m_M_all[beta]

            # Step (iii).a.2: 残余 betas の t = Σm, T = ΣM
            t = sum(m_M_all[b][0] for b in betas[idx + 1:])
            T = sum(m_M_all[b][1] for b in betas[idx + 1:])

            # Step (iii).a.3: 予算制約を適用
            r = max(m_b, l - T)
            R = min(M_b, l - t)

            # Step (iii).a.4: [r, R] から一様サンプル
            if r > R:
                val = r  # JDM が graphical なら発生しないはず
            elif r == R:
                val = r
            else:
                val = int(rng.integers(r, R + 1))
            log_weight += np.log(R - r + 1) if R > r else 0.0

            if val > 0:
                spectra[i][beta] = val
            l -= val

    return spectra, log_weight


def randomize_d2_exact(N, edges, degrees, rng):
    """
    Bassler et al. (2015) アルゴリズムによる d=2 ランダム化。
    結合次数行列 (JDM) を保存した simple graph を rejection-free で生成する。

    参考文献:
        K.E. Bassler, C.I. Del Genio, P.L. Erdős, I. Miklós, Z. Toroczkai,
        "Exact sampling of graphs with prescribed degree correlations",
        New Journal of Physics 17, 083052 (2015)

    アルゴリズム概要:
        1. Degree spectra matrix をサンプリング（各ノードの各次数クラスへの接続数を決定）
        2. 各 (α,β) サブグラフを independent に構築:
           - α=β (同次数クラス): randomize_d1_exact を使用
           - α≠β (異次数クラス): 二部グラフ exact sampler を使用
        3. 全サブグラフを統合して完成

    Returns
    -------
    edges_out : ndarray (M, 2)
    log_weight : float
        サンプル重みの対数（spectra weight + 各サブグラフの weight の合計）
    """
    jdm = _compute_jdm(N, edges, degrees)

    # 次数クラスの構築
    degree_classes = sorted(set(int(d) for d in degrees if d > 0))
    nodes_of_class = {k: [] for k in degree_classes}
    for v in range(N):
        k = int(degrees[v])
        if k > 0:
            nodes_of_class[k].append(v)

    # Step 1: degree spectra をサンプリング
    spectra, log_w_spectra = _sample_degree_spectra(
        degrees, jdm, nodes_of_class, rng
    )
    log_weight = log_w_spectra

    # Step 2: 各 (α, β) サブグラフを構築
    result_edges = []

    for alpha in degree_classes:
        for beta in degree_classes:
            if beta < alpha:
                continue

            if alpha == beta:
                e_ab = jdm.get((alpha, alpha), 0) // 2
            else:
                e_ab = jdm.get((alpha, beta), 0)

            if e_ab == 0:
                continue

            nodes_a = np.array(nodes_of_class[alpha], dtype=np.int64)
            nodes_b = np.array(nodes_of_class[beta], dtype=np.int64)

            # spectra から各ノードのこのサブグラフでの次数を取得
            deg_a = np.array([spectra[v].get(beta, 0) for v in nodes_a], dtype=np.int64)

            if alpha == beta:
                # 同次数クラス（unipartite）: randomize_d1_exact を使用
                sub_N = len(nodes_a)
                sub_edges, log_w_sub = randomize_d1_exact(
                    sub_N, np.empty((0, 2), dtype=np.int64), deg_a, rng
                )
                # ローカルインデックス → グローバルノードID
                for u_local, v_local in sub_edges:
                    result_edges.append([nodes_a[u_local], nodes_a[v_local]])
            else:
                # 異次数クラス（bipartite）: 二部グラフ sampler を使用
                deg_b = np.array([spectra[v].get(alpha, 0) for v in nodes_b], dtype=np.int64)
                sub_edges, log_w_sub = _randomize_bipartite_exact(
                    nodes_a, nodes_b, deg_a, deg_b, rng
                )
                result_edges.extend(sub_edges)

            log_weight += log_w_sub

    edges_out = (
        np.array(result_edges, dtype=np.int64)
        if result_edges
        else np.empty((0, 2), dtype=np.int64)
    )
    return edges_out, log_weight


# -----------------------------------------------------------------------
# d = 1.5  次数分布 + DDCC 保存（次数相関は保存しない）
# -----------------------------------------------------------------------

def randomize_d15(N, edges, degrees, rng, rewiring_coeff=500, simple=False):
    """
    次数分布 P(k) と度数依存クラスタリング係数 c(k) を保存したランダム化。
    次数相関（JDM）は保存しない点が d=2.5 と異なる。

    手順:
      1. d=1 で P(k) を保存したランダム化（Configuration Model）
      2. 次数保存リワイヤリングで c(k) を目標値に近づける

    Returns
    -------
    edges_out : ndarray (M, 2)
    """
    rand_edges, _ = randomize_d1_exact(N, edges, degrees, rng)

    max_k = int(degrees.max())
    k_size = max_k + 1

    target_ddcc, N_k = _compute_ddcc(N, edges, degrees, max_k)

    rand_degrees = _compute_degrees(N, rand_edges)
    current_ddcc, _ = _compute_ddcc(N, rand_edges, rand_degrees, max_k)

    node_degree = rand_degrees.astype(np.int64)
    coeff = np.zeros(k_size, dtype=np.float64)
    for k in range(2, k_size):
        if N_k[k] > 0:
            coeff[k] = 2.0 / (k * (k - 1)) / N_k[k]

    seed = int(rng.integers(0, 2**31))

    print(f"d=1.5 rewiring started ({rewiring_coeff * len(rand_edges)} trials)")
    result_edges, final_dist, norm = rewiring_loop_d15(
        N, rand_edges, node_degree,
        target_ddcc, current_ddcc,
        N_k, coeff, k_size, seed, rewiring_coeff, int(simple)
    )

    if norm > 0:
        print(f"Final L1 distance: {final_dist / norm:.6f}")

    return result_edges


# -----------------------------------------------------------------------
# d = 2  JDM 保存
# -----------------------------------------------------------------------

def randomize_d2(N, edges, degrees, rng):
    """
    結合次数行列 (JDM) を保存したランダム化。

    Returns
    -------
    edges_out : ndarray (M, 2)
    """
    jdm = _compute_jdm(N, edges, degrees)
    # jdm は有向（両方向）カウント。次数 k->l の有向エッジ数 = jdm[(k,l)]

    # 各次数ごとのスタブリストを作成してシャッフル
    stub_lists = {}
    for v in range(N):
        k = int(degrees[v])
        if k not in stub_lists:
            stub_lists[k] = []
        for _ in range(k):
            stub_lists[k].append(v)

    for k in stub_lists:
        arr = np.array(stub_lists[k], dtype=np.int64)
        rng.shuffle(arr)
        stub_lists[k] = list(arr)

    # JDM のキーをシャッフルして順番に接続
    ks = list(set(k for k, l in jdm.keys()))
    rng.shuffle(np.array(ks))  # in-place shuffle of list via numpy

    result_edges = []
    jdm_work = dict(jdm)

    for k in ks:
        ls = list(set(l for (kk, l) in jdm_work.keys() if kk == k))
        ls_arr = np.array(ls, dtype=np.int64)
        rng.shuffle(ls_arr)
        for l in ls_arr:
            while jdm_work.get((k, l), 0) > 0:
                u = stub_lists[k].pop()
                v = stub_lists[l].pop()
                result_edges.append((u, v))
                jdm_work[(k, l)] -= 1
                jdm_work[(l, k)] -= 1

    return np.array(result_edges, dtype=np.int64)


# -----------------------------------------------------------------------
# d = 2.5  JDM + DDCC 保存
# -----------------------------------------------------------------------

def randomize_d25(N, edges, degrees, rng, rewiring_coeff=500, simple=False):
    """
    JDM と度数依存クラスタリング係数 (DDCC) を保存したランダム化。
    内部ループは Numba JIT でコンパイル。

    Returns
    -------
    edges_out : ndarray (M, 2)
    """
    # まず d=2 でランダム化（simple=True の場合は exact サンプリングで simple graph を保証）
    if simple:
        rand_edges, _ = randomize_d2_exact(N, edges, degrees, rng)
    else:
        rand_edges = randomize_d2(N, edges, degrees, rng)

    max_k = int(degrees.max())
    k_size = max_k + 1

    # 目標 DDCC（元ネットワーク）
    target_ddcc, N_k = _compute_ddcc(N, edges, degrees, max_k)

    # 初期 DDCC（d=2 ランダム化後）
    rand_degrees = _compute_degrees(N, rand_edges)
    current_ddcc, _ = _compute_ddcc(N, rand_edges, rand_degrees, max_k)

    # Numba 用の係数配列
    node_degree = rand_degrees.astype(np.int64)
    coeff = np.zeros(k_size, dtype=np.float64)
    for k in range(2, k_size):
        if N_k[k] > 0:
            coeff[k] = 2.0 / (k * (k - 1)) / N_k[k]

    seed = int(rng.integers(0, 2**31))

    print(f"d=2.5 rewiring started ({rewiring_coeff * len(rand_edges)} trials)")
    result_edges, final_dist, norm = rewiring_loop_d25(
        N, rand_edges, node_degree,
        target_ddcc, current_ddcc,
        N_k, coeff, k_size, seed, rewiring_coeff, int(simple)
    )

    if norm > 0:
        print(f"Final L1 distance: {final_dist / norm:.6f}")

    return result_edges
