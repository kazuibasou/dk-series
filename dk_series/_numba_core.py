"""
Numba JIT でコンパイルされる d=2.5 / d=1.5 再配線ループ。

C++ の targeting_rewiring_d_two_five / calculate_num_tri_to_add を忠実に移植。
d=1.5 は同じ再配線ロジックを次数相関の制約なしで適用する独自拡張。
"""
import numpy as np
import numba


# -----------------------------------------------------------------------
# 隣接リスト操作（パディング付き2次元配列 nlist[N, max_k]）
# -----------------------------------------------------------------------

@numba.njit(cache=True)
def _count_in_nlist(nlist, ncount, v, target):
    """nlist[v] の中に target が何個あるか数える（C++ の std::count 相当）"""
    count = 0
    for i in range(ncount[v]):
        if nlist[v, i] == target:
            count += 1
    return count


@numba.njit(cache=True)
def _add_edge_nlist(nlist, ncount, v, w):
    """nlist に無向エッジ (v,w) を追加する"""
    nlist[v, ncount[v]] = w
    ncount[v] += 1
    nlist[w, ncount[w]] = v
    ncount[w] += 1


@numba.njit(cache=True)
def _remove_edge_nlist(nlist, ncount, v, w):
    """nlist から無向エッジ (v,w) を1本削除する（末尾要素で上書き）"""
    for i in range(ncount[v]):
        if nlist[v, i] == w:
            ncount[v] -= 1
            nlist[v, i] = nlist[v, ncount[v]]
            break
    for i in range(ncount[w]):
        if nlist[w, i] == v:
            ncount[w] -= 1
            nlist[w, i] = nlist[w, ncount[w]]
            break


# -----------------------------------------------------------------------
# 再配線可能性チェック
# -----------------------------------------------------------------------

@numba.njit(cache=True)
def _rewirable(u1, v1, u2, v2, node_degree):
    """C++ の rewirable() を移植。4ノードが異なり、かつ次数が一致するペアがあるか確認（d=2.5用）"""
    if u1 == v1 or u1 == u2 or u1 == v2:
        return False
    if v1 == u2 or v1 == v2 or u2 == v2:
        return False
    return (node_degree[u1] == node_degree[u2] or
            node_degree[u1] == node_degree[v2] or
            node_degree[v1] == node_degree[u2] or
            node_degree[v1] == node_degree[v2])


@numba.njit(cache=True)
def _rewirable_d15(u1, v1, u2, v2):
    """4ノードが全て異なることのみ確認（d=1.5用: 次数相関の制約なし）"""
    if u1 == v1 or u1 == u2 or u1 == v2:
        return False
    if v1 == u2 or v1 == v2 or u2 == v2:
        return False
    return True


# -----------------------------------------------------------------------
# 三角形変化量の計算
# -----------------------------------------------------------------------

@numba.njit(cache=True)
def _calculate_num_tri_to_add(nlist, ncount, node_degree, u1, v1, v2, num_tri_to_add):
    """
    C++ の calculate_num_tri_to_add() を移植。
    エッジ (u1,v1) を除いて (u1,v2) を加えたとき、各次数クラスの三角形数の変化を計算する。
    """
    for i in range(ncount[u1]):
        k = nlist[u1, i]
        if node_degree[k] <= 1 or u1 == k:
            continue

        if v1 != k and node_degree[v1] > 1:
            t_minus = _count_in_nlist(nlist, ncount, v1, k)
            num_tri_to_add[node_degree[u1]] -= t_minus
            num_tri_to_add[node_degree[v1]] -= t_minus
            num_tri_to_add[node_degree[k]] -= t_minus

        if v2 != k and node_degree[v2] > 1:
            t_plus = _count_in_nlist(nlist, ncount, v2, k)
            num_tri_to_add[node_degree[u1]] += t_plus
            num_tri_to_add[node_degree[k]] += t_plus
            num_tri_to_add[node_degree[v2]] += t_plus


# -----------------------------------------------------------------------
# メイン再配線ループ
# -----------------------------------------------------------------------

@numba.njit(cache=True)
def rewiring_loop_d25(N, edges, node_degree, target_ddcc, current_ddcc,
                      N_k, coeff, k_size, seed, rewiring_coeff=500, simple=0):
    """
    C++ の targeting_rewiring_d_two_five() の内部ループを移植。

    Parameters
    ----------
    N : int
    edges : ndarray, shape (M, 2)  エッジリスト（内部インデックス）
    node_degree : ndarray (N,)     各ノードの次数（不変）
    target_ddcc : ndarray (k_size,) 目標 DDCC
    current_ddcc : ndarray (k_size,) 初期 DDCC（書き換え）
    N_k : ndarray (k_size,)         各次数のノード数
    coeff : ndarray (k_size,)       coeff[k] = 2/(k*(k-1)) / N_k[k]
    k_size : int                    max_k + 1
    seed : int

    Returns
    -------
    cur_edges : ndarray (M, 2)  再配線後のエッジリスト
    final_dist : float          最終 L1 距離
    norm : float                正規化係数
    """
    np.random.seed(seed)
    M = edges.shape[0]

    # 最大次数（nlist のパディング幅を決める。余裕を持たせて +1）
    max_k = 0
    for v in range(N):
        if node_degree[v] > max_k:
            max_k = node_degree[v]
    pad = max_k + 1  # 自己ループ対策で +1

    # パディング付き隣接リストを構築
    nlist = np.full((N, pad + 1), -1, dtype=np.int64)
    ncount = np.zeros(N, dtype=np.int64)
    for i in range(M):
        u = edges[i, 0]
        v = edges[i, 1]
        nlist[u, ncount[u]] = v
        ncount[u] += 1
        nlist[v, ncount[v]] = u
        ncount[v] += 1

    cur_edges = edges.copy()

    # 初期 L1 距離
    dist = 0.0
    norm = 0.0
    for k in range(k_size):
        if N_k[k] > 0:
            norm += target_ddcc[k]
            dist += abs(target_ddcc[k] - current_ddcc[k])

    num_tri_to_add = np.zeros(k_size, dtype=np.int64)
    rewired_ddcc = current_ddcc.copy()
    R = rewiring_coeff * M

    for r in range(R):
        # num_tri_to_add をリセット
        for k in range(k_size):
            num_tri_to_add[k] = 0

        # 再配線可能なエッジペアを探す
        i_e1 = np.random.randint(0, M)
        i_e2 = np.random.randint(0, M)
        u1 = cur_edges[i_e1, 0]; v1 = cur_edges[i_e1, 1]
        u2 = cur_edges[i_e2, 0]; v2 = cur_edges[i_e2, 1]
        while not _rewirable(u1, v1, u2, v2, node_degree):
            i_e1 = np.random.randint(0, M)
            i_e2 = np.random.randint(0, M)
            u1 = cur_edges[i_e1, 0]; v1 = cur_edges[i_e1, 1]
            u2 = cur_edges[i_e2, 0]; v2 = cur_edges[i_e2, 1]

        # rewired_ddcc を初期化
        for k in range(k_size):
            rewired_ddcc[k] = current_ddcc[k]
        rewired_dist = dist

        if (node_degree[u1] == node_degree[u2] or
                node_degree[v1] == node_degree[v2]):
            # Case 0: (u1,v1)+(u2,v2) → (u1,v2)+(u2,v1)
            if simple and (_count_in_nlist(nlist, ncount, u1, v2) > 0 or
                           _count_in_nlist(nlist, ncount, u2, v1) > 0):
                continue
            _calculate_num_tri_to_add(nlist, ncount, node_degree,
                                      u1, v1, v2, num_tri_to_add)
            _calculate_num_tri_to_add(nlist, ncount, node_degree,
                                      u2, v2, v1, num_tri_to_add)

            if node_degree[v1] > 1 and node_degree[v2] > 1:
                t = _count_in_nlist(nlist, ncount, v1, v2)
                num_tri_to_add[node_degree[u1]] -= t
                num_tri_to_add[node_degree[v1]] -= 2 * t
                num_tri_to_add[node_degree[u2]] -= t
                num_tri_to_add[node_degree[v2]] -= 2 * t

            if node_degree[u1] > 1 and node_degree[u2] > 1:
                t = _count_in_nlist(nlist, ncount, u2, u1)
                if node_degree[v1] > 1:
                    num_tri_to_add[node_degree[u1]] -= t
                    num_tri_to_add[node_degree[u2]] -= t
                    num_tri_to_add[node_degree[v1]] -= t
                if node_degree[v2] > 1:
                    num_tri_to_add[node_degree[u1]] -= t
                    num_tri_to_add[node_degree[u2]] -= t
                    num_tri_to_add[node_degree[v2]] -= t

            rewiring_case = 0

        else:
            # Case 1: (u1,v1)+(u2,v2) → (u1,u2)+(v1,v2)
            if simple and (_count_in_nlist(nlist, ncount, u1, u2) > 0 or
                           _count_in_nlist(nlist, ncount, v1, v2) > 0):
                continue
            _calculate_num_tri_to_add(nlist, ncount, node_degree,
                                      u1, v1, u2, num_tri_to_add)
            _calculate_num_tri_to_add(nlist, ncount, node_degree,
                                      v2, u2, v1, num_tri_to_add)

            if node_degree[v1] > 1 and node_degree[u2] > 1:
                t = _count_in_nlist(nlist, ncount, v1, u2)
                num_tri_to_add[node_degree[u1]] -= t
                num_tri_to_add[node_degree[v1]] -= 2 * t
                num_tri_to_add[node_degree[u2]] -= 2 * t
                num_tri_to_add[node_degree[v2]] -= t

            if node_degree[u1] > 1 and node_degree[v2] > 1:
                t = _count_in_nlist(nlist, ncount, v2, u1)
                if node_degree[v1] > 1:
                    num_tri_to_add[node_degree[u1]] -= t
                    num_tri_to_add[node_degree[v2]] -= t
                    num_tri_to_add[node_degree[v1]] -= t
                if node_degree[u2] > 1:
                    num_tri_to_add[node_degree[u1]] -= t
                    num_tri_to_add[node_degree[v2]] -= t
                    num_tri_to_add[node_degree[u2]] -= t

            rewiring_case = 1

        # 新しい L1 距離を計算
        for k in range(2, k_size):
            if num_tri_to_add[k] == 0:
                continue
            rewired_ddcc[k] += num_tri_to_add[k] * coeff[k]
            rewired_dist += (abs(target_ddcc[k] - rewired_ddcc[k]) -
                             abs(target_ddcc[k] - current_ddcc[k]))

        if rewired_dist < dist:
            if rewiring_case == 0:
                _remove_edge_nlist(nlist, ncount, u1, v1)
                _add_edge_nlist(nlist, ncount, u1, v2)
                _remove_edge_nlist(nlist, ncount, u2, v2)
                _add_edge_nlist(nlist, ncount, v1, u2)
                # エッジリスト更新: v1 と v2 を交換
                cur_edges[i_e1, 1] = v2
                cur_edges[i_e2, 1] = v1
            else:
                _remove_edge_nlist(nlist, ncount, u1, v1)
                _add_edge_nlist(nlist, ncount, u1, u2)
                _remove_edge_nlist(nlist, ncount, u2, v2)
                _add_edge_nlist(nlist, ncount, v1, v2)
                # エッジリスト更新: e1の second と e2の first を交換
                cur_edges[i_e1, 1] = u2
                cur_edges[i_e2, 0] = v1

            for k in range(k_size):
                current_ddcc[k] = rewired_ddcc[k]
            dist = rewired_dist

    return cur_edges, dist, norm


@numba.njit(cache=True)
def rewiring_loop_d15(N, edges, node_degree, target_ddcc, current_ddcc,
                      N_k, coeff, k_size, seed, rewiring_coeff=500, simple=0):
    """
    d=1.5 用の再配線ループ。

    d=2.5 と同じ DDCC 最小化ロジックを使うが、次数相関（JDM）の制約を持たない。
    任意の4ノード相異なるエッジペアを交換対象とし、case 0/1 をランダムに選択する。

    Parameters
    ----------
    N : int
    edges : ndarray, shape (M, 2)   d=1 ランダム化後のエッジリスト
    node_degree : ndarray (N,)      各ノードの次数（不変）
    target_ddcc : ndarray (k_size,) 目標 DDCC（元ネットワーク）
    current_ddcc : ndarray (k_size,) 初期 DDCC（d=1 ランダム化後）
    N_k : ndarray (k_size,)         各次数のノード数
    coeff : ndarray (k_size,)       coeff[k] = 2/(k*(k-1)) / N_k[k]
    k_size : int                    max_k + 1
    seed : int
    rewiring_coeff : int            試行回数 = rewiring_coeff * M

    Returns
    -------
    cur_edges : ndarray (M, 2)
    final_dist : float
    norm : float
    """
    np.random.seed(seed)
    M = edges.shape[0]

    max_k = 0
    for v in range(N):
        if node_degree[v] > max_k:
            max_k = node_degree[v]
    pad = max_k + 1

    nlist = np.full((N, pad + 1), -1, dtype=np.int64)
    ncount = np.zeros(N, dtype=np.int64)
    for i in range(M):
        u = edges[i, 0]
        v = edges[i, 1]
        nlist[u, ncount[u]] = v
        ncount[u] += 1
        nlist[v, ncount[v]] = u
        ncount[v] += 1

    cur_edges = edges.copy()

    dist = 0.0
    norm = 0.0
    for k in range(k_size):
        if N_k[k] > 0:
            norm += target_ddcc[k]
            dist += abs(target_ddcc[k] - current_ddcc[k])

    num_tri_to_add = np.zeros(k_size, dtype=np.int64)
    rewired_ddcc = current_ddcc.copy()
    R = rewiring_coeff * M

    for r in range(R):
        for k in range(k_size):
            num_tri_to_add[k] = 0

        # 4ノード相異なるエッジペアを探す（次数一致の制約なし）
        i_e1 = np.random.randint(0, M)
        i_e2 = np.random.randint(0, M)
        u1 = cur_edges[i_e1, 0]; v1 = cur_edges[i_e1, 1]
        u2 = cur_edges[i_e2, 0]; v2 = cur_edges[i_e2, 1]
        while not _rewirable_d15(u1, v1, u2, v2):
            i_e1 = np.random.randint(0, M)
            i_e2 = np.random.randint(0, M)
            u1 = cur_edges[i_e1, 0]; v1 = cur_edges[i_e1, 1]
            u2 = cur_edges[i_e2, 0]; v2 = cur_edges[i_e2, 1]

        for k in range(k_size):
            rewired_ddcc[k] = current_ddcc[k]
        rewired_dist = dist

        # case 0 / case 1 をランダムに選択（どちらも次数保存）
        if np.random.randint(0, 2) == 0:
            # Case 0: (u1,v1)+(u2,v2) → (u1,v2)+(u2,v1)
            if simple and (_count_in_nlist(nlist, ncount, u1, v2) > 0 or
                           _count_in_nlist(nlist, ncount, u2, v1) > 0):
                continue
            _calculate_num_tri_to_add(nlist, ncount, node_degree,
                                      u1, v1, v2, num_tri_to_add)
            _calculate_num_tri_to_add(nlist, ncount, node_degree,
                                      u2, v2, v1, num_tri_to_add)

            if node_degree[v1] > 1 and node_degree[v2] > 1:
                t = _count_in_nlist(nlist, ncount, v1, v2)
                num_tri_to_add[node_degree[u1]] -= t
                num_tri_to_add[node_degree[v1]] -= 2 * t
                num_tri_to_add[node_degree[u2]] -= t
                num_tri_to_add[node_degree[v2]] -= 2 * t

            if node_degree[u1] > 1 and node_degree[u2] > 1:
                t = _count_in_nlist(nlist, ncount, u2, u1)
                if node_degree[v1] > 1:
                    num_tri_to_add[node_degree[u1]] -= t
                    num_tri_to_add[node_degree[u2]] -= t
                    num_tri_to_add[node_degree[v1]] -= t
                if node_degree[v2] > 1:
                    num_tri_to_add[node_degree[u1]] -= t
                    num_tri_to_add[node_degree[u2]] -= t
                    num_tri_to_add[node_degree[v2]] -= t

            rewiring_case = 0

        else:
            # Case 1: (u1,v1)+(u2,v2) → (u1,u2)+(v1,v2)
            if simple and (_count_in_nlist(nlist, ncount, u1, u2) > 0 or
                           _count_in_nlist(nlist, ncount, v1, v2) > 0):
                continue
            _calculate_num_tri_to_add(nlist, ncount, node_degree,
                                      u1, v1, u2, num_tri_to_add)
            _calculate_num_tri_to_add(nlist, ncount, node_degree,
                                      v2, u2, v1, num_tri_to_add)

            if node_degree[v1] > 1 and node_degree[u2] > 1:
                t = _count_in_nlist(nlist, ncount, v1, u2)
                num_tri_to_add[node_degree[u1]] -= t
                num_tri_to_add[node_degree[v1]] -= 2 * t
                num_tri_to_add[node_degree[u2]] -= 2 * t
                num_tri_to_add[node_degree[v2]] -= t

            if node_degree[u1] > 1 and node_degree[v2] > 1:
                t = _count_in_nlist(nlist, ncount, v2, u1)
                if node_degree[v1] > 1:
                    num_tri_to_add[node_degree[u1]] -= t
                    num_tri_to_add[node_degree[v2]] -= t
                    num_tri_to_add[node_degree[v1]] -= t
                if node_degree[u2] > 1:
                    num_tri_to_add[node_degree[u1]] -= t
                    num_tri_to_add[node_degree[v2]] -= t
                    num_tri_to_add[node_degree[u2]] -= t

            rewiring_case = 1

        for k in range(2, k_size):
            if num_tri_to_add[k] == 0:
                continue
            rewired_ddcc[k] += num_tri_to_add[k] * coeff[k]
            rewired_dist += (abs(target_ddcc[k] - rewired_ddcc[k]) -
                             abs(target_ddcc[k] - current_ddcc[k]))

        if rewired_dist < dist:
            if rewiring_case == 0:
                _remove_edge_nlist(nlist, ncount, u1, v1)
                _add_edge_nlist(nlist, ncount, u1, v2)
                _remove_edge_nlist(nlist, ncount, u2, v2)
                _add_edge_nlist(nlist, ncount, v1, u2)
                cur_edges[i_e1, 1] = v2
                cur_edges[i_e2, 1] = v1
            else:
                _remove_edge_nlist(nlist, ncount, u1, v1)
                _add_edge_nlist(nlist, ncount, u1, u2)
                _remove_edge_nlist(nlist, ncount, u2, v2)
                _add_edge_nlist(nlist, ncount, v1, v2)
                cur_edges[i_e1, 1] = u2
                cur_edges[i_e2, 0] = v1

            for k in range(k_size):
                current_ddcc[k] = rewired_ddcc[k]
            dist = rewired_dist

    return cur_edges, dist, norm
