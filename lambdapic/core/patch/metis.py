import pymetis
import numpy as np
from numpy.typing import NDArray
from .patch import Patches

def compute_rank(patches: Patches, nrank: int, weights: NDArray[np.int64], rank_prev: NDArray[np.int64]|None=None) -> tuple[list[int], list[int]]:
    assert all(np.array([p.index for p in patches]) == np.arange(len(patches))), "patch index must be 1..., n-1"
    adjacency_list = [p.neighbor_index[p.neighbor_index >= 0] for p in patches]

    opt = pymetis.Options(
        contig=True,
        # ufactor=1,
    )

    ncut, ranks = pymetis.part_graph(nrank, adjacency=adjacency_list, vweights=weights.astype(np.int64), options=opt)

    npatch = len(patches)
    if rank_prev is None:
        rank_prev = np.zeros(npatch, dtype=int)
    else:
        rank_prev = np.asarray(rank_prev)
        
    ipatch_rank_prev = [np.arange(npatch)[rank_prev == i] for i in range(nrank)]
    ipatch_rank = [np.arange(npatch)[np.asarray(ranks) == i] for i in range(nrank)]

    # the computed rank numbers are random,
    # compute the similarity of the previous and current ranks
    nsame = np.zeros((nrank, nrank), dtype=int)
    for i in range(nrank):
        for j in range(nrank):
            nsame[i, j] = np.intersect1d(ipatch_rank_prev[i], ipatch_rank[j]).size

    # ranks having the most same patches are the same ranks
    rank_map = arg_sort_rows_by_diagonal_max(nsame)

    ranks = [rank_map[rank_] for rank_ in ranks]
    npatch_per_rank = [sum(np.asarray(ranks) == i) for i in range(nrank)]

    return ranks, npatch_per_rank


def arg_sort_rows_by_diagonal_max(matrix) -> np.ndarray:
    # sort according to the max value of each row
    # larger value first
    cols_order = np.argsort(np.max(matrix, axis=0))[::-1]
    arr = matrix[:, cols_order]
    for i in range(min(arr.shape[0], arr.shape[1])):
        if i >= arr.shape[1]:
            break
        # 计算当前行从第i列开始的的最大值的列索引（全局）
        max_col = i + arr[i, i:].argmax()
        # 记录列交换
        cols_order[i], cols_order[max_col] = cols_order[max_col], cols_order[i]
        # 在数组中进行交换
        arr[:, [i, max_col]] = arr[:, [max_col, i]]
    rank_map = np.empty(matrix.shape[0], dtype=int)
    rank_map[cols_order] = np.arange(matrix.shape[0])
    return rank_map