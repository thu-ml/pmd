import numpy as np
from approxla import approx_linear_sum_assignment_func, hungarian_max_func, pga_func, rm_func, sparse_hungarian_min_func


def reverse(x):
    return np.max(x) - x


def approx_lsa(costs):
    N = costs.shape[0]
    rs = np.zeros(N, np.int32)
    cs = np.zeros(N, np.int32)
    rm_func(costs.astype(np.float32), rs, cs)
    return rs, cs


def approx_max_lsa(costs):
    N = costs.shape[0]
    rs = np.zeros(N, np.int32)
    cs = np.zeros(N, np.int32)
    rm_func(reverse(costs.astype(np.float32)), rs, cs)
    return rs, cs


def lsa(costs):
    N = costs.shape[0]
    rs = np.zeros(N, np.int32)
    cs = np.zeros(N, np.int32)
    hungarian_max_func(reverse(costs.astype(np.float32)), rs, cs)
    return rs, cs


def sparse_lsa(costs):
    N = costs.shape[0]
    rs = np.zeros(N, np.int32)
    cs = np.zeros(N, np.int32)
    sparse_hungarian_min_func(costs.astype(np.float32), rs, cs)
    return rs, cs


def sparse_max_lsa(costs):
    N = costs.shape[0]
    rs = np.zeros(N, np.int32)
    cs = np.zeros(N, np.int32)
    sparse_hungarian_min_func(reverse(costs.astype(np.float32)), rs, cs)
    return rs, cs


def max_lsa(costs):
    N = costs.shape[0]
    rs = np.zeros(N, np.int32)
    cs = np.zeros(N, np.int32)
    hungarian_max_func(costs.astype(np.float32), rs, cs)
    return rs, cs


def assignments(rs, cs):
    result = np.zeros(rs.shape)
    result[rs] = cs
    return result.astype(np.int32)


def get_assignments(costs, method):
    if method == 'e':
        solver = lsa
    elif method == 'r':
        solver = approx_lsa

    rs, cs = solver(costs)
    a      = assignments(rs, cs)
    result = costs[rs, cs].mean()
    print(costs[rs, cs])
    return a, result
    
