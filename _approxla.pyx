""" Example of wrapping a C function that takes C double arrays as input using
    the Numpy declarations from Cython """

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np

# if you want to use the Numpy-C-API from Cython
# (not strictly necessary for this example)
np.import_array()

# cdefine the signature of our c function
cdef extern from "approxla.h":
    void approx_linear_sum_assignment(float * in_array, int * rs, int * cs, int N, bint reverse);
    void path_growing_algorithm(float * in_array, int * rs, int * cs, int N);
    void randomized_matching(float * in_array, int * rs, int * cs, int N);
    void hungarian_min(float * in_array, int * rs, int * cs, int N);
    void hungarian_max(float * in_array, int * rs, int * cs, int N);
    void sparse_hungarian_min(float * in_array, int * rs, int * cs, int N);
    void auction_min(float * in_array, int * rs, int * cs, int N);

cdef extern from "pairwise_l1.h":
    void PairwiseL1(float *A, float *B, float *C, int n, int m, int d);

# create the wrapper code, with numpy type annotations
def approx_linear_sum_assignment_func(np.ndarray[float, ndim=2, mode="c"] in_array not None,
                     np.ndarray[int, ndim=1, mode="c"] rs not None,
                     np.ndarray[int, ndim=1, mode="c"] cs not None):
    approx_linear_sum_assignment(<float*> np.PyArray_DATA(in_array),
                                 <int*> np.PyArray_DATA(rs),
                                 <int*> np.PyArray_DATA(cs),
                                 in_array.shape[0], False)

def pga_func(np.ndarray[float, ndim=2, mode="c"] in_array not None,
                     np.ndarray[int, ndim=1, mode="c"] rs not None,
                     np.ndarray[int, ndim=1, mode="c"] cs not None):
    path_growing_algorithm(<float*> np.PyArray_DATA(in_array),
                                 <int*> np.PyArray_DATA(rs),
                                 <int*> np.PyArray_DATA(cs),
                                 in_array.shape[0])

def rm_func(np.ndarray[float, ndim=2, mode="c"] in_array not None,
                     np.ndarray[int, ndim=1, mode="c"] rs not None,
                     np.ndarray[int, ndim=1, mode="c"] cs not None):
    randomized_matching(<float*> np.PyArray_DATA(in_array),
                                 <int*> np.PyArray_DATA(rs),
                                 <int*> np.PyArray_DATA(cs),
                                 in_array.shape[0])

def hungarian_min_func(np.ndarray[float, ndim=2, mode="c"] in_array not None,
                     np.ndarray[int, ndim=1, mode="c"] rs not None,
                     np.ndarray[int, ndim=1, mode="c"] cs not None):
    hungarian_min(<float*> np.PyArray_DATA(in_array),
                                 <int*> np.PyArray_DATA(rs),
                                 <int*> np.PyArray_DATA(cs),
                                 in_array.shape[0])

def hungarian_max_func(np.ndarray[float, ndim=2, mode="c"] in_array not None,

                     np.ndarray[int, ndim=1, mode="c"] rs not None,
                     np.ndarray[int, ndim=1, mode="c"] cs not None):
    hungarian_max(<float*> np.PyArray_DATA(in_array),
                                 <int*> np.PyArray_DATA(rs),
                                 <int*> np.PyArray_DATA(cs),
                                 in_array.shape[0])

def sparse_hungarian_min_func(np.ndarray[float, ndim=2, mode="c"] in_array not None,
                     np.ndarray[int, ndim=1, mode="c"] rs not None,
                     np.ndarray[int, ndim=1, mode="c"] cs not None):
    sparse_hungarian_min(<float*> np.PyArray_DATA(in_array),
                                 <int*> np.PyArray_DATA(rs),
                                 <int*> np.PyArray_DATA(cs),
                                 in_array.shape[0])

def auction_min_func(np.ndarray[float, ndim=2, mode="c"] in_array not None,
                     np.ndarray[int, ndim=1, mode="c"] rs not None,
                     np.ndarray[int, ndim=1, mode="c"] cs not None):
    auction_min(<float*> np.PyArray_DATA(in_array),
                                 <int*> np.PyArray_DATA(rs),
                                 <int*> np.PyArray_DATA(cs),
                                 in_array.shape[0])

# create the wrapper code, with numpy type annotations
def pairwise_l1(np.ndarray[float, ndim=2, mode="c"] A not None,
                np.ndarray[float, ndim=2, mode="c"] B not None,
                np.ndarray[float, ndim=2, mode="c"] C not None):
    PairwiseL1(<float*> np.PyArray_DATA(A),
               <float*> np.PyArray_DATA(B),
               <float*> np.PyArray_DATA(C),
               A.shape[0], B.shape[0], A.shape[1])
