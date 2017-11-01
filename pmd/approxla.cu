#include <cmath>
#include <iostream>
#include <algorithm>
#include <memory>
#include <vector>
#include <chrono>
#include <queue>
#include <memory.h>
#include <omp.h>
#include "approxla.h"
#include "utils.h"
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
using namespace std;
using std::chrono::high_resolution_clock;

#define RADIX 8
#define BUFFER_SIZE 65536

thrust::device_vector<float> d_arr;
thrust::device_vector<unsigned> d_ind;

void approx_linear_sum_assignment(float * in_array, int * rs, int * cs, 
        int N, bool maximum) {
    int size = N * N;
    // Sort
    auto start = high_resolution_clock::now();   

    thrust::host_vector<float> h_arr(in_array, in_array+size);
    thrust::host_vector<unsigned> indices(size);
    auto build_start = high_resolution_clock::now();   
    int cnt = 0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++, cnt++)
            indices[cnt] = (i<<16) + j;
    auto build_time = std::chrono::duration<double>(high_resolution_clock::now() - build_start).count();

    auto sort_start = std::chrono::high_resolution_clock::now();   
    d_arr.resize(h_arr.size());
    d_ind.resize(indices.size());
    thrust::copy(h_arr.begin(), h_arr.end(), d_arr.begin());
    thrust::copy(indices.begin(), indices.end(), d_ind.begin());
    thrust::sort_by_key(d_arr.begin(), d_arr.end(), d_ind.begin());
    h_arr = d_arr;
    indices = d_ind;
    auto sort_time = std::chrono::duration<double>(high_resolution_clock::now() - sort_start).count();
         
    if (maximum)
        std::reverse(indices.begin(), indices.end());
    //for (auto d: adata)
    //    printf("%lld\n", d);

    std::vector<bool> r_visited(N);
    std::vector<bool> c_visited(N);
    int num_res = 0;

    // Scan
    auto scan_start = high_resolution_clock::now();   
    for (int i = 0; i < size; i++) {
        unsigned rc = indices[i];
        int r = rc >> 16;
        int c = rc & 65535;
        //printf("%d %d\n", r, c);
        if (!r_visited[r] && !c_visited[c]) {
            rs[num_res] = r;
            cs[num_res] = c;
            r_visited[r] = true;
            c_visited[c] = true;
            num_res++;
            //printf("%d %d %d %d %f\n", i, rc, r, c, h_arr[i]);
        }
    }
    assert(num_res == N);
    //cout << "Num res = " << num_res << endl;
    auto scan_time = std::chrono::duration<double>(high_resolution_clock::now() - scan_start).count();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end-start;
    // printf("Greedy finished in %.6f (sort %.6f s, scan %.6f s, build %.6f s)\n", diff.count(), sort_time, scan_time, build_time);
}

void path_growing_algorithm(float *in_array, int *rs, int *cs, int N) {
    float w1 = 0, w2 = 0;
    vector<int> path; path.reserve(2*N);
    vector<bool> x_visited(N), y_visited(N);

    int current_x = 0;
    x_visited[current_x] = true;
    path.push_back(current_x);
    for (int i = 0; i < N; i++) {
        // Find the minimal edge for current_x
        float min_edge = 1e9; int min_y = -1;
        for (int j = 0; j < N; j++)
            if (!y_visited[j] && in_array[current_x*N+j] < min_edge) {
                min_edge = in_array[current_x*N+j];
                min_y = j;
            }
        y_visited[min_y] = true;
        path.push_back(min_y);
        w1 += min_edge;
        //cout << min_edge << " " << i << min_y << endl;
        if (i + 1 == N)
            break;

        // Find the minimal edge for min_y
        min_edge = 1e9; current_x = -1;
        for (int x = 0; x < N; x++)
            if (!x_visited[x] && in_array[x*N+min_y] < min_edge) {
                min_edge = in_array[x*N+min_y];
                current_x = x;
            }
        x_visited[current_x] = true;
        path.push_back(current_x);
        w2 += min_edge;
        //cout << min_edge << " " << current_x << min_y << endl;
    }
    //cout << path << endl;
    w2 += in_array[path[0]*N + path[2*N-1]];
    cout << w1 << " " << w2 << endl;
    if (w1 < w2) {
        for (int i = 0; i < N; i++) {
            rs[i] = path[i*2];
            cs[i] = path[i*2+1];
        }
    } else {
        for (int i = 0; i+1<N; i++) {
            rs[i] = path[i*2+2];
            cs[i] = path[i*2+1];
        }
        rs[N-1] = path[0];
        cs[N-1] = path[2*N-1];
    }
}

void randomized_matching(float *in_array, int *rs, int *cs, int N) {
    // Initialize with greedy
    //approx_linear_sum_assignment(in_array, rs, cs, N, false);
    //cout << "rs" << endl;
    //for (int i = 0; i < N; i++)
    //    cout << rs[i] << ' ';
    //cout << endl;
    //cout << "cs" << endl;
    //for (int i = 0; i < N; i++)
    //    cout << cs[i] << ' ';
    //cout << endl;
    std::iota(rs, rs+N, 0);
    std::iota(cs, cs+N, 0);

    auto r_start = high_resolution_clock::now();   
    vector<int> match_x(N);
    vector<int> match_y(N);
    for (int i = 0; i < N; i++) {
        match_x[rs[i]] = cs[i];
        match_y[cs[i]] = rs[i];
    }
    vector<float> gain_x(N);

    for (int iter = 0; iter < 2*N; iter++) {
        int v = generator() % N;
        fill(gain_x.begin(), gain_x.end(), 1e9);
        // Find 2 alternating cycle
        for (int j = 0; j < N; j++)
            if (j != match_x[v]) {
                int i = match_y[j];
                gain_x[i] = in_array[v*N+j] - in_array[i*N+j];
            }
        int u = match_x[v];
        int best_i = -1;
        // float best_gain = 1e9;
        float best_gain = 0; // Only accepts better move
        for (int i = 0; i < N; i++)
            if (i != v) {
                float gain_cycle = gain_x[i] + in_array[i*N+u] - in_array[v*N+u];
                if (gain_cycle < best_gain) {
                    best_gain = gain_cycle;
                    best_i = i;
                }
            }
        if (best_i != -1) {
            // Augment cycle: v = u - i = j
            int i = best_i;
            int j = match_x[i];
            match_x[v] = j; match_y[j] = v;
            match_x[i] = u; match_y[u] = i;
        }

        // Find 3 alternating cycle
        // Find the best arm
        float best_arm_gain = 1e9;
        u = match_x[v];
        int py = -1; 
        for (int j = 0; j < N; j++)
            if (j != u) {
                int i = match_y[j];
                float arm_gain = in_array[v*N+j] - in_array[i*N+j];
                if (arm_gain < best_arm_gain) {
                    best_arm_gain = arm_gain;
                    py = j;
                }
            }
        int px = match_y[py];

        // Enumerate the arm for u
        best_gain = 0; int qx = -1;
        for (int i = 0; i < N; i++)
            if (i != v && i != px) {
                int qy = match_x[i];
                float gain = best_arm_gain - in_array[v*N+u] + in_array[i*N+u] 
                                           - in_array[i*N+qy] + in_array[px*N+qy];
                if (gain < best_gain) {
                    best_gain = gain;
                    qx = i;
                }
            }
        int qy = match_x[qx];
        if (qx != -1) {
            // Augment cycle: v = u - qx = qy - px = py
            match_x[v] = py; match_y[py] = v;
            match_x[qx] = u; match_y[u] = qx;
            match_x[px] = qy; match_y[qy] = px;
        }
    }
    for (int i = 0; i < N; i++) {
        rs[i] = i;
        cs[i] = match_x[i];
    }
    auto r_time = std::chrono::duration<double>(high_resolution_clock::now() - r_start).count();
    // printf("Randomized matching tooks %f\n", r_time);
}
