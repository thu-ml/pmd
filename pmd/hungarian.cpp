#include <cmath>
#include <iostream>
#include <functional>
#include <algorithm>
#include <memory>
#include <vector>
#include <chrono>
#include <queue>
#include <memory.h>
#include <omp.h>
#include "approxla.h"
#include "utils.h"
#include "network_flow.h"
using namespace std;

// I assume the costs are positive
void hungarian_max(float *costs, int *rs, int *cs, int N) {
    vector<float> lx(N), ly(N), slack(N);
    vector<int> slack_x(N);             // argmin for slack
    vector<int> x2y(N, -1), y2x(N, -1); // Matches
    vector<int> prev_y(N);              // The parent of y_j in the Hungarian tree
    vector<bool> visited_x(N);          // Whether x_i is in the Hungarian tree
    // If prev_y[j] == -1, j is not in the Hungarian tree
    // If !visited_x[i],   i is not in the Hungarian tree

    // Initialize label
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            lx[i] = max(lx[i], costs[i*N + j]);
            if (costs[i*N + j] < 0)
                throw runtime_error("Cost must be non-negative");
        }

    // TODO try ordering heuristic
    // approx_linear_sum_assignment(costs, rs, cs, N, true);

    for (int i = 0; i < N; i++) {       // Match x_i
        // Feasible: lx[i] + ly[j] >= w[i, j]
        // Delta = min_{i\in S, j\not\in T} lx[i] + ly[j] - w[i, j]
        // Slack[j] = min_{i\in S} lx[i] + ly[j] - w[i, j]
        // Delta = \min_{j\not\in T} Slack[j]
        // int si = rs[i];
        int si = i;
        queue<int> q; q.push(si);
        bool matched = false;
        fill(prev_y.begin(), prev_y.end(), -1);
        fill(visited_x.begin(), visited_x.end(), false);

        for (int j = 0; j < N; j++) {
            slack[j] = lx[si] + ly[j] - costs[si*N+j];
            slack_x[j] = si;
        }
        // cout << "Init slack = " << slack << endl;

        auto augment = [&](int j) {
            // j:                   exposed y
            // current_x - j:       unmatched edge
            // current_x - next_y:  matched edge
            int current_x = prev_y[j];
            int next_y = x2y[current_x];
            x2y[current_x] = j; y2x[j] = current_x;
            // cout << "Find augmenting path! " << j << endl;
            // cout << current_x << ' ' << next_y << endl;
            while (current_x != si) {
                int next_x = prev_y[next_y];
                int next_next_y = x2y[next_x];
                x2y[next_x] = next_y;
                y2x[next_y] = next_x;
                current_x = next_x;
                next_y = next_next_y;
            }
            // cout << x2y << endl;
        };

        auto update_slack = [&](int x) {        // Add x into the tree
            // cout << "Update slack " << x << endl;
            for (int y = 0; y < N; y++) {
                float new_slack = lx[x]+ly[y]-costs[x*N+y];
                // cout << "y " << y << " " << lx[x] << " " << ly[y] << " " << costs[x*N+y] << " " << new_slack << endl;
                if (new_slack < slack[y]) {
                    slack[y] = new_slack;
                    slack_x[y] = x;
                }
            }
        };

        auto try_grow_tree = [&](int y, int s) {    // Try grow s-->y-->y2x[y], assuming that s--y is in the equality graph
            if (y2x[y] == -1) {                     // Unmatched, found an augmenting path
                prev_y[y] = s;
                augment(y);
                return matched = true;
            } else if (prev_y[y] == -1) {           // Add y and y2x[y] into the Hungarian tree
                // cout << "Put y" << y << " and x" << y2x[y] << " in the tree" << endl;
                int x = y2x[y];
                prev_y[y] = s;
                q.push(x);
                update_slack(x);
                // cout << "Slack = " << slack << endl;
            }
            return false;
        };

        // cout << "Matching " << si << endl;
        while (!matched) {
            while (!q.empty() && !matched) {
                int s = q.front(); q.pop(); visited_x[s] = true;
                // cout << "Visiting " << s << endl;
                // Grow the Hungarian tree
                for (int j = 0; j < N; j++)
                    if (near(lx[s]+ly[j], costs[s*N + j]))      // In the equality subgraph
                        if (try_grow_tree(j, s))
                            break;
            }
            if (matched)
                break;
            // Add a new y to the Hungarian tree
            int y = -1;
            float min_slack = 1e10;
            for (int j = 0; j < N; j++)
                if (prev_y[j] == -1 && slack[j] < min_slack) {
                    min_slack = slack[j];
                    y = j;
                }
            // Update label
            for (int x = 0; x < N; x++) if (visited_x[x]) lx[x] -= min_slack;
            for (int j = 0; j < N; j++) if (prev_y[j] != -1) ly[j] += min_slack;
            for (int j = 0; j < N; j++) slack[j] -= min_slack;
            // cout << "Relabel, delta=" << min_slack << " on edge " << slack_x[y] << "--" << y << endl; 
            // cout << prev_y << endl << lx << endl << ly << endl << slack << endl;
            if (try_grow_tree(y, slack_x[y]))
                break;
        } 
    }
    for (int i = 0; i < N; i++) {
        rs[i] = i;
        cs[i] = x2y[i];
    }
}


void hungarian_min(float *costs, int *rs, int *cs, int N) {
    float max_cost = -1e9;
    for (int i = 0; i < N*N; i++)
        max_cost = max(max_cost, costs[i]);
    vector<float> costs2(N*N);
    for (int i = 0; i < N*N; i++)
        costs2[i] = max_cost - costs[i];

    hungarian_max(costs2.data(), rs, cs, N);
}

void sparse_hungarian_max(float *costs, int *rs, int *cs, int N) {
    float max_cost = -1e9;
    for (int i = 0; i < N*N; i++)
        max_cost = max(max_cost, costs[i]);
    vector<float> costs2(N*N);
    for (int i = 0; i < N*N; i++)
        costs2[i] = max_cost - costs[i];
    sparse_hungarian_min(costs2.data(), rs, cs, N);
}


void sparse_hungarian_min(float *costs, int *rs, int *cs, int N) {
    int K = 50;
    float keep_prob = 50.0 / N;
    vector<float> work(N);
    vector<Edge> e;
    for (int i = 0; i < N; i++) {
        float *row = costs + i*N;
        copy(row, row+N, work.begin());
        nth_element(work.begin(), work.begin()+K, work.end());
        float threshold = work[K];
        for (int j = 0; j < N; j++)
            if (costs[i*N+j] <= threshold || u01(generator) < keep_prob)
                e.push_back(Edge{i, j, costs[i*N+j]*100000});
    }

    edmonds_karp(e, rs, cs, N);
}
