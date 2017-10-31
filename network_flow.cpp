#include <queue>
#include <cassert>
#include "network_flow.h"
using namespace std;

void build_graph(std::vector<Edge> &b, Network &g, int N, int &V, int &S, int &T) {
    V = 2*N + 2;
    S = 2*N;
    T = 2*N + 1;

    g.resize(V);
    // Add source and target
    for (int i = 0; i < N; i++) {
        g[S].push_back(NetEdge{i, 0, 1, 0});
        g[i].push_back(NetEdge{S, 0, 0, i});

        g[T].push_back(NetEdge{i+N, 0, 0, 1});
        g[i+N].push_back(NetEdge{T, 0, 1, i});
    }

    // Add normal edges
    for (auto &e: b) {
        int uid = g[e.u].size();
        int vid = g[e.v+N].size();
        g[e.u].push_back(NetEdge{e.v+N, e.w, 1, vid});
        g[e.v+N].push_back(NetEdge{e.u, -e.w, 0, uid});
    }
}

void fold_fulkerson(std::vector<Edge> &b, int *rs, int *cs, int N) {
    Network g;
    int V, S, T;
    build_graph(b, g, N, V, S, T);

    std::vector<int> prev_u(V), prev_j(V), dist(V);
    std::vector<bool> in_queue(V);
    std::queue<int> q;
    auto enqueue = [&](int x) { 
        if (!in_queue[x]) {
            q.push(x);
            in_queue[x] = true;
        }
    };
    auto dequeue = [&]() {
        int x = q.front();
        q.pop();
        in_queue[x] = false;
        return x;
    };

    for (int iter = 0; iter < N; iter++) {
        // Find minimum cost augmenting path
        fill(dist.begin(), dist.end(), 1e9);

        dist[S] = 0;
        enqueue(S);
        while (!q.empty()) {
            int u = dequeue();
            auto &eu = g[u];
            for (int j = 0; j < eu.size(); j++) {
                auto &e = eu[j];
                if (e.r > 0 && dist[e.v] > dist[u]+e.w) {
                    dist[e.v] = dist[u] + e.w;
                    prev_u[e.v] = u;
                    prev_j[e.v] = j;
                    enqueue(e.v);
                }
            }
        }

        if (dist[T] >= 1e9) {
            printf("Warning: perfect matching does not exist.\n");
            break;
        } else {
            // Augment
            // printf("Iteration %d, cost = %d\n", iter, dist[T]);
            int current_v = T;
            while (current_v != S) {
                int u = prev_u[current_v];
                int j = prev_j[current_v];
                auto &e = g[u][j];
                auto &re = g[current_v][e.o];
                e.r--; re.r++;
                // printf("Augment %d -- %d\n", u, current_v);
                current_v = u; 
            }
        }
    }
    // Prepare results
    std::vector<int> matched_u(N, -1), matched_v(N, -1);
    for (int i = 0; i < N; i++) {
        for (auto &e: g[i])
            if (e.v != S && e.r == 0) {
                int v = e.v - N;
                matched_u[i] = v;
                matched_v[v] = i;
            }
    }
    std::vector<int> unmatched_u, unmatched_v;
    for (int i = 0; i < N; i++) if (matched_u[i] == -1) unmatched_u.push_back(i);
    for (int i = 0; i < N; i++) if (matched_v[i] == -1) unmatched_v.push_back(i);
    assert(unmatched_u.size() == unmatched_v.size());
    for (int i = 0 ; i < unmatched_u.size(); i++)
        matched_u[unmatched_u[i]] = unmatched_v[i];

    for (int i = 0; i < N; i++) {
        rs[i] = i;
        cs[i] = matched_u[i];
    }
}

void edmonds_karp(std::vector<Edge> &b, int *rs, int *cs, int N) {
    Network g;
    int V, S, T;
    build_graph(b, g, N, V, S, T);

    std::vector<int> prev_u(V), prev_j(V), dist(V), pi(V);
    std::vector<bool> visited(V);
    std::priority_queue<pair<int, int>> q;

    for (int iter = 0; iter < N; iter++) {
        // Find minimum cost augmenting path
        fill(dist.begin(), dist.end(), 1e9);
        fill(visited.begin(), visited.end(), false);

        dist[S] = 0; q.push(make_pair(0, S));
        while (!q.empty()) {
            int u = q.top().second;
            q.pop();
            if (visited[u]) continue;
            visited[u] = true;

            auto &eu = g[u];
            for (int j = 0; j < eu.size(); j++) {
                auto &e = eu[j];
                auto w = e.w + pi[u] - pi[e.v];
                if (e.r > 0 && dist[e.v] > dist[u] + w) {
                    //assert(w >= 0); //TODO
                    dist[e.v] = dist[u] + w;
                    prev_u[e.v] = u;
                    prev_j[e.v] = j;
                    q.push(make_pair(-dist[e.v], e.v));
                }
            }
        }

        if (dist[T] >= 1e9) {
            printf("Warning: perfect matching does not exist.\n");
            break;
        } else {
            // Augment
            //printf("Iteration %d, cost = %d\n", iter, dist[T]);
            int current_v = T;
            while (current_v != S) {
                int u = prev_u[current_v];
                int j = prev_j[current_v];
                auto &e = g[u][j];
                auto &re = g[current_v][e.o];
                e.r--; re.r++;
                //printf("Augment %d -- %d\n", u, current_v);
                current_v = u; 
            }
            // Update pi
            for (int v = 0; v < V; v++)
                pi[v] += dist[v];
        }
    }
    // Prepare results
    std::vector<int> matched_u(N, -1), matched_v(N, -1);
    for (int i = 0; i < N; i++) {
        for (auto &e: g[i])
            if (e.v != S && e.r == 0) {
                int v = e.v - N;
                matched_u[i] = v;
                matched_v[v] = i;
            }
    }
    std::vector<int> unmatched_u, unmatched_v;
    for (int i = 0; i < N; i++) if (matched_u[i] == -1) unmatched_u.push_back(i);
    for (int i = 0; i < N; i++) if (matched_v[i] == -1) unmatched_v.push_back(i);
    assert(unmatched_u.size() == unmatched_v.size());
    for (int i = 0 ; i < unmatched_u.size(); i++)
        matched_u[unmatched_u[i]] = unmatched_v[i];

    for (int i = 0; i < N; i++) {
        rs[i] = i;
        cs[i] = matched_u[i];
    }
}
