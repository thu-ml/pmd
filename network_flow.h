#include <iostream>
#include <vector>

struct Edge {
    int u, v, w;
};

struct NetEdge {
    int v;      // Destination
    int w;      // Weight
    int r;      // Remaining
    int o;      // Opposite
};

typedef std::vector<std::vector<NetEdge>> Network;

void build_graph(std::vector<Edge> &b, Network &g, int N, int &V, int &S, int &T);

void fold_fulkerson(std::vector<Edge> &b, int *rs, int *cs, int N);

void edmonds_karp(std::vector<Edge> &b, int *rs, int *cs, int N);
