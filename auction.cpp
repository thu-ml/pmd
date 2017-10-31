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
using namespace std;

void auction_max(float *costs, int *rs, int *cs, int N) {
    printf("Starting auction\n");
    vector<int> icosts(N*N);
    vector<int> price(N), owner(N), bid_prices(N), bid_objects(N);
    vector<bool> is_assigned(N);
    vector<int> unassigned(N);

    fill(owner.begin(), owner.end(), -1);

    for (int i = 0; i < N*N; i++)
        icosts[i] = costs[i] * 100;

    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++)
    //         cout << icosts[i*N+j] << ' ';
    //     cout << endl;
    // }

    const int epsilon = 1;

    int num_iters = 0;
    for (; ; num_iters++) {
        // Check terminate
        fill(is_assigned.begin(), is_assigned.end(), 0);
        for (int i = 0; i < N; i++)
            if (owner[i] != -1)
                is_assigned[owner[i]] = true;

        int num_matches = 0;
        for (int i = 0; i < N; i++)
            if (is_assigned[i])
                num_matches++;

        if (num_matches == N)
            break;

        // Bidding phase
        if (N - num_matches > 500) {
            printf("%d iterations, %d unassigned\n", num_iters, N-num_matches);
            unassigned.clear();
            for (int i = 0; i < N; i++)
                if (!is_assigned[i])
                    unassigned.push_back(i);
            #pragma omp parallel for schedule(static, 10)
            for (size_t l = 0; l < unassigned.size(); l++) {
                int i = unassigned[l];
                // Find the maximum profit margin
                int best_profit = -1e9;
                int best_object = -1;
                for (int j = 0; j < N; j++) {
                    int profit = icosts[i*N+j] - price[j];
                    if (profit > best_profit) {
                        best_profit = profit;
                        best_object = j;
                    }
                }
                int second_best_profit = -1e9;
                for (int j = 0; j < N; j++) {
                    if (j == best_object) continue;
                    int profit = icosts[i*N+j] - price[j];
                    if (profit > second_best_profit)
                        second_best_profit = profit;
                }
                int my_price = price[best_object] + best_profit - second_best_profit + epsilon;
                bid_objects[i] = best_object;
                bid_prices[i] = my_price;
            }
        } else {
            for (int i = 0; i < N; i++) 
                if (!is_assigned[i]) {
                    // Find the maximum profit margin
                    int best_profit = -1e9;
                    int best_object = -1;
                    for (int j = 0; j < N; j++) {
                        int profit = icosts[i*N+j] - price[j];
                        if (profit > best_profit) {
                            best_profit = profit;
                            best_object = j;
                        }
                    }
                    int second_best_profit = -1e9;
                    for (int j = 0; j < N; j++) {
                        if (j == best_object) continue;
                        int profit = icosts[i*N+j] - price[j];
                        if (profit > second_best_profit)
                            second_best_profit = profit;
                    }
                    int my_price = price[best_object] + best_profit - second_best_profit + epsilon;
                    bid_objects[i] = best_object;
                    bid_prices[i] = my_price;
                }
        }

        // cout << "bid_objects " << bid_objects << endl;
        // cout << "bid_prices " << bid_prices << endl;

        // Assignment phase
        fill(price.begin(), price.end(), 0);
        fill(owner.begin(), owner.end(), -1);
        for (int i = 0; i < N; i++)
            if (price[bid_objects[i]] < bid_prices[i]) {
                price[bid_objects[i]] = bid_prices[i];
                owner[bid_objects[i]] = i;
            }
        // cout << "prices " << price << endl;
        // cout << "owner " << owner << endl;
    }
    for (int i = 0; i < N; i++) {
        rs[i] = owner[i];
        cs[i] = i;
    }
    printf("The auction is finished in %d iterations.\n", num_iters);
}

void auction_min(float *costs, int *rs, int *cs, int N) {
    float max_cost = -1e9;
    for (int i = 0; i < N*N; i++)
        max_cost = max(max_cost, costs[i]);
    vector<float> costs2(N*N);
    for (int i = 0; i < N*N; i++)
        costs2[i] = max_cost - costs[i];

    auction_max(costs2.data(), rs, cs, N);
}
