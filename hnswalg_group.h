#pragma once

#include "visited_list_pool.h"
#include <random>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unordered_map>
#include <vector>
#include <map>
#include <cmath>
#include <queue>
#include "utils.h"

namespace hnswlib {
    typedef uint32_t idx_t;

    struct GroupHNSW
    {
        size_t ngroups;
        size_t cur_element_count;
        size_t efConstruction;

        VisitedListPool *visitedlistpool;
        idx_t enterpoint_node = -1;

        size_t dist_calc = 0;
        size_t hops = 0;

        std::vector<float> centroids;
        std::vector<std::vector<float>> data;
        std::vector<std::vector<idx_t>> ids;
        std::vector<std::vector<idx_t>> links;

        size_t d;
        size_t M;
        size_t maxM;

        std::mutex cur_element_count_guard;
        std::mutex global;

    public:
        GroupHNSW(size_t dim, size_t ncentroids, size_t M, size_t maxM, size_t efConstruction = 500);
        ~GroupHNSW();

        std::priority_queue<std::pair<float, idx_t>> searchGroupLayer(idx_t target_node, size_t ef);
        void getNeighborsByHeuristic(std::priority_queue<std::pair<float, idx_t>> &topResults, size_t NN);
        void mutuallyConnectGroup(idx_t current_node, std::priority_queue<std::pair<float, idx_t>> &topResults);
        void addGroup(const std::vector<float> &group, const std::vector<idx_t> &idxs);
        std::priority_queue<std::pair<float, idx_t >> searchKnn(const float *query, size_t ef, size_t k);

        void read(const std::string &location);
        void write(const std::string &location);

    private:
        // TODO:
        float group2group_dist(idx_t group_id1, idx_t group_id2);
        float query2group_dist(const float *query, idx_t group_id, std::priority_queue<std::pair<float, idx_t>> &knn, size_t k);
    };
}
