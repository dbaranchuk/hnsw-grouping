#pragma once

#include "visited_list_pool.h"
#include <random>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <cmath>
#include <queue>
#include "utils.h"

template<typename T>
static void writeBinaryPOD(std::ostream &out, const T &podRef) {
    out.write((char *) &podRef, sizeof(T));
}

template<typename T>
static void readBinaryPOD(std::istream &in, T &podRef) {
    in.read((char *) &podRef, sizeof(T));
}

namespace hnswlib {
    typedef uint32_t idx_t;

    struct HierarchicalNSW
    {
        size_t maxelements_;
        size_t cur_element_count;
        size_t efConstruction_;

        VisitedListPool *visitedlistpool;
        idx_t enterpoint_node;

        size_t dist_calc;
        size_t hops;

        char *data_level0_memory_;
        char *data_level0_memory_reordered_;

        size_t d_;
        size_t data_size;
        size_t offset_data;
        size_t size_data_per_element;
        size_t size_data_per_element_reordered;

        size_t M_;
        size_t maxM_;
        size_t size_links_level0;
        size_t efSearch;

    public:
        HierarchicalNSW(const std::string &infoLocation, const std::string &dataLocation, const std::string &edgeLocation);
        HierarchicalNSW(size_t d, size_t maxelements, size_t M, size_t maxM, size_t efConstruction = 500);
        ~HierarchicalNSW();

        inline float *getDataByInternalId(idx_t internal_id) const {
            return (float *) (data_level0_memory_ + internal_id * size_data_per_element + offset_data);
        }

        inline uint8_t *get_linklist(idx_t internal_id) const {
            return (uint8_t *) (data_level0_memory_ + internal_id * size_data_per_element);
        }

        inline idx_t *get_edge(idx_t internal_id) const {
            return (idx_t *) (data_level0_memory_reordered_ + internal_id * size_data_per_element_reordered);
        }

        inline float *getDataByInternalIdReordered(idx_t internal_id) const {
            return (float *) (data_level0_memory_reordered_ + internal_id * size_data_per_element_reordered + sizeof(idx_t));
        }

        std::priority_queue<std::pair<float, idx_t>> searchBaseLayer(const float *x, size_t ef);
        std::priority_queue<std::pair<float, idx_t>> searchFlipBits(const float *x, size_t ef);

        void getNeighborsByHeuristic(std::priority_queue<std::pair<float, idx_t>> &topResults, size_t NN);

        void mutuallyConnectNewElement(const float *x, idx_t id, std::priority_queue<std::pair<float, idx_t>> topResults);

        void addPoint(const float *point);

        std::priority_queue<std::pair<float, idx_t >> searchKnn(const float *query_data, size_t k);

        void SaveInfo(const std::string &location);
        void SaveEdges(const std::string &location);
        void SaveReorderTable(const std::string &location);

        void LoadInfo(const std::string &location);
        void LoadData(const std::string &location);
        void LoadEdges(const std::string &location);
        void LoadReorderTable(const std::string &location);

        std::vector<std::vector<idx_t>> edges;
        std::vector<std::vector<idx_t>> new_edges;
        std::vector<std::vector<idx_t>> final_edges;
        std::vector<std::vector<idx_t>> negative_edges;

        std::unordered_map<idx_t, idx_t> table;
        std::unordered_map<idx_t, idx_t> reverse_table;

        idx_t resolve_collision_type1_1(idx_t prev_new_edge, idx_t new_idx, std::vector<bool> &assigned_new_edges);
        idx_t resolve_collision_type1_2(idx_t prev_new_edge, idx_t new_idx, std::vector<bool> &assigned_new_edges);

        void compute_collisions();
        void collect_edges();
        void reorder();
        void reorder_data();

        static uint8_t hamming (idx_t a, idx_t b){
            uint8_t dist = 0; // count accumulates the total bits set
            idx_t n = a^b;
            while(n != 0){
                n &= (n-1); // clear the least significant bit set
                dist++;
            }
            return dist;
        }
    };
}
