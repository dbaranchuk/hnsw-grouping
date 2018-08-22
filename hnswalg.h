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

        size_t d_;
        size_t data_size;
        size_t offset_data;
        size_t size_data_per_element;

        size_t M_;
        size_t maxM_;
        size_t size_links_level0;
        size_t efSearch;

        std::mutex cur_element_count_guard;
        std::mutex global;
        int maxLevel = -1;

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

        std::priority_queue<std::pair<float, idx_t>> searchBaseLayer(const float *x, size_t ef);

        void getNeighborsByHeuristic(std::priority_queue<std::pair<float, idx_t>> &topResults, size_t NN);

        void mutuallyConnectNewElement(const float *x, idx_t id, std::priority_queue<std::pair<float, idx_t>> topResults);

        void addPoint(const float *point);

        std::priority_queue<std::pair<float, idx_t >> searchKnn(const float *query_data, size_t k);

        void SaveInfo(const std::string &location);
        void SaveEdges(const std::string &location);

        void LoadInfo(const std::string &location);
        void LoadData(const std::string &location);
        void LoadEdges(const std::string &location);
    };
}
