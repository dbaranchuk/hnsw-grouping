#pragma once

#include "visited_list_pool.h"
#include <random>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <cmath>
#include <ctime>
#include <queue>
#include "utils.h"
#include <vector>

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

    struct Vertex{
        idx_t vertex_id;
        std::vector<idx_t> prev_vertex_ids;

        bool is_visited = false;
        size_t min_path_length = 0;

        void print(){
            std::cout << "Vertex " << vertex_id;
            std::cout << " | Path length: " << min_path_length;
            std::cout << " | In edges: " << prev_vertex_ids.size() << std::endl;
        }
    };

    struct HierarchicalNSW
    {
        size_t maxelements_;
        idx_t cur_element_count;
        size_t efConstruction_;

        VisitedListPool *visitedlistpool;
        int32_t enterpoint_node;

	    size_t limit;
        size_t dist_calc;
        size_t hops;

        char *data_level0_memory_;

        size_t d_;
        size_t data_size;
        size_t offset_data;
        size_t size_data_per_element;

        size_t M_;
        size_t maxM_;
        size_t max_level;
        size_t size_links_level0;
        size_t efSearch;

        char **linkLists;
        size_t size_links_per_element;
        std::vector<uint8_t> elementLevels;

        float mult;
        std::default_random_engine generator;

        size_t tmp_query_dist_calc = 0;

    public:
        HierarchicalNSW(const std::string &infoLocation, const std::string &dataLocation, const std::string &edgeLocation);
        HierarchicalNSW(size_t d, size_t maxelements, size_t M, size_t maxM,
                        size_t efConstruction = 500, bool is_one_layer=false);
        ~HierarchicalNSW();

        inline float *getDataByInternalId(idx_t internal_id) const {
            return (float *) (data_level0_memory_ + internal_id * size_data_per_element + offset_data);
        }

        inline uint16_t *get_linklist(idx_t internal_id) const {
            return (uint16_t *) (data_level0_memory_ + internal_id * size_data_per_element);
        }

        inline uint16_t *get_linklist_level(idx_t cur_c, size_t level) const
        {
            assert(level > 0);
            return (uint16_t *)(linkLists[cur_c] + (level - 1) * size_links_per_element);
        };

        std::priority_queue<std::pair<float, idx_t>> searchBaseLayer(const float *x, size_t ef, size_t level = 0);

        void getNeighborsByHeuristic(std::priority_queue<std::pair<float, idx_t>> &topResults, size_t NN);

        void mutuallyConnectNewElement(const float *x, idx_t id,
                std::priority_queue<std::pair<float, idx_t>> topResults, size_t level = 0);

        void setElementLevels(bool one_layer=true);
        void addPoint(const float *point);

        idx_t get_enterpoint(const float *query);
        std::priority_queue<std::pair<float, idx_t >> searchKnn(const float *query_data, size_t k);

        void SaveInfo(const std::string &location);
        void SaveEdges(const std::string &location);

        void LoadInfo(const std::string &location);
        void LoadData(const std::string &location);
        void LoadEdges(const std::string &location);

        // NSG
        HierarchicalNSW(size_t maxelements, size_t d,
                        const std::string &dataLocation, const std::string &edgeLocation, bool is_nsg=true);
        void LoadNSG(const std::string &dataLocation, const std::string &edgeLocation);
        void LoadMRNG(const std::string &dataLocation, const std::string &edgeLocation);

        // BFS
        std::vector<idx_t> bfs(idx_t initial_vertex_id, idx_t gt, size_t margin=0);
    };
}
