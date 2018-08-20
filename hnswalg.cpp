#include "hnswalg.h"

#define MAX_BITS 20

namespace hnswlib {

    HierarchicalNSW::HierarchicalNSW(const std::string &infoLocation,
                                     const std::string &dataLocation,
                                     const std::string &edgeLocation)
    {
        LoadInfo(infoLocation);
        LoadData(dataLocation);
        LoadEdges(edgeLocation);
    }

    HierarchicalNSW::HierarchicalNSW(size_t d, size_t maxelements, size_t M, size_t maxM, size_t efConstruction)
{
    d_ = d;
    data_size = d * sizeof(float);

    efConstruction_ = efConstruction;
    efSearch = efConstruction;

    maxelements_ = maxelements;
    M_ = M;
    maxM_ = maxM;
    size_links_level0 = maxM * sizeof(idx_t) + sizeof(uint8_t);
    size_data_per_element = size_links_level0 + data_size;
    offset_data = size_links_level0;

    data_level0_memory_ = (char *) malloc(maxelements_ * size_data_per_element);

    size_data_per_element_reordered = sizeof(idx_t) + data_size;
    data_level0_memory_reordered_ = (char *) malloc(maxelements_ * size_data_per_element_reordered);

    std::cout << "Size Mb: " << (maxelements_ * size_data_per_element) / (1000 * 1000) << std::endl;

    visitedlistpool = new VisitedListPool(1, maxelements_);

    enterpoint_node = 0;
    cur_element_count = 0;
}

HierarchicalNSW::~HierarchicalNSW()
{
    free(data_level0_memory_);
    free(data_level0_memory_reordered_);
    delete visitedlistpool;
}


std::priority_queue<std::pair<float, idx_t>> HierarchicalNSW::searchBaseLayer(const float *point, size_t ef)
{
    VisitedList *vl = visitedlistpool->getFreeVisitedList();
    vl_type *massVisited = vl->mass;
    vl_type currentV = vl->curV;
    std::priority_queue<std::pair<float, idx_t >> topResults;
    std::priority_queue<std::pair<float, idx_t >> candidateSet;

    float dist = fvec_L2sqr(point, getDataByInternalId(enterpoint_node), d_);
    dist_calc++;

    topResults.emplace(dist, enterpoint_node);
    candidateSet.emplace(-dist, enterpoint_node);
    massVisited[enterpoint_node] = currentV;
    float lowerBound = dist;

    while (!candidateSet.empty())
    {
        std::pair<float, idx_t> curr_el_pair = candidateSet.top();
        if (-curr_el_pair.first > lowerBound)
            break;

        candidateSet.pop();
        idx_t curNodeNum = curr_el_pair.second;

        uint8_t *ll_cur = get_linklist(curNodeNum);
        size_t size = *ll_cur;
        idx_t *data = (idx_t *)(ll_cur + 1);

        _mm_prefetch((char *) (massVisited + *data), _MM_HINT_T0);
        _mm_prefetch((char *) (massVisited + *data + 64), _MM_HINT_T0);
        _mm_prefetch(getDataByInternalId(*data), _MM_HINT_T0);

        for (size_t j = 0; j < size; ++j) {
            idx_t tnum = *(data + j);

            _mm_prefetch((char *) (massVisited + *(data + j + 1)), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*(data + j + 1)), _MM_HINT_T0);

            if (!(massVisited[tnum] == currentV)) {
                massVisited[tnum] = currentV;

                float dist = fvec_L2sqr(point, getDataByInternalId(tnum), d_);
                dist_calc++;

                if (topResults.top().first > dist || topResults.size() < ef) {
                    candidateSet.emplace(-dist, tnum);

                    _mm_prefetch(get_linklist(candidateSet.top().second), _MM_HINT_T0);
                    topResults.emplace(dist, tnum);

                    if (topResults.size() > ef)
                        topResults.pop();

                    lowerBound = topResults.top().first;
                }
            }
        }
    }
    visitedlistpool->releaseVisitedList(vl);
    return topResults;
}


    std::priority_queue<std::pair<float, idx_t>> HierarchicalNSW::searchFlipBits(const float *point, size_t ef)
    {
        VisitedList *vl = visitedlistpool->getFreeVisitedList();
        vl_type *massVisited = vl->mass;
        vl_type currentV = vl->curV;
        std::priority_queue<std::pair<float, idx_t >> topResults;
        std::priority_queue<std::pair<float, idx_t >> candidateSet;

        float dist = fvec_L2sqr(point, getDataByInternalIdReordered(enterpoint_node), d_);
        dist_calc++;

        topResults.emplace(dist, enterpoint_node);
        candidateSet.emplace(-dist, enterpoint_node);
        massVisited[enterpoint_node] = currentV;
        float lowerBound = dist;

        while (!candidateSet.empty())
        {
            std::pair<float, idx_t> curr_el_pair = candidateSet.top();
            if (-curr_el_pair.first > lowerBound)
                break;

            candidateSet.pop();
            idx_t curNodeNum = curr_el_pair.second;
            /////
            idx_t tnum = *get_edge(curNodeNum);
            if (!(massVisited[tnum] == currentV)) {
                massVisited[tnum] = currentV;

                float dist = fvec_L2sqr(point, getDataByInternalIdReordered(tnum), d_);
                dist_calc++;

                if (topResults.top().first > dist || topResults.size() < ef) {
                    candidateSet.emplace(-dist, tnum);

                    topResults.emplace(dist, tnum);

                    if (topResults.size() > ef)
                        topResults.pop();

                    lowerBound = topResults.top().first;
                }
            }
            ////

            for (size_t j = 0; j < 20; ++j) {
                idx_t tnum = curNodeNum^(1 << j);
                if (tnum >= maxelements_)
                    continue;

                if (!(massVisited[tnum] == currentV)) {
                    massVisited[tnum] = currentV;

                    float dist = fvec_L2sqr(point, getDataByInternalIdReordered(tnum), d_);
                    dist_calc++;

                    if (topResults.top().first > dist || topResults.size() < ef) {
                        candidateSet.emplace(-dist, tnum);

                        topResults.emplace(dist, tnum);

                        if (topResults.size() > ef)
                            topResults.pop();

                        lowerBound = topResults.top().first;
                    }
                }
            }
        }
        visitedlistpool->releaseVisitedList(vl);
        return topResults;
    }


void HierarchicalNSW::getNeighborsByHeuristic(std::priority_queue<std::pair<float, idx_t>> &topResults, size_t NN)
{
    if (topResults.size() < NN)
        return;

    std::priority_queue<std::pair<float, idx_t>> resultSet;
    std::vector<std::pair<float, idx_t>> returnlist;

    while (topResults.size() > 0) {
        resultSet.emplace(-topResults.top().first, topResults.top().second);
        topResults.pop();
    }

    while (resultSet.size()) {
        if (returnlist.size() >= NN)
            break;
        std::pair<float, idx_t> curen = resultSet.top();
        float dist_to_query = -curen.first;
        resultSet.pop();
        bool good = true;
        for (std::pair<float, idx_t> curen2 : returnlist) {
            float curdist = fvec_L2sqr(getDataByInternalId(curen2.second),
                                         getDataByInternalId(curen.second), d_);
            if (curdist < dist_to_query) {
                good = false;
                break;
            }
        }
        if (good) returnlist.push_back(curen);
    }
    for (std::pair<float, idx_t> elem : returnlist)
        topResults.emplace(-elem.first, elem.second);
}

void HierarchicalNSW::mutuallyConnectNewElement(const float *point, idx_t cur_c,
                               std::priority_queue<std::pair<float, idx_t>> topResults)
{
    getNeighborsByHeuristic(topResults, M_);

    std::vector<idx_t> res;
    res.reserve(M_);
    while (topResults.size() > 0) {
        res.push_back(topResults.top().second);
        topResults.pop();
    }
    {
        uint8_t *ll_cur = get_linklist(cur_c);
        if (*ll_cur)
            throw std::runtime_error("Should be blank");

        *ll_cur = res.size();

        idx_t *data = (idx_t *)(ll_cur + 1);
        for (size_t idx = 0; idx < res.size(); idx++) {
            if (data[idx])
                throw std::runtime_error("Should be blank");
            data[idx] = res[idx];
        }
    }
    for (size_t idx = 0; idx < res.size(); idx++) {
        if (res[idx] == cur_c)
            throw std::runtime_error("Connection to the same element");

        size_t resMmax = maxM_;
        uint8_t *ll_other = get_linklist(res[idx]);
        uint8_t sz_link_list_other = *ll_other;

        if (sz_link_list_other > resMmax || sz_link_list_other < 0)
            throw std::runtime_error("Bad sz_link_list_other");

        if (sz_link_list_other < resMmax) {
            idx_t *data = (idx_t *) (ll_other + 1);
            data[sz_link_list_other] = cur_c;
            *ll_other = sz_link_list_other + 1;
        } else {
            // finding the "weakest" element to replace it with the new one
            idx_t *data = (idx_t *) (ll_other + 1);
            float d_max = fvec_L2sqr(getDataByInternalId(cur_c), getDataByInternalId(res[idx]), d_);
            // Heuristic:
            std::priority_queue<std::pair<float, idx_t>> candidates;
            candidates.emplace(d_max, cur_c);

            for (size_t j = 0; j < sz_link_list_other; j++)
                candidates.emplace(fvec_L2sqr(getDataByInternalId(data[j]), getDataByInternalId(res[idx]), d_), data[j]);

            getNeighborsByHeuristic(candidates, resMmax);

            size_t indx = 0;
            while (candidates.size() > 0) {
                data[indx] = candidates.top().second;
                candidates.pop();
                indx++;
            }
            *ll_other = indx;
        }
    }
}

void HierarchicalNSW::addPoint(const float *point)
{
    if (cur_element_count >= maxelements_) {
        std::cout << "The number of elements exceeds the specified limit\n";
        throw std::runtime_error("The number of elements exceeds the specified limit");
    }
    idx_t cur_c = cur_element_count++;
    memset((char *) get_linklist(cur_c), 0, size_data_per_element);
    memcpy(getDataByInternalId(cur_c), point, data_size);

    // Do nothing for the first element
    if (cur_c != 0) {
        std::priority_queue <std::pair<float, idx_t>> topResults = searchBaseLayer(point, efConstruction_);
        mutuallyConnectNewElement(point, cur_c, topResults);
    }
};

std::priority_queue<std::pair<float, idx_t>> HierarchicalNSW::searchKnn(const float *query, size_t k)
{
    //auto topResults = searchBaseLayer(query, efSearch);
    auto topResults = searchFlipBits(query, efSearch);

    while (topResults.size() > k)
        topResults.pop();

    return topResults;
};

void HierarchicalNSW::SaveInfo(const std::string &location)
{
    std::cout << "Saving info to " << location << std::endl;
    std::ofstream output(location, std::ios::binary);

    writeBinaryPOD(output, maxelements_);
    writeBinaryPOD(output, enterpoint_node);
    writeBinaryPOD(output, data_size);
    writeBinaryPOD(output, offset_data);
    writeBinaryPOD(output, size_data_per_element);
    writeBinaryPOD(output, M_);
    writeBinaryPOD(output, maxM_);
    writeBinaryPOD(output, size_links_level0);
}


void HierarchicalNSW::SaveEdges(const std::string &location)
{
    std::cout << "Saving edges to " << location << std::endl;
    std::ofstream output(location, std::ios::binary);

    for (size_t i = 0; i < maxelements_; i++) {
        uint8_t *ll_cur = get_linklist(i);
        uint32_t size = *ll_cur;

        output.write((char *) &size, sizeof(uint32_t));
        idx_t *data = (idx_t *)(ll_cur + 1);
        output.write((char *) data, sizeof(idx_t) * size);
    }
}

void HierarchicalNSW::LoadInfo(const std::string &location)
{
    std::cout << "Loading info from " << location << std::endl;
    std::ifstream input(location, std::ios::binary);

    readBinaryPOD(input, maxelements_);
    readBinaryPOD(input, enterpoint_node);
    readBinaryPOD(input, data_size);
    readBinaryPOD(input, offset_data);
    readBinaryPOD(input, size_data_per_element);
    readBinaryPOD(input, M_);
    readBinaryPOD(input, maxM_);
    readBinaryPOD(input, size_links_level0);

    d_ = data_size / sizeof(float);
    data_level0_memory_ = (char *) malloc(maxelements_ * size_data_per_element);
    size_data_per_element_reordered = sizeof(idx_t) + data_size;
    data_level0_memory_reordered_ = (char *) malloc(maxelements_ * size_data_per_element_reordered);
    //data_level0_memory_reordered_ = (char *) malloc(maxelements_ * data_size);

    efConstruction_ = 0;
    cur_element_count = maxelements_;

    visitedlistpool = new VisitedListPool(1, maxelements_);
}

void HierarchicalNSW::LoadData(const std::string &location)
{
    std::cout << "Loading data from " << location << std::endl;
    std::ifstream input(location, std::ios::binary);

    uint32_t dim;
    float mass[d_];
    for (size_t i = 0; i < maxelements_; i++) {
        input.read((char *) &dim, sizeof(uint32_t));
        if (dim != d_) {
            std::cout << "Wront data dim" << std::endl;
            exit(1);
        }
        input.read((char *) mass, dim * sizeof(float));
        memcpy(getDataByInternalId(i), mass, data_size);
    }
}

void HierarchicalNSW::LoadEdges(const std::string &location)
{
    std::cout << "Loading edges from " << location << std::endl;
    std::ifstream input(location, std::ios::binary);

    uint32_t size;

    for (size_t i = 0; i < maxelements_; i++) {
        input.read((char *) &size, sizeof(uint32_t));

        uint8_t *ll_cur = get_linklist(i);
        *ll_cur = size;
        idx_t *data = (idx_t *)(ll_cur + 1);

        input.read((char *) data, size * sizeof(idx_t));
    }
}


    void HierarchicalNSW::collect_edges() {
        negative_edges.resize(maxelements_);
        //Collect negative edges
        for (size_t i = 0; i < maxelements_; i++) {
            uint8_t *ll_cur = get_linklist(i);
            idx_t *data = (idx_t *)(ll_cur + 1);
            size_t size = *ll_cur;
            negative_edges[i].resize(size);
            for (size_t j = 0; j < size; j++)
                negative_edges[i][j] = data[j];
        }
        //Collect positive edges
        edges.resize(maxelements_);
        for (size_t i = 0; i < maxelements_; i++){
            for (idx_t negative_edge : negative_edges[i])
                edges[i].push_back(negative_edge);

//            for (idx_t negative_edge : negative_edges[i]){
//                bool inList = false;
//                for (idx_t edge : edges[negative_edge]) {
//                    inList = i == edge;
//                    if (inList) break;
//                }
//
//                if (!inList)
//                    edges[negative_edge].push_back(i);
//            }
        }
    }

    idx_t HierarchicalNSW::resolve_collision_type1_2(idx_t prev_new_edge, idx_t new_idx,
                                                    std::vector<bool> &assigned_new_edges)
    {
        //return prev_new_edge;
        idx_t new_edge = prev_new_edge;
        size_t prev_collisions = 0, new_collisions = 0;

        for (idx_t edge : new_edges[prev_new_edge])
            prev_collisions += hamming(prev_new_edge, edge) > 1;

        new_collisions = prev_collisions;

        for (size_t i = 0; i < MAX_BITS; i++) {
            idx_t tmp_new_edge = prev_new_edge ^ (1 << i);

            if (tmp_new_edge >= maxelements_||
                assigned_new_edges[tmp_new_edge] ||
                hamming(tmp_new_edge, new_idx) > 1)
                continue;

            size_t tmp_collisions = 0;

            for (idx_t edge : new_edges[prev_new_edge])
                tmp_collisions += hamming(tmp_new_edge, edge) > 1;

            if (tmp_collisions <= new_collisions) {
                new_edge = tmp_new_edge;
                new_collisions = tmp_collisions;
            }
        }
        if (new_edge != prev_new_edge){
            for (idx_t edge : new_edges[prev_new_edge])
                new_edges[new_edge].push_back(edge);
            new_edges[prev_new_edge].clear();
        }
//        if ((new_edge != prev_new_edge) && (prev_collisions || new_collisions))
//            std::cout << "List collisions: " << prev_collisions << " --> "
//                      << new_collisions << std::endl;
        return new_edge;
    }

    idx_t HierarchicalNSW::resolve_collision_type1_1(idx_t prev_new_edge, idx_t new_idx,
                                                     std::vector<bool> &assigned_new_edges)
    {
        for (size_t i = 0; i < MAX_BITS; i++)
        {
            idx_t tmp_new_edge = prev_new_edge ^ (1 << i);

            if (tmp_new_edge >= maxelements_ ||
                assigned_new_edges[tmp_new_edge])
                continue;

            if (hamming(tmp_new_edge, new_idx) == 1){
                return tmp_new_edge;
            }
        }
        return prev_new_edge;
    }

    void HierarchicalNSW::SaveReorderTable(const std::string &location)
    {
        std::cout << "Saving reordered edges to " << location << std::endl;
        std::ofstream output(location, std::ios::binary);

        for (idx_t i = 0; i < maxelements_; i++) {
            idx_t new_i = table[i];
            output.write((char *) &new_i, sizeof(idx_t));
        }
    }

    void HierarchicalNSW::LoadReorderTable(const std::string &location)
    {
        std::cout << "Loading reordered edges from " << location << std::endl;
        std::ifstream input(location, std::ios::binary);

        for (idx_t i = 0; i < maxelements_; i++) {
            idx_t new_i;
            input.read((char *) &new_i, sizeof(idx_t));
            table[i] = new_i;
            reverse_table[new_i] = i;
        }
    }

    void HierarchicalNSW::reorder()
    {
        collect_edges();
        std::vector<bool> assigned_edges(maxelements_);
        std::fill(assigned_edges.begin(), assigned_edges.end(), 0);
        assigned_edges[0] = true;

        std::vector<bool> assigned_new_edges(maxelements_);
        std::fill(assigned_new_edges.begin(), assigned_new_edges.end(), 0);
        assigned_new_edges[0] = true;

        std::vector<bool> bool_table(maxelements_);
        std::fill(bool_table.begin(), bool_table.end(), 0);

        std::vector<bool> reassign_edges(maxelements_);
        std::fill(reassign_edges.begin(), reassign_edges.end(), 0);

        std::vector<bool> filled_nodes(maxelements_);
        std::fill(filled_nodes.begin(), filled_nodes.end(), 0);

        table.insert({0, 0});

        std::queue<idx_t> queue;
        queue.push(0);

        new_edges.resize(maxelements_);

        //Collisions
        size_t ncollision_type_1 = 0;
        size_t ncollision_type_2 = 0;

        size_t counter = 0;

        while (!queue.empty()){
            idx_t old_idx = queue.front();
            queue.pop();
            idx_t new_idx = table[old_idx];

            if (counter++ % 100000 == 0) {
                std::cout << counter / 10000 << "%\t Collisions type 1: " << ncollision_type_1
                          << "\t Collisions type 2: " << ncollision_type_2 << std::endl;
            }
            if (bool_table[old_idx]){
                std::cout << "Repeated node: " << old_idx
                          << " Counter: " << counter <<  std::endl;
                exit(1);
            }
            bool_table[old_idx] = true;

            size_t shift = 0;
            for (idx_t edge : edges[old_idx]){
                if (assigned_edges[edge]){
                    idx_t prev_new_edge = table[edge];
                    if (!filled_nodes[new_idx] && hamming(new_idx, prev_new_edge) > 1){
                        idx_t *stored_edge = get_edge(new_idx);
                        *stored_edge = prev_new_edge;

                        filled_nodes[new_idx] = true;
                    } else {
                        new_edges[new_idx].push_back(prev_new_edge);
                        ncollision_type_1 += hamming(new_idx, prev_new_edge) > 1;
                    }
//                    if (!reassign_edges[edge] &&
//                        hamming(new_idx, prev_new_edge) > 1)
//                    {
//                        idx_t new_edge;
//                        if (!bool_table[edge])
//                            new_edge = resolve_collision_type1_1(prev_new_edge, new_idx, assigned_new_edges);
//                        else
//                            new_edge = resolve_collision_type1_2(prev_new_edge, new_idx, assigned_new_edges);
//
//                        new_edges[new_idx].push_back(new_edge);
//                        if (new_edge != prev_new_edge) {
//                            table[edge] = new_edge;
//                            assigned_new_edges[prev_new_edge] = false;
//                            assigned_new_edges[new_edge] = true;
//                        } else {
//                            ncollision_type_1++;
//                        }
//                    } else {
//                        new_edges[new_idx].push_back(prev_new_edge);
//                        ncollision_type_1 += hamming(new_idx, prev_new_edge) > 1;
//                    }
//                    reassign_edges[edge] = true;
                    continue;
                }

                idx_t new_edge = new_idx ^ (1 << shift);
                while (new_edge < maxelements_ && assigned_new_edges[new_edge])
                    new_edge = new_idx ^ (1 << ++shift);

                if (new_edge >= maxelements_){
                    ncollision_type_2++;

                    // Heuristic for Collisions Type 2
                    uint8_t min_hamming_dist = 32;
                    idx_t min_i = 0;
                    for (size_t i = 0; i < maxelements_; i++){
                        if (!assigned_new_edges[i]){
                            uint8_t hamming_dist = hamming(new_idx, i);
                            if (hamming_dist < min_hamming_dist) {
                                min_hamming_dist = hamming_dist;
                                min_i = i;
                            }
                            if (min_hamming_dist == 2)
                                break;
                        }
                    }
                    new_edge = min_i;
                }
                new_edges[new_idx].push_back(new_edge);
                assigned_new_edges[new_edge] = true;

                table[edge] = new_edge;
                assigned_edges[edge] = true;
                queue.push(edge);
            }

            // if queue become empty ahead of time
            if (queue.empty() and counter < maxelements_) {
                idx_t next_idx, next_new_idx;
                for (size_t i = 0; i < maxelements_; i++) {
                    if (!assigned_edges[i]) {
                        next_idx = i;
                        break;
                    }
                }
                for (size_t i = 0; i < maxelements_; i++){
                    if (!assigned_new_edges[i]) {
                        next_new_idx = i;
                        break;
                    }
                }
                assigned_edges[next_idx] = true;
                assigned_new_edges[next_new_idx] = true;
                table[next_idx] = next_new_idx;
                queue.push(next_idx);
            }
        }

        for (size_t i = 0; i < maxelements_; i++){
            if (!filled_nodes[i]){
                memset((char *) get_edge(i), 0, sizeof(idx_t));
                filled_nodes[i] = true;
            }
        }

        std::cout << "100%\t Collisions type 1: " << ncollision_type_1
                  << "\t Collisions type 2: " << ncollision_type_2 << std::endl;
    }

    // Reorder data
    void HierarchicalNSW::reorder_data()
    {
        for (size_t i = 0; i < maxelements_; i++){
            const float *src_point = getDataByInternalId(i);
            size_t new_i = table[i];
            reverse_table[new_i] = i;
            memcpy(getDataByInternalIdReordered(new_i), src_point, data_size);
        }
    }


    void HierarchicalNSW::compute_collisions() {
        size_t edge_collsions=0, final_edge_collisions=0;

        // Get final edges
        final_edges.resize(maxelements_);

        for (size_t i = 0; i < maxelements_; i++){
            idx_t idx = table[i];

            for (idx_t edge : negative_edges[i]) {
                edge_collsions += hamming(i, edge) > 1 ;
                final_edge_collisions += hamming(idx, table[edge]) > 1;
                final_edges[idx].push_back(table[edge]);
            }
        }
        std::cout << "Original collisions: " << edge_collsions
                  << "\t Final collisions: " << final_edge_collisions << std::endl;
    }
}