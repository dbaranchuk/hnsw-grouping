#include "hnswalg.h"

namespace hnswlib {

    HierarchicalNSW::HierarchicalNSW(const std::string &infoLocation,
                                     const std::string &dataLocation,
                                     const std::string &edgeLocation) {
        LoadInfo(infoLocation);
        LoadData(dataLocation);
        LoadEdges(edgeLocation);
    }

    HierarchicalNSW::HierarchicalNSW(size_t d, size_t maxelements, size_t M, size_t maxM, size_t efConstruction) {
        d_ = d;
        data_size = d * sizeof(float);

        efConstruction_ = efConstruction > M ? efConstruction : M;
        efSearch = efConstruction;

        maxelements_ = maxelements;
        M_ = M;
        maxM_ = maxM;
        size_links_level0 = maxM * sizeof(idx_t) + sizeof(uint16_t);
        size_data_per_element = size_links_level0 + data_size;
        offset_data = size_links_level0;

        data_level0_memory_ = (char *) malloc(maxelements_ * size_data_per_element);
        std::cout << "Size Mb: " << (maxelements_ * size_data_per_element) / (1000 * 1000) << std::endl;
        size_links_per_element = M * sizeof(idx_t) + sizeof(uint16_t);

        elementLevels = std::vector<uint8_t>(maxelements_);

        linkLists = (char **) malloc(sizeof(void *) * maxelements_);
        mult = 1.0 / log(1.0 * M);

        visitedlistpool = new VisitedListPool(1, maxelements_);

        max_level = 0;
        enterpoint_node = -1; //0;
        cur_element_count = 0;

        generator = std::default_random_engine(100);
        setElementLevels(false);
    }

    HierarchicalNSW::~HierarchicalNSW() {
        free(data_level0_memory_);
        delete visitedlistpool;

        for (idx_t i = 0; i < cur_element_count; i++) {
            if (elementLevels[i] > 0)
                free(linkLists[i]);
        }
        free(linkLists);
    }


    std::priority_queue<std::pair<float, idx_t>>
    HierarchicalNSW::searchBaseLayer(const float *point, size_t ef, size_t level) {
        VisitedList *vl = visitedlistpool->getFreeVisitedList();
        vl_type *massVisited = vl->mass;
        vl_type currentV = vl->curV;
        std::priority_queue<std::pair<float, idx_t >> topResults;
        std::priority_queue<std::pair<float, idx_t >> candidateSet;

        float dist = fvec_L2sqr(point, getDataByInternalId(enterpoint_node), d_);
        size_t query_dist_calc = 1;

        topResults.emplace(dist, enterpoint_node);
        candidateSet.emplace(-dist, enterpoint_node);
        massVisited[enterpoint_node] = currentV;
        float lowerBound = dist;

        while (!candidateSet.empty()) {
            std::pair<float, idx_t> curr_el_pair = candidateSet.top();
            if (-curr_el_pair.first > lowerBound)
                break;

            candidateSet.pop();
            idx_t curNodeNum = curr_el_pair.second;

            uint16_t *ll_cur = level ? get_linklist_level(curNodeNum, level) : get_linklist(curNodeNum);

            size_t size = *ll_cur;
            auto *data = (idx_t *) (ll_cur + 1);

            _mm_prefetch((char *) (massVisited + *data), _MM_HINT_T0);
            _mm_prefetch((char *) (massVisited + *data + 64), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*data), _MM_HINT_T0);

            for (size_t j = 0; j < size; ++j) {
                idx_t tnum = *(data + j);

                _mm_prefetch((char *) (massVisited + *(data + j + 1)), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(data + j + 1)), _MM_HINT_T0);

                if (massVisited[tnum] != currentV) {
                    massVisited[tnum] = currentV;

                    if (limit > 0 && query_dist_calc == limit) {
                        dist_calc += query_dist_calc;
                        visitedlistpool->releaseVisitedList(vl);
                        return topResults;
                    }

                    float dist = fvec_L2sqr(point, getDataByInternalId(tnum), d_);
                    query_dist_calc++;

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
            hops++;
        }
        dist_calc += query_dist_calc;
        visitedlistpool->releaseVisitedList(vl);
        return topResults;
    }


    void HierarchicalNSW::getNeighborsByHeuristic(std::priority_queue<std::pair<float, idx_t>> &topResults, size_t NN) {
        if (topResults.size() < NN)
            return;

        std::priority_queue<std::pair<float, idx_t>> resultSet;
        std::priority_queue<std::pair<float, idx_t>> templist;
        std::vector<std::pair<float, idx_t>> returnlist;

        while (!topResults.empty()) {
            resultSet.emplace(-topResults.top().first, topResults.top().second);
            topResults.pop();
        }

        while (!resultSet.empty()) {
            if (returnlist.size() >= NN)
                break;
            std::pair<float, idx_t> curen = resultSet.top();
            float dist_to_query = -curen.first;
            resultSet.pop();
            bool good = true;
            for (std::pair<float, idx_t> curen2 : returnlist) {
                float curdist = fvec_L2sqr(getDataByInternalId(curen2.second), getDataByInternalId(curen.second), d_);
                if (curdist < dist_to_query) {
                    good = false;
                    break;
                }
            }
            if (good)
                returnlist.push_back(curen);
//        else
//            templist.push(curen);
        }
//    while (returnlist.size() < NN && templist.size() > 0) {
//        returnlist.push_back(templist.top());
//        templist.pop();
//    }
        for (std::pair<float, idx_t> elem : returnlist)
            topResults.emplace(-elem.first, elem.second);
    }


    void HierarchicalNSW::mutuallyConnectNewElement(const float *point, idx_t cur_c,
                                                    std::priority_queue<std::pair<float, idx_t>> topResults,
                                                    size_t level) {
        getNeighborsByHeuristic(topResults, M_);

        std::vector<idx_t> res;
        res.reserve(M_);
        while (!topResults.empty()) {
            res.push_back(topResults.top().second);
            topResults.pop();
        }
        {
            uint16_t *ll_cur = level ? get_linklist_level(cur_c, level) : get_linklist(cur_c);
            if (*ll_cur)
                throw std::runtime_error("Should be blank");

            *ll_cur = (uint16_t) res.size();
            auto *data = (idx_t *) (ll_cur + 1);
            for (size_t idx = 0; idx < res.size(); idx++) {
                if (data[idx])
                    throw std::runtime_error("Should be blank");
                if (level > elementLevels[res[idx]])
                    throw std::runtime_error("Bad level");
                data[idx] = res[idx];
            }
        }
        for (idx_t idx : res) {
            if (idx == cur_c)
                throw std::runtime_error("Connection to the same element");

            size_t resMmax = level ? M_ : maxM_;
            uint16_t *ll_other = level ? get_linklist_level(idx, level) : get_linklist(idx);
            size_t sz_link_list_other = *ll_other;
            if (sz_link_list_other > resMmax || sz_link_list_other < 0)
                throw std::runtime_error("Bad sz_link_list_other");
            if (sz_link_list_other < resMmax) {
                auto *data = (idx_t *) (ll_other + 1);
                data[sz_link_list_other] = cur_c;
                *ll_other = (uint16_t) (sz_link_list_other + 1);
            } else {
                // finding the "weakest" element to replace it with the new one
                auto *data = (idx_t *) (ll_other + 1);
                float d_max = fvec_L2sqr(getDataByInternalId(cur_c), getDataByInternalId(idx), d_);
                // Heuristic:
                std::priority_queue<std::pair<float, idx_t>> candidates;
                candidates.emplace(d_max, cur_c);

                for (size_t j = 0; j < sz_link_list_other; j++)
                    candidates.emplace(fvec_L2sqr(getDataByInternalId(data[j]), getDataByInternalId(idx), d_),
                                       data[j]);

                getNeighborsByHeuristic(candidates, resMmax);

                uint16_t indx = 0;
                while (!candidates.empty()) {
                    data[indx] = candidates.top().second;
                    candidates.pop();
                    indx++;
                }
                *ll_other = indx;
            }
        }
    }

    void HierarchicalNSW::setElementLevels(bool one_layer) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        std::cout << -log(distribution(generator)) << " " << (double) mult << std::endl;
        for (size_t i = 0; i < maxelements_; ++i) {
            elementLevels[i] = !one_layer ? (uint8_t) (-log(distribution(generator)) * mult) : (uint8_t) (0);
            //elementLevels[i] = (elementLevels[i] > 4) ? (uint8_t) 4 : elementLevels[i];
        }
    }


    void HierarchicalNSW::addPoint(const float *point) {
        if (cur_element_count >= maxelements_) {
            std::cout << "The number of elements exceeds the specified limit\n";
            throw std::runtime_error("The number of elements exceeds the specified limit");
        }
        size_t curlevel = elementLevels[cur_element_count];
        size_t maxlevelcopy = max_level;

        memset((char *) get_linklist(cur_element_count), 0, size_data_per_element);
        memcpy(getDataByInternalId(cur_element_count), point, data_size);

        if (curlevel) {
            // Above levels contain only clusters
            linkLists[cur_element_count] = (char *) malloc(size_links_per_element * curlevel);
            memset(linkLists[cur_element_count], 0, size_links_per_element * curlevel);
        }

        // Do nothing for the first element
        int32_t currObj = enterpoint_node;
        if (currObj != -1) {
            if (curlevel < maxlevelcopy) {
                float curdist = fvec_L2sqr(point, getDataByInternalId(currObj), d_);
                for (size_t level = maxlevelcopy; level > curlevel; level--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        uint16_t *data = level ? get_linklist_level(currObj, level) : get_linklist(currObj);
                        size_t size = *data;
                        auto *datal = (idx_t *) (data + 1);
                        for (uint8_t i = 0; i < size; i++) {
                            idx_t candidate = datal[i];
                            if (candidate < 0 || candidate > maxelements_)
                                throw std::runtime_error("candidate error");
                            float d = fvec_L2sqr(point, getDataByInternalId(candidate), d_);
                            if (d < curdist) {
                                curdist = d;
                                currObj = candidate;
                                changed = true;
                            }
                        }
                    }
                }
            }
            for (size_t level = 0; level <= fmin(curlevel, maxlevelcopy); level++) {
                if (level > maxlevelcopy || level < 0)
                    throw std::runtime_error("Level error");

                int32_t preserved_enterpoint_node = enterpoint_node;
                enterpoint_node = currObj;
                auto topResults = searchBaseLayer(point, efConstruction_, level);
                mutuallyConnectNewElement(point, cur_element_count, topResults, level);
                enterpoint_node = preserved_enterpoint_node;
            }
        } else {
            // Do nothing for the first element
            enterpoint_node = 0;
            max_level = curlevel;

        }
        if (curlevel > maxlevelcopy) {
            enterpoint_node = cur_element_count;
            max_level = curlevel;
        }
        cur_element_count++;
    };


    idx_t HierarchicalNSW::get_enterpoint(const float *query) {
        idx_t currObj = enterpoint_node;
        float curdist = fvec_L2sqr(query, getDataByInternalId(currObj), d_);
        dist_calc += (int) (max_level > 0);

        for (size_t level = max_level; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                uint16_t *data = level ? get_linklist_level(currObj, level) : get_linklist(currObj);
                size_t size = *data;
                auto *datal = (idx_t *) (data + 1);
                for (uint8_t i = 0; i < size; i++) {
                    idx_t cand = datal[i];
                    if (cand < 0 || cand > maxelements_)
                        throw std::runtime_error("candidate error");

                    float dist = fvec_L2sqr(query, getDataByInternalId(cand), d_);
                    dist_calc++;

                    if (dist < curdist) {
                        hops++;
                        curdist = dist;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }
        limit -= dist_calc;
        return currObj;
    }


    std::priority_queue<std::pair<float, idx_t>> HierarchicalNSW::searchKnn(const float *query, size_t k) {
        idx_t preserved_enterpoint_node = enterpoint_node;
        enterpoint_node = get_enterpoint(query);
        size_t query_dist_calc = dist_calc;
        auto topResults = searchBaseLayer(query, efSearch);
        enterpoint_node = preserved_enterpoint_node;
        limit += query_dist_calc;

        while (topResults.size() > k)
            topResults.pop();

        return topResults;
    }


    void HierarchicalNSW::SaveInfo(const std::string &location) {
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
        writeBinaryPOD(output, size_links_per_element);

        for (size_t i = 0; i < maxelements_; ++i) {
            idx_t linkListSize = elementLevels[i] > 0 ? size_links_per_element * elementLevels[i] : 0;
            writeBinaryPOD(output, linkListSize);
            if (linkListSize)
                output.write((char *) linkLists[i], linkListSize);
        }
    }


    void HierarchicalNSW::SaveEdges(const std::string &location) {
        std::cout << "Saving edges to " << location << std::endl;
        std::ofstream output(location, std::ios::binary);

        for (idx_t i = 0; i < maxelements_; i++) {
            uint16_t *ll_cur = get_linklist(i);
            uint32_t size = *ll_cur;

            output.write((char *) &size, sizeof(uint32_t));
            auto *data = (idx_t *) (ll_cur + 1);
            output.write((char *) data, sizeof(idx_t) * size);
        }
    }

    void HierarchicalNSW::LoadInfo(const std::string &location) {
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
        readBinaryPOD(input, size_links_per_element);

        d_ = data_size / sizeof(float);
        data_level0_memory_ = (char *) malloc(maxelements_ * size_data_per_element);

        efConstruction_ = 0;
        cur_element_count = maxelements_;

        visitedlistpool = new VisitedListPool(1, maxelements_);

        /** Hierarcy **/
        linkLists = (char **) malloc(sizeof(void *) * maxelements_);

        elementLevels = std::vector<uint8_t>(maxelements_);
        max_level = 0;

        for (size_t i = 0; i < maxelements_; i++) {
            idx_t linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize == 0) {
                elementLevels[i] = 0;
                linkLists[i] = nullptr;
            } else {
                elementLevels[i] = linkListSize / size_links_per_element;
                linkLists[i] = (char *) malloc(linkListSize);
                input.read(linkLists[i], linkListSize);
            }
            if (elementLevels[i] > max_level) {
                max_level = elementLevels[i];
            }
        }
        std::cout << enterpoint_node << " " << (int) (elementLevels[enterpoint_node]) << " " << max_level << std::endl;
    }

    void HierarchicalNSW::LoadData(const std::string &location) {
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

    void HierarchicalNSW::LoadEdges(const std::string &location) {
        std::cout << "Loading edges from " << location << std::endl;
        std::ifstream input(location, std::ios::binary);

        uint32_t size;

        for (idx_t i = 0; i < maxelements_; i++) {
            input.read((char *) &size, sizeof(uint32_t));

            uint16_t *ll_cur = get_linklist(i);
            *ll_cur = size;
            auto *data = (idx_t *) (ll_cur + 1);

            input.read((char *) data, size * sizeof(idx_t));
        }
    }

    HierarchicalNSW::HierarchicalNSW(size_t maxelements, size_t d,
                                     const std::string &dataLocation,
                                     const std::string &edgeLocation) {
        maxelements_ = maxelements;
        d_ = d;
        data_size = d_ * sizeof(float);
        LoadNSG(dataLocation, edgeLocation);
    }

    void HierarchicalNSW::LoadNSG(const std::string &dataLocation,
                                  const std::string &edgeLocation) {
        std::cout << "Loading nsg from " << edgeLocation << std::endl;
        std::ifstream input(edgeLocation, std::ios::binary);
        unsigned width, ep_;
        input.read((char *) &width, sizeof(unsigned));
        input.read((char *) &ep_, sizeof(unsigned));
        maxM_ = width;
        enterpoint_node = ep_;

        size_links_level0 = maxM_ * sizeof(idx_t) + sizeof(uint16_t);
        size_data_per_element = size_links_level0 + data_size;
        offset_data = size_links_level0;

        data_level0_memory_ = (char *) malloc(maxelements_ * size_data_per_element);
        std::cout << "Size Mb: " << (maxelements_ * size_data_per_element) / (1000 * 1000) << std::endl;

        visitedlistpool = new VisitedListPool(1, maxelements_);
        efConstruction_ = 0;
        max_level = 0;
        cur_element_count = maxelements_;
        elementLevels = std::vector<uint8_t>(maxelements_);

        int i = 0;
        while (!input.eof()) {
            unsigned size;
            input.read((char *) &size, sizeof(unsigned));
            if (input.eof()) break;

            uint16_t *ll_cur = get_linklist(i++);
            *ll_cur = size;
            auto *data = (idx_t *) (ll_cur + 1);

            input.read((char *) data, size * sizeof(idx_t));

        }
        LoadData(dataLocation);
    }

    /*
     * BFS for optimal paths construction
     */
    std::vector<idx_t> HierarchicalNSW::bfs(idx_t initial_vertex_id, idx_t gt, size_t margin)
    {
        size_t min_path_length = maxelements_;
        size_t current_depth = 0;

        std::vector<Vertex> forward_vertices(maxelements_);
        std::queue<std::pair<size_t, idx_t>> forward_queue;

        forward_queue.push({current_depth, initial_vertex_id});
        forward_vertices[initial_vertex_id].vertex_id = initial_vertex_id;
        forward_vertices[initial_vertex_id].is_visited = true;

        size_t forward_counter = 0;
        // Forward pass
        while (!forward_queue.empty()) {
            current_depth = forward_queue.front().first;
            Vertex *vertex = forward_vertices.data() + forward_queue.front().second;
            forward_queue.pop();

            if (vertex->vertex_id == gt)
                min_path_length = vertex->min_path_length;

            if (current_depth == min_path_length + margin)
                break;

            uint16_t *ll_cur = get_linklist(vertex->vertex_id);
            size_t size = *ll_cur;
            auto *data = (idx_t *) (ll_cur + 1);
            for (size_t i = 0; i < size; i++) {
                idx_t next_vertex_id = *(data + i);
                if (next_vertex_id != gt &&
                    (vertex->min_path_length + 1) == min_path_length + margin)
                    continue;

                Vertex *next_vertex = forward_vertices.data() + next_vertex_id;
                next_vertex->prev_vertex_ids.push_back(vertex->vertex_id);

                if (next_vertex->is_visited)
                    continue;

                next_vertex->vertex_id = next_vertex_id;
                next_vertex->is_visited = true;
                next_vertex->min_path_length = vertex->min_path_length + 1;
                forward_queue.push({current_depth + 1, next_vertex_id});
                forward_counter++;
            }
        }

        // Backward pass
        current_depth = 0;
        min_path_length = maxelements_;
        std::queue<std::pair<size_t, idx_t>> backward_queue;
        std::vector<Vertex> backward_vertices(maxelements_);
        backward_vertices[gt].vertex_id = gt;
        backward_vertices[gt].is_visited = true;

        backward_queue.push({current_depth, gt});

        size_t backward_counter = 0;
        while (!backward_queue.empty()) {
            current_depth = backward_queue.front().first;
            Vertex *vertex = backward_vertices.data() + backward_queue.front().second;
            backward_queue.pop();

            // check if is enterpoint
            if (vertex->vertex_id == initial_vertex_id)
                min_path_length = vertex->min_path_length;

            if (current_depth == min_path_length + margin)
                break;

            for (idx_t prev_vertex_id : forward_vertices[vertex->vertex_id].prev_vertex_ids) {
                if (prev_vertex_id != initial_vertex_id &&
                    (vertex->min_path_length + 1) == min_path_length + margin)
                    continue;

                Vertex *backward_vertex = backward_vertices.data() + prev_vertex_id;
                if (backward_vertex->is_visited)
                    continue;

                backward_vertex->vertex_id = prev_vertex_id;
                backward_vertex->is_visited = true;
                backward_vertex->min_path_length = vertex->min_path_length + 1;
                backward_queue.push({current_depth + 1, prev_vertex_id});
                backward_counter++;
            }
        }

        std::vector<idx_t> results;
        for (auto vertex : backward_vertices){
            if (!vertex.is_visited)
                continue;
            results.push_back(vertex.vertex_id);
            results.push_back((idx_t)vertex.min_path_length);
        }
        std::cout << forward_counter << " " << backward_counter << std::endl; //" Time: " <<
//                     stopw.getElapsedTimeMicro() * 1e-6 << std::endl;
        return results;
    }
}
