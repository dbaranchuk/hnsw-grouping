#include "hnswalg_group.h"

namespace hnswlib {

    GroupHNSW::GroupHNSW(size_t dim, size_t ncentroids, size_t _M, size_t _maxM, size_t _efConstruction):
        d(dim), ngroups(ncentroids), M(_M), maxM(_maxM), efConstruction(_efConstruction)
    {
        data.resize(ngroups);
        links.resize(ngroups);
        ids.resize(ngroups);
        visitedlistpool = new VisitedListPool(1, ngroups);
    }


    GroupHNSW::~GroupHNSW(){
        delete visitedlistpool;
    }


    std::priority_queue<std::pair<float, idx_t>> GroupHNSW::searchKnn(const float *query, size_t k)
    {
        VisitedList *vl = visitedlistpool->getFreeVisitedList();
        vl_type *massVisited = vl->mass;
        vl_type currentV = vl->curV;

        std::priority_queue<std::pair<float, idx_t >> knn;
        std::priority_queue<std::pair<float, idx_t >> topResults;
        std::priority_queue<std::pair<float, idx_t >> candidateSet;

        float dist = query2group_dist(query, enterpoint_node, knn);
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
            idx_t current_node = curr_el_pair.second;
            const std::vector<idx_t> &grouplinks = links[current_node];

            for (idx_t node : grouplinks){
                if (massVisited[node] != currentV) {
                    massVisited[node] = currentV;
                    float dist = query2group_dist(query, node, knn);

                    if (topResults.top().first > dist || topResults.size() < efSearch) {
                        candidateSet.emplace(-dist, node);
                        topResults.emplace(dist, node);
                        if (topResults.size() > efSearch)
                            topResults.pop();

                        lowerBound = topResults.top().first;
                    }
                }
            }
            while(knn.size() > k)
                knn.pop();
        }
        visitedlistpool->releaseVisitedList(vl);

        return knn;
    }


    std::priority_queue<std::pair<float, idx_t>> GroupHNSW::searchGroupLayer(idx_t target_node, size_t ef)
    {
        VisitedList *vl = visitedlistpool->getFreeVisitedList();
        vl_type *massVisited = vl->mass;
        vl_type currentV = vl->curV;

        std::priority_queue<std::pair<float, idx_t >> topResults;
        std::priority_queue<std::pair<float, idx_t >> candidateSet;

        //float dist = group2group_dist(target_node, enterpoint_node);
        float dist = fvec_L2sqr(centroids.data() + target_node*d, centroids.data() + enterpoint_node*d, d);
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
            idx_t current_node = curr_el_pair.second;
            const std::vector<idx_t> &grouplinks = links[current_node];

            for (idx_t node : grouplinks){
                if (massVisited[node] != currentV) {
                    massVisited[node] = currentV;
                    //float dist = group2group_dist(target_node, node);
                    float dist = fvec_L2sqr(centroids.data() + target_node*d, centroids.data() + node*d, d);

                    if (topResults.top().first > dist || topResults.size() < ef) {
                        candidateSet.emplace(-dist, node);
                        topResults.emplace(dist, node);
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


    void GroupHNSW::mutuallyConnectGroup(idx_t current_node, std::priority_queue<std::pair<float, idx_t>> &topResults)
    {
        getNeighborsByHeuristic(topResults, M);

        std::vector<idx_t> &grouplinks = links[current_node];
        
        if (grouplinks.size() > 0)
            throw std::runtime_error("Should be blank");
        
        grouplinks.reserve(M);
        while (topResults.size() > 0) {
            grouplinks.push_back(topResults.top().second);
            topResults.pop();
        }
        for (idx_t node : grouplinks) {
            if (node == current_node)
                throw std::runtime_error("Connection to the same element");

            std::vector<idx_t> &other_grouplinks = links[node];
            if (other_grouplinks.size() > maxM)
                throw std::runtime_error("Bad other_grouplinks size");

            else if (other_grouplinks.size() < maxM) {
                other_grouplinks.push_back(current_node);
            } else {
                // finding the "weakest" element to replace it with the new one
                //float d_max = group2group_dist(current_node, node);
                float d_max = fvec_L2sqr(centroids.data() + current_node*d, centroids.data() + node*d, d);
                std::priority_queue<std::pair<float, idx_t>> candidates;
                candidates.emplace(d_max, current_node);

                for (idx_t other_grouplink : other_grouplinks)
                    //candidates.emplace(group2group_dist(other_grouplink, node), other_grouplink);
                    candidates.emplace(fvec_L2sqr(centroids.data() + other_grouplink*d, centroids.data() + node*d, d);, other_grouplink);

                getNeighborsByHeuristic(candidates, maxM);

                size_t i = 0;
                while (candidates.size() > 0) {
                    idx_t link = candidates.top().second;
                    if (i == other_grouplinks.size())
                        other_grouplinks.push_back(link);
                    else
                        other_grouplinks[i] = link;
                    candidates.pop();
                    i++;
                }
            }
        }
    }


    void GroupHNSW::getNeighborsByHeuristic(std::priority_queue<std::pair<float, idx_t>> &topResults, size_t NN)
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
            std::pair<float, idx_t> current = resultSet.top();
            float dist_to_query = -current.first;
            resultSet.pop();
            bool good = true;
            for (std::pair<float, idx_t> current2 : returnlist) {
                //float curdist = group2group_dist(current2.second, current.second);
                float curdist = fvec_L2sqr(centroids.data() + current2.second*d, centroids.data() + current.second*d, d);
                if (curdist < dist_to_query) {
                    good = false;
                    break;
                }
            }
            if (good) returnlist.push_back(current);
        }
        for (std::pair<float, idx_t> elem : returnlist)
            topResults.emplace(-elem.first, elem.second);
    }


    void GroupHNSW::addGroup(const std::vector<float> &group, const std::vector<idx_t> &idxs)
    {
        idx_t current_node = 0;
        {
            std::unique_lock <std::mutex> lock(cur_element_count_guard);
            if (cur_element_count >= ngroups) {
                std::cout << "The number of elements exceeds the specified limit\n";
                throw std::runtime_error("The number of elements exceeds the specified limit");
            };
            current_node = cur_element_count;
            cur_element_count++;
        }

        std::unique_lock <std::mutex> templock(global);
        if (enterpoint_node != -1)
            templock.unlock();

        data[current_node] = group;
        ids[current_node] = idxs;
            
        if (enterpoint_node != -1) {
            std::priority_queue<std::pair<float, idx_t>> topResults = searchGroupLayer(current_node, efConstruction);
            mutuallyConnectGroup(current_node, topResults);
        } else {
            // Do nothing for the first element
            enterpoint_node = 0;
        }
    }


    void GroupHNSW::read(const std::string &location)
    {
        std::cout << "Loading index from " << location << std::endl;
        std::ifstream input(location, std::ios::binary);

        read_variable(input, d);
        read_variable(input, ngroups);
        read_variable(input, M);
        read_variable(input, maxM);
        read_variable(input, enterpoint_node);

        // Read data
        for (size_t i = 0; i < ngroups; i++)
            read_vector(input, data[i]);

        // Read vector indices
        for (size_t i = 0; i < ngroups; i++)
            read_vector(input, ids[i]);

        // Read links
        for (size_t i = 0; i < ngroups; i++)
            read_vector(input, links[i]);

        // Read centroids
        read_vector(input, centroids);
    }

    void GroupHNSW::write(const std::string &location)
    {
        std::cout << "Saving index to " << location << std::endl;
        std::ofstream output(location, std::ios::binary);

        write_variable(output, d);
        write_variable(output, ngroups);
        write_variable(output, M);
        write_variable(output, maxM);
        write_variable(output, enterpoint_node);

        // Write data
        for (size_t i = 0; i < ngroups; i++)
            write_vector(output, data[i]);

        // Write vector indices
        for (size_t i = 0; i < ngroups; i++)
            write_vector(output, ids[i]);

        // Write links
        for (size_t i = 0; i < ngroups; i++)
            write_vector(output, links[i]);

        // Write centroids
        write_vector(output, centroids);
    }


    float GroupHNSW::group2group_dist(idx_t group_id1, idx_t group_id2)
    {
        size_t groupsize1 = data[group_id1].size() / d;
        const float *group1 = data[group_id1].data();
        size_t groupsize2 = data[group_id2].size() / d;
        const float *group2 = data[group_id2].data();

        float min_dist = -1;
        for (size_t i = 0; i < groupsize1; i++){
            for (size_t j = 0; j < groupsize2; j++){
                float dist = fvec_L2sqr(group1 + i*d, group2 + j*d, d);
                if (dist < min_dist || min_dist < 0)
                    min_dist = dist;
            }
        }
        return min_dist;
    }


    float GroupHNSW::query2group_dist(const float *query, idx_t group_id, std::priority_queue<std::pair<float, idx_t>> &knn)
    {
        std::priority_queue<std::pair<float, idx_t>> heap;
        size_t groupsize = ids[group_id].size();
        const float *group = data[group_id].data();
        const idx_t *id = ids[group_id].data();

        for (size_t i = 0; i < groupsize; i++){
            float dist = fvec_L2sqr(query, group + i*d, d);
            dist_calc++;
            heap.emplace(-dist, id[i]);
        }
        float result = -heap.top().first;
        //join heaps
        while(!heap.empty()) {
            knn.emplace(-heap.top().first, heap.top().second);
            heap.pop();
        }
        return result;
    }
}