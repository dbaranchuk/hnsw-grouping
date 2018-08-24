#include "../utils.h"
#include "../Parser.h"
#include "../hnswalg_group.h"

using namespace hnswlib;

static float test_approx(float *massQ, size_t nq,  HierarchicalNSW *quantizer, size_t d,
                         std::vector<std::priority_queue< std::pair<float,  idx_t >>> &answers, size_t k)
{
    size_t correct = 0;
    size_t total = 0;
    for (int i = 0; i < nq; i++) {
        std::priority_queue< std::pair<float,  idx_t >> result = quantizer->searchKnn(massQ + d*i, k);
        std::priority_queue<std::pair<float, idx_t >> gt(answers[i]);
        std::unordered_set<idx_t> g;
        total += gt.size();
        while (gt.size()) {
            g.insert(gt.top().second);
            gt.pop();
        }
        while (result.size()) {
            if (g.find(result.top().second) != g.end())
                correct++;
            result.pop();
        }
    }
    return 1.0f * correct / total;
}

static void test_vs_recall(float *massQ, size_t nq,  HierarchicalNSW *quantizer,
                           size_t d, std::vector<std::priority_queue< std::pair<float,  idx_t>>> &answers, size_t k)
{
    std::vector<size_t> efs;// = {k}; //= {30, 100, 460};
    for (int i = 1; i < 10; i++) efs.push_back(i);
    for (int i = 10; i < 100; i += 10) efs.push_back(i);
    for (int i = 100; i <= 500; i += 40) efs.push_back(i);

    for (size_t ef : efs) {
        quantizer->efSearch = ef;
        quantizer->dist_calc = 0;
        quantizer->hops = 0.0;
        StopW stopw =  StopW();
        float recall = test_approx(massQ, nq, quantizer, d, answers, k);
        float time_us_per_query = stopw.getElapsedTimeMicro() / nq;
        float avr_dist_count = quantizer->dist_calc*1.f / nq;
        std::cout << ef << "\t" << recall << "\t" << time_us_per_query << " us\t" << avr_dist_count << " dcs\t" << quantizer->hops << " hps\n";
    }
}

int main(int argc, char **argv)
{
    //===============
    // Parse Options
    //===============
    Parser opt = Parser(argc, argv);

    //==================
    // Load Groundtruth
    //==================
    std::cout << "Loading groundtruth from " << opt.path_gt << std::endl;
    std::vector<idx_t> massQA(opt.nq * opt.ngt);
    {
        std::ifstream gt_input(opt.path_gt, std::ios::binary);
        readXvec<idx_t>(gt_input, massQA.data(), opt.ngt, opt.nq);
    }

    //==============
    // Load Queries
    //==============
    std::cout << "Loading queries from " << opt.path_q << std::endl;
    std::vector<float> massQ(opt.nq * opt.d);
    {
        std::ifstream query_input(opt.path_q, std::ios::binary);
        readXvec<float>(query_input, massQ.data(), opt.d, opt.nq);
    }

    //============
    // Build HNSW
    //============
    GroupHNSW *quantizer = new  GroupHNSW(opt.d, opt.nc, opt.M, 2*opt.M, opt.efConstruction);
    if ( exists(opt.path_info) &&  exists(opt.path_edges)) {
        quantizer->read(opt.path_index);
    } else {
        //================
        // Load centroids
        //================
        std::cout << "Loading centroids from " << opt.path_centroids << std::endl;
        std::vector<float> centroids(opt.nc * opt.d);
        {
            std::ifstream centroids_input(opt.path_centroids, std::ios::binary);
            readXvec<float>(centroids_input, centroids.data(), opt.d, opt.nc);
        }
        quantizer->centroids = centroids;

        //===========
        // Load Base
        //===========
        std::cout << "Loading base from " << opt.path_base << std::endl;
        std::vector<float> massB(opt.nb * opt.d);
        {
            std::ifstream base_input(opt.path_base, std::ios::binary);
            readXvec<float>(base_input, massB.data(), opt.d, opt.nb);
        }

        //==========
        // Load Ids
        //==========
        std::cout << "Loading group idxs from " << opt.path_group_ids << std::endl;
        std::vector<idx_t> group_idxs(opt.nc);
        {
            std::ifstream group_idxs_input(opt.path_group_idxs, std::ios::binary);
            readXvec<idx_t>(group_idxs_input, group_idxs.data(), 1, opt.nc);
        }

        //============
        // Get groups
        //============
        std::cout << "Forming groups" << std::endl;
        std::vector<std::vector<float>> groups(opt.nc);
        std::vector<std::vector<idx_t>> ids(opt.nc);
        for (size_t i = 0; i < nb; i++){
            int idx = group_idxs[i];
            ids[idx].push_back(i);
            for (int j = 0; j < d; j++)
                groups[idx].push_back(massB[i*d+j]);
        }

        //=====================
        // Construct GroupHNSW
        //=====================
        std::cout << "Constructing quantizer\n";
        size_t report_every = 10000;
        int j1 = 0;
        quantizer->addGroup(groups[0], ids[0]);
#pragma omp parallel for
        for (int i = 1; i < opt.nc; i++) {
#pragma omp critical
            {
                if (++j1 % report_every == 0)
                    std::cout << j1 / (0.01 * opt.nc) << " %\n";
            }
            quantizer->addGroup(groups[i], ids[i]);
        }
        quantizer->write(opt.path_index);
    }

    //===================
    // Parse groundtruth
    //===================
    std::cout << "Parsing groundtruth" << std::endl;
    std::vector<std::priority_queue< std::pair<float,  idx_t >>> answers;
    (std::vector<std::priority_queue< std::pair<float, idx_t >>>(opt.nq)).swap(answers);
    for (size_t i = 0; i < opt.nq; i++)
        answers[i].emplace(0.0f, massQA[opt.ngt*i]);

    test_vs_recall(massQ.data(), opt.nq, quantizer, opt.d, answers, opt.k);

    delete quantizer;
}