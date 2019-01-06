#include "../utils.h"
#include "../Parser.h"
#include "../hnswalg.h"

using namespace hnswlib;

static void test_approx(float *massQ, size_t nq,
                        HierarchicalNSW *quantizer, size_t d,
                        std::vector<idx_t> &answers)
{
    std::vector<std::vector<idx_t>> results(nq);
    for (int i = 0; i < 10; i++) {
        idx_t enterpoint = quantizer->get_enterpoint(massQ + d*i);
        results[i] = quantizer->bfs(enterpoint, answers[i], 2);
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

    //===========
    // Load HNSW
    //===========
    HierarchicalNSW *quantizer;
    if (exists(opt.path_info) &&  exists(opt.path_edges)) {
        quantizer = new HierarchicalNSW(opt.path_info, opt.path_base, opt.path_edges);
        quantizer->efSearch = opt.efConstruction;
        quantizer->limit = opt.limit;
    } else {
        throw std::string(" Graph does not exist\n");
    }

    //===================
    // Parse groundtruth
    //===================
    std::cout << "Parsing groundtruth" << std::endl;
    std::vector<idx_t> answers(opt.nq);
    for (size_t i = 0; i < opt.nq; i++)
        answers[i] = massQA[opt.ngt*i];

    test_approx(massQ.data(), opt.nq, quantizer, opt.d, answers);

    delete quantizer;
}
