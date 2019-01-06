#ifndef IVF_HNSW_LIB_PARSER_H
#define IVF_HNSW_LIB_PARSER_H

#include <cstring>
#include <iostream>

//==============
// Parser Class
//==============
struct Parser
{
    const char *cmd;        ///< main command - argv[0]

    //=================
    // Limit parameter
    //=================
	
    size_t limit;

    //=================
    // HNSW parameters
    //=================
    size_t M;               ///< Min number of edges per point
    size_t efConstruction;  ///< Max number of candidate vertices in priority queue to observe during construction

    //=================
    // Data parameters
    //=================
    size_t nb;             ///< Number of base vectors
    size_t nt;             ///< Number of learn vectors
    size_t nq;             ///< Number of queries
    size_t nc;             ///< Number of groups
    size_t ngt;            ///< Number of groundtruth neighbours per query
    size_t d;              ///< Vector dimension


    //===================
    // Search parameters
    //===================
    size_t k;              ///< Number of the closest vertices to search
    size_t efSearch;       ///< Max number of candidate vertices in priority queue to observe during searching

    //=======
    // Paths
    //=======
    const char *path_base;             ///< Path to a base set
    const char *path_learn;            ///< Path to a learn set
    const char *path_q;                ///< Path to queries
    const char *path_gt;               ///< Path to groundtruth
    const char *path_group_idxs;       ///< Path to group idxs
    const char *path_centroids;        ///< Path to centroids

    const char *path_info;             ///< Path to parameters of HNSW graph
    const char *path_edges;            ///< Path to edges of HNSW graph
    const char *path_index;            ///< Path to the GroupHNSW index

    Parser(int argc, char **argv)
    {
        cmd = argv[0];
        if (argc == 1)
            usage();

        for (size_t i = 1 ; i < argc; i++) {
            char *a = argv[i];

            if (!strcmp (a, "-h") || !strcmp (a, "--help"))
                usage();

            if (i == argc-1)
                break;

            //=================
            // HNSW parameters
            //=================
            if (!strcmp (a, "-M")) sscanf(argv[++i], "%zu", &M);
            else if (!strcmp (a, "-efConstruction")) sscanf(argv[++i], "%zu", &efConstruction);
	    else if (!strcmp (a, "-limit")) sscanf(argv[++i], "%zu", &limit);

            //=================
            // Data parameters
            //=================
            else if (!strcmp (a, "-nb")) sscanf(argv[++i], "%zu", &nb);
            else if (!strcmp (a, "-nt")) sscanf(argv[++i], "%zu", &nt);
            else if (!strcmp (a, "-nq")) sscanf(argv[++i], "%zu", &nq);
            else if (!strcmp (a, "-nc")) sscanf(argv[++i], "%zu", &nc);
            else if (!strcmp (a, "-ngt")) sscanf(argv[++i], "%zu", &ngt);
            else if (!strcmp (a, "-d")) sscanf(argv[++i], "%zu", &d);

            //===================
            // Search parameters
            //===================
            else if (!strcmp (a, "-k")) sscanf(argv[++i], "%zu", &k);
            else if (!strcmp (a, "-efSearch")) sscanf(argv[++i], "%zu", &efSearch);

            //=======
            // Paths
            //=======
            else if (!strcmp (a, "-path_base")) path_base = argv[++i];
            else if (!strcmp (a, "-path_learn")) path_learn = argv[++i];
            else if (!strcmp (a, "-path_q")) path_q = argv[++i];
            else if (!strcmp (a, "-path_gt")) path_gt = argv[++i];
            else if (!strcmp (a, "-path_group_idxs")) path_group_idxs = argv[++i];
            else if (!strcmp (a, "-path_centroids")) path_centroids = argv[++i];
            else if (!strcmp (a, "-path_info")) path_info = argv[++i];
            else if (!strcmp (a, "-path_edges")) path_edges = argv[++i];
            else if (!strcmp (a, "-path_index")) path_index = argv[++i];
        }
    }

    void usage()
    {
        printf ("Usage: %s [options]\n", cmd);
        printf ("###################\n"
                "# HNSW Parameters #\n"
                "###################\n"
                "    -M #                  Min number of edges per point\n"
                "    -efConstruction #     Max number of candidate vertices in priority queue to observe during construction\n"
                "###################\n"
                "# Data Parameters #\n"
                "###################\n"
                "    -nb #                 Number of base vectors\n"
                "    -nt #                 Number of learn vectors\n"
                "    -nq #                 Number of queries\n"
                "    -ngt #                Number of groundtruth neighbours per query\n"
                "    -d #                  Vector dimension\n"
                "####################\n"
                "# Search Parameters #\n"
                "#####################\n"
                "    -k #                  Number of the closest vertices to search\n"
                "    -efSearch #           Max number of candidate vertices in priority queue to observe during searching\n"
                "#########\n"
                "# Paths #\n"
                "#########\n"
                "    -path_base filename               Path to a base set\n"
                "    -path_learn filename              Path to a learn set\n"
                "    -path_q filename                  Path to queries\n"
                "    -path_gt filename                 Path to groundtruth\n"
                "    -path_info filename               Path to parameters of HNSW graph\n"
                "    -path_edges filename              Path to edges of HNSW graph\n"
                "    -path_index filename              Path to the GroupHNSW index\n"
        );
        exit(0);
    }
};

#endif //IVF_HNSW_LIB_PARSER_H
