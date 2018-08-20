#!/bin/bash

################################
# HNSW construction parameters #
################################

M="16"                # Min number of edges per point
efConstruction="500"  # Max number of candidate vertices in priority queue to observe during construction

###################
# Data parameters #
###################

nb="1000000"          # Number of base vectors
nt="100000"           # Number of learn vectors
nq="10000"            # Number of queries
ngt="1"               # Number of groundtruth neighbours per query

d="96"                # Vector dimension

#####################
# Search parameters #
#####################

k="100"                # Number of the closest vertices to search
efSearch="130"         # Max number of candidate vertices in priority queue to observe during searching

#########
# Paths #
#########

path_data="${PWD}/data/deep"
path_model="${PWD}/models/deep"

path_base="${path_data}/deep_base.fvecs"
path_learn="${path_data}/deep_learn.fvecs"
path_gt="${path_data}/deep_groundtruth.ivecs"
path_q="${path_data}/deep_queries.fvecs"

path_edges="${path_model}/hnsw_M${M}_ef${efConstruction}.ivecs"
path_info="${path_model}/hnsw_M${M}_ef${efConstruction}.bin"

#######
# Run #
#######
${PWD}/bin/test_deep -M ${M} \
                     -efConstruction ${efConstruction} \
                     -nb ${nb} \
                     -nt ${nt} \
                     -nq ${nq} \
                     -ngt ${ngt} \
                     -d ${d} \
                     -k ${k} \
                     -efSearch ${efSearch} \
                     -path_base ${path_base} \
                     -path_learn ${path_learn} \
                     -path_gt ${path_gt} \
                     -path_q ${path_q} \
                     -path_edges ${path_edges} \
                     -path_info ${path_info}