#!/bin/bash

limit="0"

################################
# HNSW construction parameters #
################################

M="8"                # Min number of edges per point
efConstruction="300"  # Max number of candidate vertices in priority queue to observe during construction
onelayer="0"

###################
# Data parameters #
###################

nb="100000"          # Number of base vectors
nt="100000"           # Number of learn vectors

nq="89983"            # Number of queries
ngt="1"             # Number of groundtruth neighbours per query

d="128"               # Vector dimension

#####################
# Search parameters #
#####################

k="1"                  # Number of the closest vertices to search
efSearch="300"         # Max number of candidate vertices in priority queue to observe during searching

#########
# Paths #
#########

path_data="${PWD}/data/sift"
path_model="${PWD}/models/sift"
path_base="${path_data}/sift_base.fvecs"
path_learn="${path_data}/sift_learn.fvecs"
path_gt="${path_data}/train_gt.ivecs" #sift_groundtruth.ivecs"
path_q="${path_data}/sift_learn_filtered.fvecs"

path_edges="${path_model}/test_hnsw100k_M${M}_ef${efConstruction}.ivecs" #levels
path_info="${path_model}/test_hnsw100k_M${M}_ef${efConstruction}.bin" #levels

#######
# Run #
#######
 ${PWD}/bin/test_bfs -M ${M} \
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
                     -path_info ${path_info} \
		             -limit ${limit} \
		             -onelayer ${onelayer}
