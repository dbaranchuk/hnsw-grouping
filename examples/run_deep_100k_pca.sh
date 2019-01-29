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

nq="10000"            # Number of queries
ngt="1"             # Number of groundtruth neighbours per query

d="24"               # Vector dimension

#####################
# Search parameters #
#####################

k="1"                  # Number of the closest vertices to search
efSearch="300"         # Max number of candidate vertices in priority queue to observe during searching

#########
# Paths #
#########

path_data="${PWD}/data/DEEP100K"
path_model="${PWD}/models/DEEP100K"
path_base="${path_data}/deep_base_pca${d}.fvecs"
path_learn="${path_data}/deep_learn_pca${d}.fvecs"
path_gt="${path_data}/test_gt.ivecs"
path_q="${path_data}/deep_query_pca${d}.fvecs"

#path_edges="${path_model}/test_hnsw100k_M16_ef300_RL_efSearch1_batch4k_hid1024_stoh88k.ivecs"
#path_edges="${path_model}/hnsw100k_M10_ef300_RL_CHreward_alpha10.ivecs"
#path_edges="${path_model}/hnsw100k_M8_ef300_RL_dmax1000_2.ivecs"
path_edges="${path_model}/test_hnsw100k_M${M}_ef${efConstruction}_onelevel${onelayer}_pca${d}.ivecs"
path_info="${path_model}/test_hnsw100k_M${M}_ef${efConstruction}_onelevel${onelayer}_pca${d}.bin"

#######
# Run #
#######
${PWD}/bin/test_hnsw -M ${M} \
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
