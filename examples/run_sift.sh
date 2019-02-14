#!/bin/bash

#########################
# Release baseline test #
#########################

limit="0"

################################
# HNSW construction parameters #
################################

M="12"                # Min number of edges per point
efConstruction="300"  # Max number of candidate vertices in priority queue to observe during construction
onelayer="1"

###################
# Data parameters #
###################

nb="1000000"          # Number of base vectors
nt="1000000"          # Number of learn vectors

nq="10000"            # Number of queries
ngt="1"               # Number of groundtruth neighbours per query

d="128"               # Vector dimension

#####################
# Search parameters #
#####################

k="1"                  # Number of the closest vertices to search
efSearch="300"         # Max number of candidate vertices in priority queue to observe during searching

#########
# Paths #
#########

path_data="${PWD}/data/SIFT1M"
path_model="${PWD}/models/SIFT1M"
path_base="${path_data}/sift_base.fvecs"
path_learn="${path_data}/sift_learn.fvecs"
path_gt="${path_data}/test_gt.ivecs" #sift_groundtruth.ivecs"
path_q="${path_data}/sift_query.fvecs"

path_edges="${path_model}/test_hnsw_M${M}_ef${efConstruction}_onelevel${onelayer}.ivecs" 
path_info="${path_model}/test_hnsw_M${M}_ef${efConstruction}_onelevel${onelayer}.bin"

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
