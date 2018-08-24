cmake_minimum_required (VERSION 2.8)

# hnswlib project
project(hnsw C CXX)

include_directories("${PROJECT_BINARY_DIR}")

# specify output bin_path and lib_path
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

file(GLOB hnsw_cpu_headers ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
file(GLOB hnsw_cpu_cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_library(hnsw STATIC ${hnsw_cpu_headers} ${hnsw_cpu_cpp})

# Build each source file independently
include_directories(../../)	# hnsw root directory

SET( CMAKE_CXX_FLAGS "-Ofast -lrt -DNDEBUG -std=c++14 -DHAVE_CXX0X -openmp -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )
target_link_libraries(hnsw)

# build tests
add_subdirectory(tests)
