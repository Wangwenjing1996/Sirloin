cmake_minimum_required( VERSION 3.24.0)

project( baseline)

include_directories( faiss-main)
include_directories( include)

add_subdirectory( faiss-main)

find_package(OpenMP REQUIRED)

add_executable( main main.cpp)

target_link_libraries( main faiss OpenMP::OpenMP_CXX)