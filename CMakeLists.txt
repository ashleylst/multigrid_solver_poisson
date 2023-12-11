cmake_minimum_required(VERSION 3.27)
project(multigrid)

set(CMAKE_CXX_STANDARD 17)

FIND_PACKAGE(OpenMP REQUIRED)
if (OPENMP_FOUND)
    MESSAGE("openmp found")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

add_executable(multigrid main.cpp)
