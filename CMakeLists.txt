cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(aluserer
        LANGUAGES CXX CUDA
        DESCRIPTION "Geospatial algorithms on steroids."
        )

# Project wide compilation directives.
# For specific ones make yourself acquainted with different CMake files of each algorithm/entity.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "-Wall -Werror -Wextra -pedantic")

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Dependencies which should be installed on a host system and not included separately by this project.
find_package(CUDA REQUIRED)
message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")

find_package(GDAL REQUIRED)

# Algorithms
add_subdirectory(algs/inverter)
add_subdirectory(algs/cuda_algorithm)

# Main terminal application
add_subdirectory(app)
