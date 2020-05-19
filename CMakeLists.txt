cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(aluserer
        LANGUAGES CXX CUDA
        DESCRIPTION "Geospatial algorithms on steroids."
        )

# Project wide compilation directives.
# For specific ones make yourself acquainted with different CMake files of each algorithm/entity.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "-Wall -Werror -Wextra -pedantic")

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Dependencies which should be installed on a host system and not included separately by this project.
find_package(CUDA REQUIRED)
if (CUDA_FOUND)
    message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler -Wextra")
    INCLUDE_DIRECTORIES(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
else (CUDA_FOUND)
    message(FATAL_ERROR "No CUDA toolchain found. Exiting.")
endif()

find_package(GDAL REQUIRED)

# Utilities
add_subdirectory(util/gdal)
add_subdirectory(util/sentinel1)
add_subdirectory(util/exotic_operations)
add_subdirectory(util/common)
add_subdirectory(util/cuda)
add_subdirectory(util/snap-engine)


# Algorithms
add_subdirectory(algs/inverter)
add_subdirectory(algs/inverter_parallel)
add_subdirectory(algs/cuda_algorithm)
add_subdirectory(algs/range_doppler_terrain_correction)
add_subdirectory(algs/backgeocoding)

# Test helpers
add_subdirectory(test/EGM96)

# Main terminal application
add_subdirectory(app)

# Tests
option(ENABLE_UNIT_TESTS "Enable unit tests" ON)
message(STATUS "Enable testing: ${ENABLE_UNIT_TESTS}")

if(ENABLE_UNIT_TESTS)
    add_subdirectory(test)
endif()
