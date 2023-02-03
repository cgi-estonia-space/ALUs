cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(ALUs
        LANGUAGES CXX CUDA
        DESCRIPTION "Parallel ALUs for geospatial data processing"
        VERSION 1.4.0
        )

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Exports compile commands used by clang-tidy
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Dependencies which should be installed on a host system and not included separately by this project.

find_package(CUDAToolkit 11.2 REQUIRED)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wall")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Werror")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wextra")
# To suppress using of cmath std::min and std::max.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --Werror all-warnings")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
# mapping executable GPU code to source code, only enable for testing/profiling
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --resource-usage")

INCLUDE_DIRECTORIES(${CUDAToolkit_INCLUDE_DIRS})

if (NOT DEFINED ENV{CUDAARCHS})
    message(STATUS "CUDAARCHS not set, defaulting")
    set(CMAKE_CUDA_ARCHITECTURES 60 CACHE STRING "CUDA architectures" FORCE)
endif()
message(STATUS "CUDA binary code will be generated for the following architecture(s) - ${CMAKE_CUDA_ARCHITECTURES}")

find_package(Boost 1.71.0 REQUIRED COMPONENTS program_options date_time filesystem iostreams log log_setup)

find_package(GDAL REQUIRED)
include_directories(${GDAL_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)

set(ALUS_BINARY_OUTPUT_DIR ${PROJECT_BINARY_DIR}/alus_package)
set(ALUS_UNIT_TEST_BINARY_OUTPUT_DIR ${PROJECT_BINARY_DIR}/unit-test)
set(ALUS_LFS_INVOKE ${CMAKE_CURRENT_LIST_DIR}/.alus-lfs)
set(ALUS_LFS_GOODS ${CMAKE_CURRENT_LIST_DIR}/resources)

option(ENABLE_TESTS "Enable tests" OFF)

#external library as git submodule with separate CMakeLists.txt project
set(BUILD_TEST OFF CACHE BOOL "Build tests")
set(BUILD_STATIC_VERSION ON CACHE BOOL "Build static library")
set(BUILD_SHARED_VERSION OFF CACHE BOOL "Build shared library")
add_subdirectory(external/zipper)

# Project wide compilation directives.
# For specific ones make yourself acquainted with different CMake files of each algorithm/entity.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -Wextra -pedantic -DBOOST_LOG_DYN_LINK")

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(STATUS "Setting GNU compiler specific flags")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=attributes")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
else ()
    message(WARNING "Building the project may fail with not supported compiler - " ${CMAKE_CXX_COMPILER_ID})
endif ()

if (SONARCLOUD_BUILD)
    # fix build errors with SonarCloud code scanner
    # workaround for parser crash with boost::asio - SonarCFamily CPP-2696
    message(STATUS "SonarCloud configuration")
    add_compile_definitions(BOOST_ASIO_DISABLE_NOEXCEPT)
endif ()

# External libraries
add_subdirectory(external/pugixml)

# Utilities
add_subdirectory(util/gdal)
add_subdirectory(sentinel1)
add_subdirectory(util/exotic-operations)
add_subdirectory(util/common)
add_subdirectory(util/cuda)
add_subdirectory(util/dem)
add_subdirectory(snap-engine)

# Algorithms
add_subdirectory(algs/coherence-cuda)
add_subdirectory(algs/range-doppler-terrain-correction)
add_subdirectory(algs/backgeocoding)
add_subdirectory(algs/apply-orbit-file-op)
add_subdirectory(algs/feature-extraction-gabor)
add_subdirectory(algs/sentinel1-calibrate)
add_subdirectory(algs/topsar-deburst-op)
add_subdirectory(algs/coherence-estimation-routine)
add_subdirectory(algs/coregistration)
add_subdirectory(algs/topsar-split)
add_subdirectory(algs/calibration-routine)
add_subdirectory(algs/resample)
add_subdirectory(algs/topsar-merge)
add_subdirectory(algs/thermal-noise-removal)
add_subdirectory(algs/palsar-focus)

# Test helpers
add_subdirectory(test/range-doppler-geocoding)
add_subdirectory(test/tie-point-grid)
add_subdirectory(test/sar-geocoding)

# Main terminal application
add_subdirectory(app)

include(FetchContent)

FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.10.0
)

FetchContent_GetProperties(googletest)

if (ENABLE_TESTS)

    if (NOT googletest_POPULATED)
        FetchContent_Populate(googletest)

        # Prevent GoogleTest from overriding our compiler/linker options
        # when building with Visual Studio
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        # Prevent GoogleTest from using PThreads
        set(gtest_disable_pthreads ON CACHE BOOL "" FORCE)
        set(BUILD_SHARED_VERSION OFF CACHE BOOL "" FORCE)
        set(BUILD_TEST OFF CACHE BOOL "" FORCE)
        set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

        # adds the targers: gtest, gtest_main, gmock, gmock_main
        add_subdirectory(
                ${googletest_SOURCE_DIR}
                ${googletest_BINARY_DIR}
        )

        # Silence std::tr1 warning on MSVC
        if (MSVC)
            foreach (_tgt gmock)
                target_compile_definitions(${_tgt}
                        PRIVATE
                        "_SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING"
                        )
            endforeach ()
        endif ()
    endif ()
    add_subdirectory(test)
    add_subdirectory(test-integration)
endif ()
