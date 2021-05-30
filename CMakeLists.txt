cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(aluserer
        LANGUAGES CXX CUDA
        DESCRIPTION "Geospatial algorithms on steroids."
        VERSION 0.5.0
        )

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Dependencies which should be installed on a host system and not included separately by this project.
find_package(CUDA REQUIRED)
if (CUDA_FOUND)
    message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wall")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Werror")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wextra")
    # To suppress using of cmath std::min and std::max.
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --Werror all-warnings")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
    #    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --resource-usage")
    INCLUDE_DIRECTORIES(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    if (CUDA_ARCH)
        set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH} CACHE STRING "CUDA architectures" FORCE)
    else (CUDA_ARCH)
        set(CMAKE_CUDA_ARCHITECTURES 50 CACHE STRING "CUDA architectures" FORCE)
    endif ()
    message(STATUS "CUDA binary code will be generated for the following architecture(s) - ${CMAKE_CUDA_ARCHITECTURES}")
else (CUDA_FOUND)
    message(FATAL_ERROR "No CUDA toolchain found. Exiting.")
endif ()

find_package(Boost 1.74.0 REQUIRED COMPONENTS program_options date_time filesystem iostreams)

find_package(GDAL REQUIRED)
include_directories(${GDAL_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)

set(ALUS_BINARY_OUTPUT_DIR ${PROJECT_BINARY_DIR}/alus_package)
set(ALUS_UNIT_TEST_BINARY_OUTPUT_DIR ${PROJECT_BINARY_DIR}/unit-test)

option(ENABLE_UNIT_TESTS "Enable unit tests" ON)
option(ENABLE_INTEGRATION_TESTS "Enable integration tests" ON)

#external library as git submodule with separate CMakeLists.txt project
add_subdirectory(external/zipper)

# Project wide compilation directives.
# For specific ones make yourself acquainted with different CMake files of each algorithm/entity.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "-Wall -Werror -Wextra -pedantic")

set(CMAKE_SKIP_BUILD_RPATH  FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH $ORIGIN)

if (SONARCLOUD_BUILD)
    # fix build errors with SonarCloud code scanner
    # workaround for parser crash with boost::asio - SonarCFamily CPP-2696
    message(STATUS "SonarCloud configuration")
    add_compile_definitions(BOOST_ASIO_DISABLE_NOEXCEPT)
endif()

# External libraries
add_subdirectory(external/pugixml)

# Utilities
add_subdirectory(util/gdal)
add_subdirectory(util/sentinel1)
add_subdirectory(util/exotic_operations)
add_subdirectory(util/common)
add_subdirectory(util/cuda)
add_subdirectory(util/snap-engine)

# Algorithms
add_subdirectory(algs/algorithm_bond)
add_subdirectory(algs/coherence)
add_subdirectory(algs/range_doppler_terrain_correction)
add_subdirectory(algs/backgeocoding)
add_subdirectory(algs/apply-orbit-file-op)
add_subdirectory(algs/sentinel1_calibrate)
add_subdirectory(algs/topsar-deburst-op)
add_subdirectory(algs/coherence_estimation_routine)
add_subdirectory(algs/coregistration)
add_subdirectory(algs/topsar_split)
add_subdirectory(algs/calibration_routine)

# Test helpers
add_subdirectory(test/range_doppler_geocoding)
add_subdirectory(test/tie_point_grid)
add_subdirectory(test/sar_geocoding)

# Main terminal application
add_subdirectory(app)

# Tests
message(STATUS "Enable testing: ${ENABLE_UNIT_TESTS}")
message(STATUS "Integration tests enabled: ${ENABLE_INTEGRATION_TESTS}")

include(FetchContent)

FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.10.0
)

FetchContent_GetProperties(googletest)

if (ENABLE_UNIT_TESTS OR ENABLE_INTEGRATION_TESTS)

    if (NOT googletest_POPULATED)
        FetchContent_Populate(googletest)

        # Prevent GoogleTest from overriding our compiler/linker options
        # when building with Visual Studio
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        # Prevent GoogleTest from using PThreads
        set(gtest_disable_pthreads ON CACHE BOOL "" FORCE)

        # adds the targers: gtest, gtest_main, gmock, gmock_main
        add_subdirectory(
                ${googletest_SOURCE_DIR}
                ${googletest_BINARY_DIR}
        )

        # Silence std::tr1 warning on MSVC
        if (MSVC)
            foreach (_tgt gtest gtest_main gmock gmock_main)
                target_compile_definitions(${_tgt}
                        PRIVATE
                        "_SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING"
                        )
            endforeach ()
        endif ()
    endif ()
endif ()

if (ENABLE_UNIT_TESTS)
    add_subdirectory(test)
endif ()

if (ENABLE_INTEGRATION_TESTS)
    add_subdirectory(test_integration)
endif ()
