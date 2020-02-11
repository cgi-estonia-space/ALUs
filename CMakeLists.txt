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

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Dependencies which should be installed on a host system and not included separately by this project.
find_package(CUDA REQUIRED)
message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")

find_package(GDAL REQUIRED)

# Algorithms
add_subdirectory(algs/inverter)

# Main terminal application
add_subdirectory(app)


#list(APPEND aluserer_SOURCES
#    ${CMAKE_CURRENT_LIST_DIR}/main.cpp
#    )
#
#list(APPEND aluserer_INCLUDES )
#
#add_executable(aluserer ${aluserer_SOURCES})

#include(${CMAKE_CURRENT_LIST_DIR}/src/main/CMakeLists.txt)
#include(${CMAKE_CURRENT_LIST_DIR}/src/test/CMakeLists.txt)

#cmake_print_variables(${Utils_DEP_INCLUDES})

#list(APPEND Utils_DEP_INCLUDES
        #${CMAKE_CURRENT_LIST_DIR}/target/test-nar/gmock-1.7.0-noarch/include
        #${CMAKE_CURRENT_LIST_DIR}/target/test-nar/gtest-1.7.0-noarch/include
        #${CMAKE_CURRENT_LIST_DIR}/target/test-nar/gtest-extra-1.0.1-SNAPSHOT-noarch/include
        #        )

#cmake_print_variables(${Utils_DEP_INCLUDES})

#add_executable(${CMAKE_PROJECT_NAME}-test ${Utils_TEST_SOURCES})
#target_include_directories(${CMAKE_PROJECT_NAME}-test PRIVATE ${Utils_INCLUDES} ${Utils_DEP_INCLUDES})

#Some boost include paths needed for gtest-extra.
#list(APPEND LINK_DIRS
       # ${CMAKE_CURRENT_LIST_DIR}/target/test-nar/boost-1.0.1-SNAPSHOT-amd64-Linux-gpp-shared/lib/amd64-Linux-gpp/shared/
       #       ${CMAKE_CURRENT_LIST_DIR}/target/test-nar/gtest-extra-1.0.1-SNAPSHOT-amd64-Linux-gpp-shared/lib/amd64-Linux-gpp/shared
        #${CMAKE_CURRENT_LIST_DIR}/target/test-nar/gtest-1.7.0-amd64-Linux-gpp-shared/lib/amd64-Linux-gpp/shared
        #$${CMAKE_CURRENT_LIST_DIR}/target/test-nar/gmock-1.7.0-amd64-Linux-gpp-shared/lib/amd64-Linux-gpp/shared
        #${CMAKE_CURRENT_LIST_DIR}/target/test-nar/regex-1.57.0+nar.10-amd64-Linux-gpp-shared/lib/amd64-Linux-gpp/shared
        #${CMAKE_CURRENT_LIST_DIR}/target/test-nar/exception-1.57.0+nar.10-amd64-Linux-gpp-shared/lib/amd64-Linux-gpp/shared
        #${CMAKE_CURRENT_LIST_DIR}/target/test-nar/smart-ptr-1.57.0+nar.10-amd64-Linux-gpp-shared/lib/amd64-Linux-gpp/shared
        #${CMAKE_CURRENT_LIST_DIR}/target/test-nar/integer-1.57.0+nar.10-amd64-Linux-gpp-shared/lib/amd64-Linux-gpp/shared
        #)

        #list(APPEND LINK_LIBS
        # gtest-extra-1.0.1-SNAPSHOT
        #gmock-1.7.0
        #gtest
        #gmock
        #)

        #target_link_directories(${CMAKE_PROJECT_NAME}-test PRIVATE ${LINK_DIRS})
        #target_link_libraries(${CMAKE_PROJECT_NAME}-test
        #PUBLIC
        #Boost::filesystem Boost::regex utils-static ${LINK_LIBS}
        #)

        #add_test(
        #NAME ${CMAKE_PROJECT_NAME}_test
        #COMMAND $<TARGET_FILE:${CMAKE_PROJECT_NAME}-test>
        #)


