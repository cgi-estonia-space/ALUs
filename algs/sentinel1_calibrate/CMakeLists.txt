list(APPEND SENTINEL_1_CALIBRATE_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/sentinel1_calibrate.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/sentinel1_calibrate_kernel.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/sentinel1_calibrate_executor.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/sentinel1_calibrate_kernel_utils.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/sentinel1_calibrate_safe_helper.cc
        )

list(APPEND SENTINEL_1_CALIBRATE_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/sentinel1/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/snap-engine/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/snap-engine/include/snap-core/datamodel
        ${CMAKE_CURRENT_LIST_DIR}/../../util/snap-engine/include/snap-core/dataio
        ${CMAKE_CURRENT_LIST_DIR}/../../util/snap-engine/include/snap-engine-utilities/datamodel/metadata
        ${CMAKE_CURRENT_LIST_DIR}/../../util/exotic_operations/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/common/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/cuda/include
        ${CMAKE_CURRENT_LIST_DIR}/../algorithm_bond/include
        ${CMAKE_CURRENT_LIST_DIR}/../../app/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/gdal/include
        ${CMAKE_CURRENT_LIST_DIR}/../../external/pugixml/include
        )

add_library(sentinel1-calibrate-obj
        OBJECT
        ${SENTINEL_1_CALIBRATE_SOURCES})

target_include_directories(sentinel1-calibrate-obj
        PRIVATE
        ${SENTINEL_1_CALIBRATE_INCLUDE_DIRS}
        )

add_library(sentinel1-calibrate-static
        STATIC
        $<TARGET_OBJECTS:sentinel1-calibrate-obj>
        )

target_include_directories(sentinel1-calibrate-static
        PUBLIC
        ${SENTINEL_1_CALIBRATE_INCLUDE_DIRS}
        )

set_target_properties(sentinel1-calibrate-static
        PROPERTIES
        OUTPUT_NAME sentinel1-calibrate
        )

target_link_libraries(sentinel1-calibrate-static
        PUBLIC
        Boost::filesystem
        Boost::iostreams
        sentinel1-util-static
        snap-engine-static
        gdal-util-static
        cuda_util-static
        )

add_library(sentinel1-calibrate-shared
        SHARED
        $<TARGET_OBJECTS:sentinel1-calibrate-obj>
        ${CMAKE_CURRENT_LIST_DIR}/src/sentinel1_calibrate_executor_shared.cc
        )

target_include_directories(sentinel1-calibrate-shared
        PUBLIC
        ${SENTINEL_1_CALIBRATE_INCLUDE_DIRS}
        )

set_target_properties(sentinel1-calibrate-shared
        PROPERTIES
        OUTPUT_NAME sentinel1-calibrate
        LIBRARY_OUTPUT_DIRECTORY ${ALUS_BINARY_OUTPUT_DIR}
        )

target_link_libraries(sentinel1-calibrate-shared
        PUBLIC
        Boost::filesystem
        Boost::iostreams
        sentinel1-util-static
        snap-engine-static
        gdal-util-static
        cuda_util-static
        )