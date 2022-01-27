list(APPEND SENTINEL_1_CALIBRATE_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/sentinel1_calibrate.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/sentinel1_calibrate_kernel.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/sentinel1_calibrate_executor.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/sentinel1_calibrate_kernel_utils.cu
        )

list(APPEND SENTINEL_1_CALIBRATE_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/../../sentinel1
        ${CMAKE_CURRENT_LIST_DIR}/../../snap-engine
        ${CMAKE_CURRENT_LIST_DIR}/../../snap-engine/snap-core/core/datamodel
        ${CMAKE_CURRENT_LIST_DIR}/../../snap-engine/snap-core/core/dataio
        ${CMAKE_CURRENT_LIST_DIR}/../../snap-engine/snap-engine-utilities/engine-utilities/datamodel/metadata
        ${CMAKE_CURRENT_LIST_DIR}/../../util/exotic-operations/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/common/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/cuda/include
        ${CMAKE_CURRENT_LIST_DIR}/../algorithm-bond/include
        ${CMAKE_CURRENT_LIST_DIR}/../../app/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/gdal/include
        ${CMAKE_CURRENT_LIST_DIR}/../../external/pugixml/include
        ${CMAKE_CURRENT_LIST_DIR}/../topsar-split/include
        )

add_library(sentinel1-calibrate-static
        STATIC
        ${SENTINEL_1_CALIBRATE_SOURCES}
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
        cuda-util-static
        common-util-static
        topsar-split-static
        )