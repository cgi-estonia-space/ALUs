list(APPEND TOPSAR_SPLIT_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/topsar_split.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/aoi_burst_extract.cc
        )

list(APPEND TOPSAR_SPLIT_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/../algorithm_bond/include
        ${CMAKE_CURRENT_LIST_DIR}/../../app/include
        ${CMAKE_CURRENT_LIST_DIR}/../../sentinel1
        ${CMAKE_CURRENT_LIST_DIR}/../../snap-engine
        ${CMAKE_CURRENT_LIST_DIR}/../../util/gdal/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/common/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/cuda/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/exotic_operations/include
        ${CMAKE_CURRENT_LIST_DIR}/../../external/pugixml/include
        )

add_library(topsar-split-obj
        OBJECT
        ${TOPSAR_SPLIT_SOURCES}
        )
target_include_directories(topsar-split-obj
        PRIVATE
        ${TOPSAR_SPLIT_INCLUDE_DIRS}
        )

add_library(topsar-split-static
        STATIC
        $<TARGET_OBJECTS:topsar-split-obj>
        )
target_include_directories(topsar-split-static
        PUBLIC
        ${TOPSAR_SPLIT_INCLUDE_DIRS}
        )
set_target_properties(topsar-split-static
        PROPERTIES
        OUTPUT_NAME topsar-split
        )
target_link_libraries(topsar-split-static
        PUBLIC
        gdal-util-static
        exotic_operations-static
        sentinel1-util-static
        snap-engine-static
        Boost::program_options
        common-util-static
        )

add_library(topsar-split-shared
        SHARED
        $<TARGET_OBJECTS:topsar-split-obj>
        )
target_include_directories(topsar-split-shared
        PUBLIC
        ${TOPSAR_SPLIT_INCLUDE_DIRS}
        )
set_target_properties(topsar-split-shared
        PROPERTIES
        OUTPUT_NAME topsar-split
        LIBRARY_OUTPUT_DIRECTORY ${ALUS_BINARY_OUTPUT_DIR}
        )

target_link_libraries(topsar-split-shared
        PUBLIC
        exotic_operations-static
        sentinel1-util-static
        snap-engine-static
        Boost::program_options
        common-util-static
        )
