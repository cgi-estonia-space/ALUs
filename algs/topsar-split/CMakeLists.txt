list(APPEND TOPSAR_SPLIT_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/topsar_split.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/aoi_burst_extract.cc
        )

list(APPEND TOPSAR_SPLIT_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/../../sentinel1
        ${CMAKE_CURRENT_LIST_DIR}/../../snap-engine
        ${CMAKE_CURRENT_LIST_DIR}/../../external/pugixml/include
        )

add_library(topsar-split-static
        STATIC
        ${TOPSAR_SPLIT_SOURCES}
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
        cuda-util-static
        gdal-util-static
        exotic-operations-static
        sentinel1-util-static
        snap-engine-static
        common-util-static
        staticZipper
        )
