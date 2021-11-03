list(APPEND TOPSAR_MERGE_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/topsar_merge.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/topsar_merge_utils.cc
        )

LIST(APPEND TOPSAR_MERGE_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include
        )

add_library(topsar-merge-static
        STATIC
        ${TOPSAR_MERGE_SOURCES}
        )

target_include_directories(topsar-merge-static
        PUBLIC
        ${TOPSAR_MERGE_INCLUDE_DIRS}
        )

set_target_properties(topsar-merge-static
        PROPERTIES
        OUTPUT_NAME topsar-merge
        )

target_link_libraries(topsar-merge-static
        PUBLIC
        sentinel1-util-static
        snap-engine-static
        common-util-static
)

