list(APPEND COREGISTRATION_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/coregistration_controller.cc
        )

list(APPEND COREGISTRATION_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include
        )

add_library(coregistration-static
        STATIC
        ${COREGISTRATION_SOURCES}
        )
target_include_directories(coregistration-static
        PUBLIC
        ${COREGISTRATION_INCLUDE_DIRS}
        )
set_target_properties(coregistration-static
        PROPERTIES
        OUTPUT_NAME coregistration
        )
target_link_libraries(coregistration-static
        PUBLIC
        gdal-util-static
        common-util-static
        cuda-util-static
        backgeocoding-static
        topsar-split-static
        apply-orbit-file-op-static
        exotic-operations-static
        sentinel1-util-static
        snap-engine-static
        )