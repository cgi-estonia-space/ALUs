list(APPEND TERRAIN_CORRECTION_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/get_position.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/terrain_correction.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/terrain_correction_kernel.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/terrain_correction_metadata.cc
        )

list(APPEND TERRAIN_CORRECTION_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include
        )

add_library(terrain-correction-static
        STATIC
        ${TERRAIN_CORRECTION_SOURCES}
        )
target_include_directories(terrain-correction-static
        PUBLIC
        ${TERRAIN_CORRECTION_INCLUDE_DIRS}
        )
set_target_properties(terrain-correction-static
        PROPERTIES
        OUTPUT_NAME terrain-correction
        )
target_link_libraries(terrain-correction-static
        PUBLIC
        cuda-util-static
        common-util-static
        exotic-operations-static
        gdal-util-static
        snap-engine-static
        sentinel1-util-static
        )
