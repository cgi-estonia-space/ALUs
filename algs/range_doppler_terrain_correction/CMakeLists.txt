
list(APPEND TERRAIN_CORRECTION_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/get_position.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/local_dem.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/terrain_correction.cpp
        )

list(APPEND TERRAIN_CORRECTION_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/common/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/cuda/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/gdal/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/sentinel1/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/snap-engine/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/exotic_operations/include

        )

add_library(terrain-correction-obj
        OBJECT
        ${TERRAIN_CORRECTION_SOURCES}
        )
target_include_directories(terrain-correction-obj
        PRIVATE
        ${TERRAIN_CORRECTION_INCLUDE_DIRS}
        )

add_library(terrain-correction-static
        STATIC
        $<TARGET_OBJECTS:terrain-correction-obj>
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
        gdal-util-static
        )

