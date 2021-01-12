
list(APPEND TERRAIN_CORRECTION_SOURCES_CU
        ${CMAKE_CURRENT_LIST_DIR}/src/get_position.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/local_dem.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/terrain_correction_kernel.cu
        )
list(APPEND TERRAIN_CORRECTION_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/execute.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/terrain_correction.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/terrain_correction_metadata.cc
        )

list(APPEND TERRAIN_CORRECTION_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/../algorithm_bond/include
        ${CMAKE_CURRENT_LIST_DIR}/../../app/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/common/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/cuda/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/gdal/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/sentinel1/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/snap-engine/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/snap-engine/include/snap-core/datamodel
        ${CMAKE_CURRENT_LIST_DIR}/../../util/exotic_operations/include
        ${CMAKE_CURRENT_LIST_DIR}/../../external/pugixml/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/multi_threading/include
        )

add_library(terrain-correction-obj-cu
        OBJECT
        ${TERRAIN_CORRECTION_SOURCES_CU}
        )
target_include_directories(terrain-correction-obj-cu
        PRIVATE
        ${TERRAIN_CORRECTION_INCLUDE_DIRS}
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
        $<TARGET_OBJECTS:terrain-correction-obj-cu>
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
        snap-engine-static
        multi_threading-static
        sentinel1-util-static
        pugixml-static
        )

add_library(terrain-correction-shared
        SHARED
        $<TARGET_OBJECTS:terrain-correction-obj>
        )
target_include_directories(terrain-correction-shared
        PUBLIC
        ${TERRAIN_CORRECTION_INCLUDE_DIRS}
        )
set_target_properties(terrain-correction-shared
        PROPERTIES
        OUTPUT_NAME terrain-correction
        LIBRARY_OUTPUT_DIRECTORY ${ALUS_BINARY_OUTPUT_DIR}
        )

target_link_libraries(terrain-correction-shared
        PUBLIC
        gdal-util-static
        snap-engine-static
        sentinel1-util-static
        multi_threading-static
        pugixml-static
        )

