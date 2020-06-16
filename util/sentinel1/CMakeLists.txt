list(APPEND SENTINEL1_UTIL_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/sentinel1_utils.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/subswath_info.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/orbit_state_vectors.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/orbit_state_vectors.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/sar_geocoding.cu
        )

list(APPEND SENTINEL1_UTIL_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/../snap-engine/include
        ${CMAKE_CURRENT_LIST_DIR}/../exotic_operations/include
        ${CMAKE_CURRENT_LIST_DIR}/../cuda/include
        ${CMAKE_CURRENT_LIST_DIR}/../common/include
        )

add_library(sentinel1-util-obj
        OBJECT
        ${SENTINEL1_UTIL_SOURCES}
        )
target_include_directories(sentinel1-util-obj
        PRIVATE
        ${SENTINEL1_UTIL_INCLUDE_DIRS}
        )

add_library(sentinel1-util-static
        STATIC
        $<TARGET_OBJECTS:sentinel1-util-obj>
        )
target_include_directories(sentinel1-util-static
        PUBLIC
        ${SENTINEL1_UTIL_INCLUDE_DIRS}
        )
set_target_properties(sentinel1-util-static
        PROPERTIES
        OUTPUT_NAME sentinel1-util
        )
target_link_libraries(sentinel1-util-static
        PUBLIC
        gdal
        exotic_operations-static
        )
