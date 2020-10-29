list(APPEND SENTINEL1_UTIL_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/sentinel1_utils.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/subswath_info.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/point.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/sar_utils.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/orbit_state_vectors.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/sar_geocoding.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/poly_utils.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/orbit.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/orbit_state_vectors.cc
        )

list(APPEND SENTINEL1_UTIL_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/../cuda/include
        ${CMAKE_CURRENT_LIST_DIR}/../common/include
        )

add_library(sentinel1-util-static
        STATIC
        ${SENTINEL1_UTIL_SOURCES}
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
        snap-engine-static
        exotic_operations-static
        Eigen3::Eigen
        )
