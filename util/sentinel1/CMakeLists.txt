list(APPEND SENTINEL1_UTIL_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/Sentinel1Utils.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/SubSwathInfo.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/OrbitStateVectors.cpp
        )

list(APPEND SENTINEL1_UTIL_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/../snap-engine/include
        ${CMAKE_CURRENT_LIST_DIR}/../exotic_operations/include
        ${CMAKE_CURRENT_LIST_DIR}/../cuda/include
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
