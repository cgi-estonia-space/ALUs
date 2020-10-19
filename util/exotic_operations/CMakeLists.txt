list(APPEND EXOTIC_OPERATIONS_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/comparators.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/delaunay_triangulator.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/delaunay_triangulator.cu
        ${CMAKE_CURRENT_LIST_DIR}/../../external/delaunay/src/delaunay_triangulator_cpu.cc
        ${CMAKE_CURRENT_LIST_DIR}/../../external/delaunay/src/snap_delaunay_triangulator.cc
        )

list(APPEND EXOTIC_OPERATIONS_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/../cuda/include
        ${CMAKE_CURRENT_LIST_DIR}/../common/include
        ${CMAKE_CURRENT_LIST_DIR}/../../external/delaunay/include
        )

add_library(exotic_operations-obj
        OBJECT
        ${EXOTIC_OPERATIONS_SOURCES}
        )
target_include_directories(exotic_operations-obj
        PRIVATE
        ${EXOTIC_OPERATIONS_INCLUDE_DIRS}
        )

add_library(exotic_operations-static
        STATIC
        $<TARGET_OBJECTS:exotic_operations-obj>
        )
target_include_directories(exotic_operations-static
        PUBLIC
        ${EXOTIC_OPERATIONS_INCLUDE_DIRS}
        )
set_target_properties(exotic_operations-static
        PROPERTIES
        OUTPUT_NAME exotic_operations
        )
target_link_libraries(exotic_operations-static
        PUBLIC
        gdal
        cuda_util-static
        )
