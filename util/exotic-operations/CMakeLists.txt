list(APPEND EXOTIC_OPERATIONS_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/comparators.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/delaunay_triangulator.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/delaunay_triangulator.cu
        ${CMAKE_CURRENT_LIST_DIR}/../../external/delaunay/src/delaunay_triangulator_cpu.cc
        ${CMAKE_CURRENT_LIST_DIR}/../../external/delaunay/src/snap_delaunay_triangulator.cc
        )

list(APPEND EXOTIC_OPERATIONS_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/../../external/delaunay/include
        )

add_library(exotic-operations-static
        STATIC
        ${EXOTIC_OPERATIONS_SOURCES}
        )
target_include_directories(exotic-operations-static
        PUBLIC
        ${EXOTIC_OPERATIONS_INCLUDE_DIRS}
        )
set_target_properties(exotic-operations-static
        PROPERTIES
        OUTPUT_NAME exotic-operations
        )
target_link_libraries(exotic-operations-static
        PUBLIC
        gdal
        cuda-util-static
        common-util-static
        )
