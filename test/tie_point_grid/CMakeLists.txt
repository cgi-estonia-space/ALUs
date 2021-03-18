list(APPEND TIE_POINT_GRID_TEST_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/tie_point_grid_test.cu
        )

list(APPEND TIE_POINT_GRID_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/snap-engine/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/cuda/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/common/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/snap-engine/include/snap-core/datamodel
        )

add_library(tiePointGrid-test-obj
        OBJECT
        ${TIE_POINT_GRID_TEST_SOURCES}
        )
target_include_directories(tiePointGrid-test-obj
        PRIVATE
        ${TIE_POINT_GRID_INCLUDE_DIRS}
        )

add_library(tiePointGrid-test-static
        STATIC
        $<TARGET_OBJECTS:tiePointGrid-test-obj>
        )
target_include_directories(tiePointGrid-test-static
        PUBLIC
        ${TIE_POINT_GRID_INCLUDE_DIRS}
        )

target_link_libraries(tiePointGrid-test-static
        PUBLIC
        snap-engine-static
        )

set_target_properties(tiePointGrid-test-static
        PROPERTIES
        OUTPUT_NAME tiePointGrid-test
        )