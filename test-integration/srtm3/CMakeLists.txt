list(APPEND SRTM3_TEST_UTIL_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/srtm3_test_util.cu
        )

list(APPEND SRTM3_TEST_UTIL_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/../../snap-engine
        ${CMAKE_CURRENT_LIST_DIR}/../../util/exotic-operations/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/cuda/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/common/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/dem/include
        )

add_library(srtm3-test-util-obj
        OBJECT
        ${SRTM3_TEST_UTIL_SOURCES}
        )
target_include_directories(srtm3-test-util-obj
        PRIVATE
        ${SRTM3_TEST_UTIL_INCLUDE_DIRS}
        )

add_library(srtm3-test-util-static
        STATIC
        $<TARGET_OBJECTS:srtm3-test-util-obj>
        )
target_include_directories(srtm3-test-util-static
        PUBLIC
        ${SRTM3_TEST_UTIL_INCLUDE_DIRS}
        )
set_target_properties(srtm3-test-util-static
        PROPERTIES
        OUTPUT_NAME srtm3-test-util
        )
target_link_libraries(srtm3-test-util-static
        PUBLIC
        sentinel1-util-static
        )
