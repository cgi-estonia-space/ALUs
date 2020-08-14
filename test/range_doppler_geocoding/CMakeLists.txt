list(APPEND RANGE_DOPPLER_GEOCODING_TEST_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/range_doppler_geocoding_test.cu
        )

list(APPEND RANGE_DOPPLER_GEOCODING_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/../../algs/range_doppler_terrain_correction/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/cuda/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/snap-engine/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/common/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/exotic_operations/include
        )

add_library(range-doppler-geocoding-test-obj
        OBJECT
        ${RANGE_DOPPLER_GEOCODING_TEST_SOURCES}
        )
target_include_directories(range-doppler-geocoding-test-obj
        PRIVATE
        ${RANGE_DOPPLER_GEOCODING_INCLUDE_DIRS}
        )

add_library(range-doppler-geocoding-test-static
        STATIC
        $<TARGET_OBJECTS:range-doppler-geocoding-test-obj>
        )
target_include_directories(range-doppler-geocoding-test-static
        PUBLIC
        ${RANGE_DOPPLER_GEOCODING_INCLUDE_DIRS}
        )

set_target_properties(range-doppler-geocoding-test-static
        PROPERTIES
        OUTPUT_NAME range-doppler-geocoding-test
        )