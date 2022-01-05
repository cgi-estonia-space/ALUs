list(APPEND SAR_GEOCODING_TEST_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/sar_geocoding_test.cu
        )

list(APPEND SAR_GEOCODING_TEST_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/../../snap-engine
        ${CMAKE_CURRENT_LIST_DIR}/../../sentinel1
        ${CMAKE_CURRENT_LIST_DIR}/../../util/cuda/include
        )

add_library(sar-geocoding-test-obj
        OBJECT
        ${SAR_GEOCODING_TEST_SOURCES}
        )
target_include_directories(sar-geocoding-test-obj
        PRIVATE
        ${SAR_GEOCODING_TEST_INCLUDE_DIRS}
        )

add_library(sar-geocoding-test-static
        STATIC
        $<TARGET_OBJECTS:sar-geocoding-test-obj>
        )
target_include_directories(sar-geocoding-test-static
        PUBLIC
        ${SAR_GEOCODING_TEST_INCLUDE_DIRS}
        )
set_target_properties(sar-geocoding-test-static
        PROPERTIES
        OUTPUT_NAME sar-geocoding-test
        )
target_link_libraries(sar-geocoding-test-static
        PUBLIC
        sentinel1-util-static
        )