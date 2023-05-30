
list(APPEND COMMON_UTIL_TEST_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/product_name.cc
        )

add_executable(common-util-test ${COMMON_UTIL_TEST_SOURCES})
target_link_libraries(common-util-test
        PRIVATE
        alus-gmock-main-static
        common-util-static
        )
set_target_properties(common-util-test
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${ALUS_UNIT_TEST_BINARY_OUTPUT_DIR}
        )
