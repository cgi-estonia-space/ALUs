
list(APPEND MATH_UTIL_TEST_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/interpolations.cc
        ${CMAKE_CURRENT_LIST_DIR}/polynomials.cc
        )

add_executable(math-util-test ${MATH_UTIL_TEST_SOURCES})
target_link_libraries(math-util-test
        PRIVATE
        alus-gmock-main-static
        math-util-static
        )
set_target_properties(math-util-test
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${ALUS_UNIT_TEST_BINARY_OUTPUT_DIR}
        )
