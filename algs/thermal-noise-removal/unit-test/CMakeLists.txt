list(APPEND REMOVE_THERMAL_NOISE_UNIT_TEST_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/thermal_noise_utils_tests.cc
        ${CMAKE_CURRENT_LIST_DIR}/thermal_noise_kernel_tests.cc
        )

list(APPEND REMOVE_THERMAL_NOISE_UNIT_TEST_INCLUDES
        ${CMAKE_CURRENT_LIST_DIR}/test_utils.h
        ${CMAKE_CURRENT_LIST_DIR}/test_constants.h
        ${CMAKE_CURRENT_LIST_DIR}/test_expected_values.h
        )

add_executable(remove-thermal-noise-unit-test ${REMOVE_THERMAL_NOISE_UNIT_TEST_SOURCES} ${REMOVE_THERMAL_NOISE_UNIT_TEST_INCLUDES})

target_link_libraries(remove-thermal-noise-unit-test
        PRIVATE
        thermal-noise-static
        sentinel1-util-static
        alus-gmock-main-static
        snap-engine-static
        )

set_target_properties(remove-thermal-noise-unit-test
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${ALUS_UNIT_TEST_BINARY_OUTPUT_DIR}
        )