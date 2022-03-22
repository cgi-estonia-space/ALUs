list(APPEND RESAMPLE_UNIT_TEST_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/sentinel2_tools.cc
        ${CMAKE_CURRENT_LIST_DIR}/tyler_the_creator.cc
        )

add_executable(resample-unit-test ${RESAMPLE_UNIT_TEST_SOURCES})

target_link_libraries(resample-unit-test
        PRIVATE
        alus-gmock-main-static
        resample-static
        )

set_target_properties(resample-unit-test
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${ALUS_UNIT_TEST_BINARY_OUTPUT_DIR}
        )
