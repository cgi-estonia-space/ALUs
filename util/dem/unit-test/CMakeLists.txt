
list(APPEND DEM_UTIL_TEST_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/dem_assistant.cc
        ${CMAKE_CURRENT_LIST_DIR}/copdem_cog_30m.cc
        ${CMAKE_CURRENT_LIST_DIR}/line_resample.cc
        )

add_executable(dem-util-test ${DEM_UTIL_TEST_SOURCES})
target_link_libraries(dem-util-test
        PRIVATE
        alus-gmock-main-static
        dem-util-static
        )
set_target_properties(dem-util-test
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${ALUS_UNIT_TEST_BINARY_OUTPUT_DIR}
        )
