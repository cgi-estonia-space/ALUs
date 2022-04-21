
list(APPEND APP_UNIT_TEST_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/algorithm_parameters.cc
        ${CMAKE_CURRENT_LIST_DIR}/dem_assistant.cc
        )

add_executable(app-unit-test ${APP_UNIT_TEST_SOURCES})
target_link_libraries(app-unit-test
        PRIVATE
        stdc++fs
        gmock_main
        app-util-static
        )
set_target_properties(app-unit-test
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${ALUS_UNIT_TEST_BINARY_OUTPUT_DIR}
        )
