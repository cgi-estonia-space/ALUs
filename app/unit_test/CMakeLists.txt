
list(APPEND APP_UNIT_TEST_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/algorithm_parameters.cc
        )

list(APPEND APP_UNIT_TEST_INCLUDES
        ${CMAKE_CURRENT_LIST_DIR}/../include
        )

add_executable(app_unit_test ${APP_UNIT_TEST_SOURCES})
target_include_directories(app_unit_test
        PRIVATE
        ${APP_UNIT_TEST_INCLUDES}
        )
target_link_libraries(app_unit_test
        PRIVATE
        stdc++fs
        gmock_main
        app-util-static
        )
set_target_properties(app_unit_test
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${ALUS_UNIT_TEST_BINARY_OUTPUT_DIR}
        )
