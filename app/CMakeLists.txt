
list(APPEND APP_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/main.cc)

list(APPEND APP_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/../algs/algorithm_bond/include
        ${CMAKE_CURRENT_LIST_DIR}/../util/common/include
        )

add_executable(alus ${APP_SOURCES})

set_target_properties(alus
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${ALUS_BINARY_OUTPUT_DIR}
        )

target_include_directories(alus PUBLIC ${APP_INCLUDE_DIRS})
target_link_libraries(alus
        PUBLIC
        dl
        Boost::program_options
        app-util-static
        )

list(APPEND APP_UTIL_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/algorithm_parameters.cc)
add_library(app-util-static STATIC ${APP_UTIL_SOURCES})
target_include_directories(app-util-static
        PUBLIC
        ${APP_INCLUDE_DIRS}
        )
set_target_properties(app-util-static
        PROPERTIES
        OUTPUT_NAME app-util
        )

if(ENABLE_UNIT_TESTS)
        add_subdirectory(unit_test)
endif()