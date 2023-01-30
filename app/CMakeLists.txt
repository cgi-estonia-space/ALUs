list(APPEND APP_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/../util/common/include
        )

list(APPEND APP_UTIL_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/algorithm_parameters.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/app_utils.cc
        )
add_library(app-util-static STATIC ${APP_UTIL_SOURCES})
target_include_directories(app-util-static
        PUBLIC
        ${APP_INCLUDE_DIRS}
        )
target_link_libraries(app-util-static
        PUBLIC
        Boost::program_options
        )
set_target_properties(app-util-static
        PROPERTIES
        OUTPUT_NAME app-util
        )

if (ENABLE_TESTS)
    add_subdirectory(unit-test)
endif ()
