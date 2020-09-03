
list(APPEND APP_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/main.cc)

list(APPEND APP_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/../algs/algorithm_bond/include
        ${CMAKE_CURRENT_LIST_DIR}/../util/common/include
        )

add_executable(aluserer ${APP_SOURCES})

set_target_properties(aluserer
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${ALUS_BINARY_OUTPUT_DIR}
        )

target_include_directories(aluserer PUBLIC ${APP_INCLUDE_DIRS})
target_link_libraries(aluserer PUBLIC dl ${Boost_PROGRAM_OPTIONS_LIBRARY})
