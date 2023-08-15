list(APPEND ASAR_FOCUS_SOURCES
        src/main.cc
        src/doris_orbit.cc
        src/orbit_util.cc)

list(APPEND ASAR_FOCUS_INCLUDES
        ${CMAKE_CURRENT_LIST_DIR}/include
        )


add_executable(alus-asar-focus ${ASAR_FOCUS_SOURCES})

target_include_directories(alus-asar-focus
        PUBLIC
        ${ASAR_FOCUS_INCLUDES}
        )

target_link_libraries(alus-asar-focus
        PUBLIC
        Boost::boost
        Boost::date_time
        )

set_target_properties(alus-asar-focus
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${ALUS_BINARY_OUTPUT_DIR}
        )
