list(APPEND SAR_SEGMENT_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/cli_args.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/execute.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/main.cc
        )

list(APPEND SAR_SEGMENT_INCLUDES
        ${CMAKE_CURRENT_LIST_DIR}/include
        )

add_executable(alus-sar-segment ${SAR_SEGMENT_SOURCES})

target_include_directories(alus-sar-segment
        PRIVATE
        ${SAR_SEGMENT_INCLUDES}
        )

target_link_libraries(alus-sar-segment
        PUBLIC
        app-util-static
        dem-util-static
        gdal-util-static
        math-util-static
        snap-engine-static
        terrain-correction-static
        sentinel1-calibrate-static
        common-util-static
        thermal-noise-static
        )

set_target_properties(alus-sar-segment
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${ALUS_BINARY_OUTPUT_DIR}
        )

if (ENABLE_TESTS)
    add_subdirectory(unit-test)
endif ()
