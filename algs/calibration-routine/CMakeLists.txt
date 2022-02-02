list(APPEND CALIBRATION_ROUTINE_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/cli_args.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/execute.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/main.cc
        )

list(APPEND CALIBRATION_ROUTINE_INCLUDES
        ${CMAKE_CURRENT_LIST_DIR}/include
        )

add_executable(alus-cal ${CALIBRATION_ROUTINE_SOURCES})

target_include_directories(alus-cal
        PRIVATE
        ${CALIBRATION_ROUTINE_INCLUDES}
        )

target_link_libraries(alus-cal
        PUBLIC
        app-util-static
        gdal-util-static
        snap-engine-static
        terrain-correction-static
        sentinel1-calibrate-static
        topsar-deburst-op-static
        topsar-split-static
        common-util-static
        )

set_target_properties(alus-cal
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${ALUS_BINARY_OUTPUT_DIR}
        )
