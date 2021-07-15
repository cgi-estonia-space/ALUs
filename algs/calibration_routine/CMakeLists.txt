list(APPEND CALIBRATION_ROUTINE_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/calibration_routine.cc
        )

list(APPEND CALIBRATION_ROUTINE_INCLUDES
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/../algorithm_bond/include
        ${CMAKE_CURRENT_LIST_DIR}/../range_doppler_terrain_correction/include
        ${CMAKE_CURRENT_LIST_DIR}/../../app/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/exotic_operations/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/common/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/cuda/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/gdal/include
        ${CMAKE_CURRENT_LIST_DIR}/../../snap-engine/include
        )


add_library(calibration-routine-shared
        SHARED
        ${CALIBRATION_ROUTINE_SOURCES}
        )

target_include_directories(calibration-routine-shared
        PUBLIC
        ${CALIBRATION_ROUTINE_INCLUDES}
        )

target_link_libraries(calibration-routine-shared
        PUBLIC
        alg-bond-static
        gdal-util-static
        snap-engine-static
        terrain-correction-static
        sentinel1-calibrate-static
        topsar-deburst-op-static
        topsar-split-static
        common-util-static
        )

set_target_properties(calibration-routine-shared
        PROPERTIES
        OUTPUT_NAME calibration-routine
        LIBRARY_OUTPUT_DIRECTORY ${ALUS_BINARY_OUTPUT_DIR}
        )
