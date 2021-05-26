list(APPEND COHERENCE_ESTIMATION_ROUTINE_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/coherence_estimation_routine_execute.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/geotiff_and_beam_dimap_execute.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/safe_execute.cc
        )

list(APPEND COHERENCE_ESTIMATION_ROUTINE_INCLUDES
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/../../app/include
        )

add_library(coherence-estimation-routine-static
        STATIC
        ${COHERENCE_ESTIMATION_ROUTINE_SOURCES}
        )

target_include_directories(coherence-estimation-routine-static
        PUBLIC
        ${COHERENCE_ESTIMATION_ROUTINE_INCLUDES}
        )

target_link_libraries(coherence-estimation-routine-static
        PUBLIC
        alg-bond-static
        apply-orbit-file-op-static
        coregistration-static
        backgeocoding-static
        coherence-static
        gdal-util-static
        snap-engine-static
        topsar-deburst-op-static
        terrain-correction-static
        )

set_target_properties(coherence-estimation-routine-static
        PROPERTIES
        OUTPUT_NAME coherence_estimation_routine
        )

add_library(coherence-estimation-routine-shared
        SHARED
        ${COHERENCE_ESTIMATION_ROUTINE_SOURCES}
        )

target_include_directories(coherence-estimation-routine-shared
        PUBLIC
        ${COHERENCE_ESTIMATION_ROUTINE_INCLUDES}
        )

target_link_libraries(coherence-estimation-routine-shared
        PUBLIC
        alg-bond-static
        apply-orbit-file-op-static
        coregistration-static
        backgeocoding-static
        coherence-static
        gdal-util-static
        snap-engine-static
        topsar-deburst-op-static
        terrain-correction-static
        )

set_target_properties(coherence-estimation-routine-shared
        PROPERTIES
        OUTPUT_NAME coherence-estimation-routine
        LIBRARY_OUTPUT_DIRECTORY ${ALUS_BINARY_OUTPUT_DIR}
        )
