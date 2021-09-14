list(APPEND COHERENCE_ESTIMATION_ROUTINE_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/coherence_estimation_routine_execute.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/safe_execute.cc
        )

list(APPEND COHERENCE_ESTIMATION_ROUTINE_INCLUDES
        ${CMAKE_CURRENT_LIST_DIR}/include
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
        coherence-cuda-static
        gdal-util-static
        snap-engine-static
        topsar-deburst-op-static
        terrain-correction-static
        common-util-static
        )

set_target_properties(coherence-estimation-routine-shared
        PROPERTIES
        OUTPUT_NAME coherence-estimation-routine
        LIBRARY_OUTPUT_DIRECTORY ${ALUS_BINARY_OUTPUT_DIR}
        )
