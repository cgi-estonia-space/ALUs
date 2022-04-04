list(APPEND COHERENCE_ESTIMATION_ROUTINE_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/cli_args.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/execute.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/main.cc
        )

list(APPEND COHERENCE_ESTIMATION_ROUTINE_INCLUDES
        ${CMAKE_CURRENT_LIST_DIR}/include
        )

add_executable(alus-coh ${COHERENCE_ESTIMATION_ROUTINE_SOURCES})

target_include_directories(alus-coh
        PUBLIC
        ${COHERENCE_ESTIMATION_ROUTINE_INCLUDES}
        )

target_link_libraries(alus-coh
        PUBLIC
        app-util-static
        apply-orbit-file-op-static
        coregistration-static
        backgeocoding-static
        coherence-cuda-static
        gdal-util-static
        snap-engine-static
        topsar-deburst-op-static
        topsar-merge-static
        terrain-correction-static
        common-util-static
        )

set_target_properties(alus-coh
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${ALUS_BINARY_OUTPUT_DIR}
        )
