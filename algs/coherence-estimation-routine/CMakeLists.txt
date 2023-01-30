list(APPEND COHERENCE_ESTIMATION_ROUTINE_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/cli_args.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/execute.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/main.cc
        )

list(APPEND COHERENCE_ESTIMATION_ROUTINE_INCLUDES
        ${CMAKE_CURRENT_LIST_DIR}/include
        )

add_executable(alus-coh ${COHERENCE_ESTIMATION_ROUTINE_SOURCES})
add_executable(alus-coht ${COHERENCE_ESTIMATION_ROUTINE_SOURCES})

target_include_directories(alus-coh
        PUBLIC
        ${COHERENCE_ESTIMATION_ROUTINE_INCLUDES}
        )

target_include_directories(alus-coht
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
        dem-util-static
        )

target_link_libraries(alus-coht
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
        dem-util-static
        )

target_compile_definitions(alus-coh PRIVATE COHERENCE_TIMELINE=0)
target_compile_definitions(alus-coht PRIVATE COHERENCE_TIMELINE=1)

set_target_properties(alus-coh
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${ALUS_BINARY_OUTPUT_DIR}
        )

set_target_properties(alus-coht
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${ALUS_BINARY_OUTPUT_DIR}
        )
