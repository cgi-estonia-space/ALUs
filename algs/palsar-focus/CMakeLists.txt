list(APPEND PALSAR_FOCUS_SOURCES
        src/palsar_file.cc
        src/palsar_metadata.cc
        src/palsar_record.cc
        src/chirp.cc
        src/cufft_plan.cc
        src/binary_output.cc
        src/cli_args.cc
        src/img_output.cc
        src/processing_velocity_estimate.cc
        src/dc_bias.cu
        src/doppler_centroid.cu
        src/device_padded_image.cu
        src/range_compression.cu
        src/range_doppler_algorithm.cu
        src/main.cc
        )


list(APPEND PALSAR_FOCUS_INCLUDES
        ${CMAKE_CURRENT_LIST_DIR}/include
        )


add_executable(alus-palsar-focus ${PALSAR_FOCUS_SOURCES})

target_include_directories(alus-palsar-focus
        PUBLIC
        ${PALSAR_FOCUS_INCLUDES}
        )

target_link_libraries(alus-palsar-focus
        PUBLIC
        app-util-static
        common-util-static
        cufft
        )

set_target_properties(alus-palsar-focus
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${ALUS_BINARY_OUTPUT_DIR}
        )