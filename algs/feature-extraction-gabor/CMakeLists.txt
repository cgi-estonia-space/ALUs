
list(APPEND FEATURE_EXTRACTION_GABOR_SOURCES
        src/execute.cc
        src/filter_bank.cc
        src/patch_assembly.cc
        src/patched_image.cc
        src/conv_kernel.cu
        src/patch_reduction.cu
        )

list(APPEND FEATURE_EXTRACTION_GABOR_INCLUDES
        ../feature-extraction-gabor/include)

add_library(feature-extraction-gabor-static
        STATIC
        ${FEATURE_EXTRACTION_GABOR_SOURCES})

target_include_directories(feature-extraction-gabor-static
        PUBLIC
        ${FEATURE_EXTRACTION_GABOR_INCLUDES})

target_link_libraries(feature-extraction-gabor-static
        PUBLIC
        gdal
        app-util-static
        common-util-static
        gdal-util-static
        )

set_target_properties(feature-extraction-gabor-static
        PROPERTIES
        OUTPUT_NAME feature-extraction-gabor)

add_executable(alus-gfe src/main.cc src/command_line_options.cc)

target_link_libraries(alus-gfe
        PRIVATE
        feature-extraction-gabor-static
        Boost::program_options
        common-util-static
        )

set_target_properties(alus-gfe
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${ALUS_BINARY_OUTPUT_DIR}
        )


if(ENABLE_TESTS)
    add_subdirectory(../feature-extraction-gabor/unit-test)
endif()
