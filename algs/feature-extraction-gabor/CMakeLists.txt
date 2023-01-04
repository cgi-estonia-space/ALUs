
list(APPEND FEATURE_EXTRACTION_GABOR_SOURCES
        src/execute.cc
        src/filter_bank.cc
        src/patch_assembly.cc
        src/patched_image.cc
        src/conv_kernel.cu
        src/patch_reduction.cu
        src/result_export.cc
        )

list(APPEND FEATURE_EXTRACTION_GABOR_INCLUDES
        ${CMAKE_CURRENT_LIST_DIR}/include)

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
        cuda-util-static
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
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/unit-test)
endif()
