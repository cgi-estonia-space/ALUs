
list(APPEND FEATURE_EXTRACTION_GABOR_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/execute.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/filter_bank.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/patch_assembly.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/patched_image.cc
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
        common-util-static
        gdal-util-static
        )

set_target_properties(feature-extraction-gabor-static
        PROPERTIES
        OUTPUT_NAME feature-extraction-gabor)

add_executable(alus-gfe ${CMAKE_CURRENT_LIST_DIR}/src/main.cc ${CMAKE_CURRENT_LIST_DIR}/src/command_line_options.cc)

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
    add_subdirectory(unit_test)
endif()