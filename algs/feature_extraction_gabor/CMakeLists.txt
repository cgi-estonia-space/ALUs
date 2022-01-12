
list(APPEND FEATURE_EXTRACTION_GABOR_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/filter_bank.cc)

list(APPEND FEATURE_EXTRACTION_GABOR_INCLUDES
        ${CMAKE_CURRENT_LIST_DIR}/include)

add_library(feature-extraction-gabor-static
        STATIC
        ${FEATURE_EXTRACTION_GABOR_SOURCES})

target_include_directories(feature-extraction-gabor-static
        PUBLIC
        ${FEATURE_EXTRACTION_GABOR_INCLUDES})

set_target_properties(feature-extraction-gabor-static
        PROPERTIES
        OUTPUT_NAME feature-extraction-gabor)


if(ENABLE_TESTS)
    add_subdirectory(unit_test)
endif()