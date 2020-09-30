
list(APPEND TEST_SOURCES_TENSORFLOW
        src/coherence_test.cc
        )

list(APPEND TEST_SOURCES
        src/terrain_correction_test.cc
        src/terrain_correction_test.cu
        )

list(APPEND TEST_INCLUDE
        ${CMAKE_CURRENT_LIST_DIR}/include
        )

add_executable(integration_tests_tensorflow ${TEST_SOURCES_TENSORFLOW})
target_include_directories(integration_tests_tensorflow PUBLIC ${TEST_INCLUDE})
target_link_libraries(integration_tests_tensorflow
        PRIVATE
        Boost::filesystem
        Boost::iostreams
        gtest_main
        cuda_util-static
        gdal-util-static
        sentinel1-util-static
        coherence-static
        )

target_compile_options(integration_tests_tensorflow
        PRIVATE
        -Wno-pedantic
        -Wno-unused-parameter
        -Wno-ignored-qualifiers
        -Wno-unused-variable
        )

add_custom_command(TARGET integration_tests_tensorflow
        POST_BUILD
        COMMAND cp -r ${CMAKE_CURRENT_LIST_DIR}/goods/coherence ${CMAKE_BINARY_DIR}/test_integration/.
        )

add_executable(integration_tests ${TEST_SOURCES})
target_include_directories(integration_tests PUBLIC ${TEST_INCLUDE})
target_link_libraries(integration_tests
        PRIVATE
        Boost::filesystem
        Boost::iostreams
        gtest_main
        cuda_util-static
        gdal-util-static
        sentinel1-util-static
        terrain-correction-static
        )

add_custom_command(TARGET integration_tests
        POST_BUILD
        COMMAND cp -r ${CMAKE_CURRENT_LIST_DIR}/goods ${CMAKE_BINARY_DIR}/test_integration/.
        COMMAND cd ${CMAKE_BINARY_DIR}/test_integration/goods && unzip -oq ${CMAKE_BINARY_DIR}/test_integration/goods/\\*.zip
        )

