
list(APPEND TEST_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include
        )

list(APPEND TEST_SOURCES
        src/alus_gmock_main.cc
        src/dataset_test.cc
        src/terrain_correction_test.cc
        src/terrain_correction_test_kernels.cu
        src/orbit_state_vectors_test.cc
        src/sentinel1_test.cc
        src/geo_utils_test.cc
        src/product_data_test.cc
        src/sar_geocoding_test.cc
        src/tie_point_grid_test.cc
        src/range_doppler_geocoding_test.cc
        src/zero_doppler_time_test.cc
        src/product_test.cc
        src/dummy_product_reader_plug_in.cc
        src/dummy_product_reader.cc
        src/productdata_utc_test.cc
        src/productdata_int_test.cc
        src/productdata_uint_test.cc
        src/productdata_short_test.cc
        src/productdata_ushort_test.cc
        src/productdata_ubyte_test.cc
        src/productdata_ascii_test.cc
        src/productdata_byte_test.cc
        src/productdata_long_test.cc
        src/metadata_element_test.cc
        src/metadata_attribute_test.cc
        src/delaunay_test.cc
        src/triangular_interpolation_test.cc
        src/burst_indices_test.cc
        src/burst_indices_kernels.cu
        src/compute_burst_offset_test.cc
        src/multithreading_test.cc
        src/sentinel_1_calibrate_test.cc
        src/sentinel_1_calibrate_utils_test.cc
        src/aoi_burst_extract_test.cc
        src/cuda_util_test.cc
        src/geotiff_write_test.cc
        src/topsar_merge_test.cc
        src/zip_test.cc
        )

add_executable(unit-test ${TEST_SOURCES})

target_include_directories(unit-test
        PRIVATE
        ${TEST_INCLUDE_DIRS}
        )

target_link_libraries(unit-test
        PRIVATE
        gmock
        gdal-util-static
        terrain-correction-static
        sentinel1-util-static
        exotic-operations-static
        backgeocoding-static
        snap-engine-static
        cuda-util-static
        tiePointGrid-test-static
        range-doppler-geocoding-test-static
        sar-geocoding-test-static
        sentinel1-calibrate-static
        topsar-split-static
        crypto
        Boost::filesystem
        topsar-merge-static
        )

# Only because of thrust::device_vector included
target_compile_options(unit-test
       PRIVATE
       $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter>
       )

set_target_properties(unit-test
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${ALUS_UNIT_TEST_BINARY_OUTPUT_DIR}
        )
add_dependencies(unit-test integration-test-prepare-data)
add_custom_command(TARGET unit-test
        POST_BUILD
        COMMAND echo "Moving and preparing unit test resources..."
        COMMAND cp -r ${CMAKE_CURRENT_LIST_DIR}/goods ${ALUS_UNIT_TEST_BINARY_OUTPUT_DIR}/.
        COMMAND cp ${ALUS_LFS_GOODS}/S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb_data.zip ${ALUS_UNIT_TEST_BINARY_OUTPUT_DIR}/goods/.
        COMMAND cp ${ALUS_LFS_GOODS}/srtm_41_01.zip ${ALUS_UNIT_TEST_BINARY_OUTPUT_DIR}/goods/.
        COMMAND cp ${ALUS_LFS_GOODS}/srtm_42_01.zip ${ALUS_UNIT_TEST_BINARY_OUTPUT_DIR}/goods/.
        COMMAND cd ${ALUS_UNIT_TEST_BINARY_OUTPUT_DIR}/goods && unzip -oq ${ALUS_UNIT_TEST_BINARY_OUTPUT_DIR}/goods/\\*.zip
        COMMAND mv ${ALUS_UNIT_TEST_BINARY_OUTPUT_DIR}/goods/S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb_data/*deb.tif ${ALUS_UNIT_TEST_BINARY_OUTPUT_DIR}/goods/.
        COMMAND mv ${ALUS_UNIT_TEST_BINARY_OUTPUT_DIR}/goods/S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb_data/*deb.dim ${ALUS_UNIT_TEST_BINARY_OUTPUT_DIR}/goods/.
        )

list(APPEND ALUS_GMOCK_MAIN_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/alus_gmock_main.cc
        )

add_library(alus-gmock-main-static
        STATIC
        ${ALUS_GMOCK_MAIN_SOURCES})

target_link_libraries(alus-gmock-main-static
        PUBLIC
        gmock
        common-util-static
        gdal-util-static
        )
