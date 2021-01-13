
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/egm96)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/srtm3)

list(APPEND TEST_SOURCES_COHERENCE
        src/coherence_test.cc
        )

list(APPEND TEST_SOURCES_SNAP_ENGINE
        src/egm96_test.cc
        src/srtm3_altitude_test.cc
        src/srtm3_formating_test.cc)

list(APPEND TEST_SOURCES_TC
        src/terrain_correction_test.cc
        src/terrain_correction_test.cu
        src/terrain_correction_metadata_test.cc
        )

list(APPEND TEST_SOURCES_BACKGEOCODING
        src/backgeocoding/backgeocoding_test.cc
        src/extended_amount_test.cc
        )

list(APPEND TEST_SOURCES_BACKGEOCODING_FULL
        src/backgeocoding/backgeocoding_threads_test.cc
        )

list(APPEND TEST_INCLUDE
        ${CMAKE_CURRENT_LIST_DIR}/include
        )

add_executable(backgeocoding-full ${TEST_SOURCES_BACKGEOCODING_FULL})
target_include_directories(backgeocoding-full
        PUBLIC
        ${TEST_INCLUDE}
        )

target_link_libraries(backgeocoding-full
        PRIVATE
        gmock_main
        cuda_util-static
        backgeocoding-static
        exotic_operations-static
        app-util-static
        )


add_executable(integration-test-snap-engine ${TEST_SOURCES_SNAP_ENGINE})
target_include_directories(integration-test-snap-engine
        PUBLIC
        ${TEST_INCLUDE}
        ${CMAKE_CURRENT_LIST_DIR}/srtm3/include
        )
target_link_libraries(integration-test-snap-engine
        PRIVATE
        Boost::filesystem
        gmock_main
        egm96-test-static
        srtm3-test-util-static
        snap-engine-static
        gdal-util-static
        cuda_util-static
        exotic_operations-static
        sentinel1-util-static
        )
add_custom_command(TARGET integration-test-snap-engine
        POST_BUILD
        COMMAND mkdir -p ${CMAKE_BINARY_DIR}/test_integration/goods
        # Since grid file is already symlink from unit test goods, option '-r' is needed
        COMMAND ln -rfns ${CMAKE_CURRENT_LIST_DIR}/goods/ww15mgh_b.grd ${CMAKE_BINARY_DIR}/test_integration/goods/.
        COMMAND ln -rfns ${CMAKE_CURRENT_LIST_DIR}/goods/master_metadata.dim ${CMAKE_BINARY_DIR}/test_integration/goods/.
        COMMAND ln -rfns ${CMAKE_CURRENT_LIST_DIR}/goods/slave_metadata.dim ${CMAKE_BINARY_DIR}/test_integration/goods/.
        COMMAND ln -fns ${CMAKE_CURRENT_LIST_DIR}/goods/egm96TestData.txt ${CMAKE_BINARY_DIR}/test_integration/goods/.
        COMMAND ln -fns ${CMAKE_CURRENT_LIST_DIR}/goods/tileFormatTestData.txt ${CMAKE_BINARY_DIR}/test_integration/goods/.
        COMMAND ln -fns ${CMAKE_CURRENT_LIST_DIR}/goods/altitudeTestData.txt ${CMAKE_BINARY_DIR}/test_integration/goods/.
        )

add_executable(integration-test-coherence ${TEST_SOURCES_COHERENCE})
target_include_directories(integration-test-coherence PUBLIC ${TEST_INCLUDE})
target_link_libraries(integration-test-coherence
        PRIVATE
        Boost::filesystem
        Boost::iostreams
        gtest_main
        cuda_util-static
        gdal-util-static
        sentinel1-util-static
        coherence-static
        crypto
        )
target_compile_options(integration-test-coherence
        PRIVATE
        -Wno-pedantic
        -Wno-unused-parameter
        -Wno-ignored-qualifiers
        -Wno-unused-variable
        -Wno-unused-but-set-parameter
        )
add_custom_command(TARGET integration-test-coherence
        POST_BUILD
        COMMAND mkdir -p ${CMAKE_BINARY_DIR}/test_integration/goods
        COMMAND rm -rf ${CMAKE_BINARY_DIR}/test_integration/goods/coherence
        COMMAND ln -s ${CMAKE_CURRENT_LIST_DIR}/goods/coherence
        ${CMAKE_BINARY_DIR}/test_integration/goods/coherence
        )

add_executable(integration-test-tc ${TEST_SOURCES_TC})
target_include_directories(integration-test-tc PUBLIC ${TEST_INCLUDE})
target_link_libraries(integration-test-tc
        PRIVATE
        Boost::filesystem
        Boost::iostreams
        gmock_main
        terrain-correction-static
        crypto
        )
add_custom_command(TARGET integration-test-tc
        POST_BUILD
        COMMAND mkdir -p ${CMAKE_BINARY_DIR}/test_integration/goods/terrain_correction

        COMMAND cp -aln ${CMAKE_CURRENT_LIST_DIR}/goods/S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb.data.zip
        ${CMAKE_BINARY_DIR}/test_integration/goods/.

        COMMAND cp -aln ${CMAKE_CURRENT_LIST_DIR}/goods/terrain_correction/Beirut_IW1_6_VH_orb_stack_cor_deb_coh.data.zip
        ${CMAKE_BINARY_DIR}/test_integration/goods/terrain_correction/.

        COMMAND cd ${CMAKE_BINARY_DIR}/test_integration/goods &&
        unzip -oq S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb.data.zip

        COMMAND cd ${CMAKE_BINARY_DIR}/test_integration/goods/terrain_correction &&
        unzip -oq Beirut_IW1_6_VH_orb_stack_cor_deb_coh.data.zip

        COMMAND cp -aln ${CMAKE_CURRENT_LIST_DIR}/goods/S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb.tif
        ${CMAKE_BINARY_DIR}/test_integration/goods/.

        COMMAND cp -aln ${CMAKE_CURRENT_LIST_DIR}/goods/S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb_TC.tif
        ${CMAKE_BINARY_DIR}/test_integration/goods/.

        COMMAND cp -aln ${CMAKE_CURRENT_LIST_DIR}/goods/terrain_correction/Beirut_IW1_6_VH_orb_stack_cor_deb_coh.tif
        ${CMAKE_BINARY_DIR}/test_integration/goods/terrain_correction/.

        COMMAND cp -aln ${CMAKE_CURRENT_LIST_DIR}/goods/terrain_correction/Beirut_IW1_6_VH_orb_stack_cor_deb_coh_TC.tif
        ${CMAKE_BINARY_DIR}/test_integration/goods/terrain_correction/.

#        Copy srtm3 files necessary for Beirut test image
        COMMAND cp -aln ${CMAKE_CURRENT_LIST_DIR}/goods/srtm_43_06.zip ${CMAKE_BINARY_DIR}/test_integration/goods/.
        COMMAND cd ${CMAKE_BINARY_DIR}/test_integration/goods && unzip -qn srtm_43_06.zip
        COMMAND cp -aln ${CMAKE_CURRENT_LIST_DIR}/goods/srtm_44_06.zip ${CMAKE_BINARY_DIR}/test_integration/goods/.
        COMMAND cd ${CMAKE_BINARY_DIR}/test_integration/goods && unzip -qn srtm_44_06.zip
        )

add_executable(integration-test-backgeocoding ${TEST_SOURCES_BACKGEOCODING})
target_include_directories(integration-test-backgeocoding
        PUBLIC
        ${TEST_INCLUDE}
        )
target_link_libraries(integration-test-backgeocoding
        PRIVATE
        gmock_main
        backgeocoding-static
        sentinel1-util-static
        snap-engine-static
        gdal-util-static
        exotic_operations-static
        app-util-static
        )
add_custom_command(TARGET integration-test-backgeocoding
        POST_BUILD
        COMMAND mkdir -p ${CMAKE_BINARY_DIR}/test_integration/goods
        COMMAND rm -rf ${CMAKE_BINARY_DIR}/test_integration/goods/backgeocoding
        COMMAND ln -fsn ${CMAKE_CURRENT_LIST_DIR}/goods/backgeocoding ${CMAKE_BINARY_DIR}/test_integration/goods/backgeocoding
        COMMAND cp -n ${CMAKE_CURRENT_LIST_DIR}/goods/srtm_41_01.zip ${CMAKE_BINARY_DIR}/test_integration/goods/.
        COMMAND cd ${CMAKE_BINARY_DIR}/test_integration/goods && unzip -qn srtm_41_01.zip
        COMMAND cp -n ${CMAKE_CURRENT_LIST_DIR}/goods/srtm_42_01.zip ${CMAKE_BINARY_DIR}/test_integration/goods/.
        COMMAND cd ${CMAKE_BINARY_DIR}/test_integration/goods && unzip -qn srtm_42_01.zip
        )

#---------------------------apply orbit file integration test-----------------------
list(APPEND TEST_SOURCES_APPLY_ORBIT_FILE_OP
        src/apply_orbit_file_test.cc
        )
add_executable(integration-test-apply-orbit-file-op ${TEST_SOURCES_APPLY_ORBIT_FILE_OP})

target_include_directories(integration-test-apply-orbit-file-op PUBLIC ${TEST_INCLUDE})

target_link_libraries(integration-test-apply-orbit-file-op
        PRIVATE
        Boost::filesystem
        Boost::iostreams
        gmock_main
        cuda_util-static
        gdal-util-static
        sentinel1-util-static
        apply-orbit-file-op-static
        crypto
        )

add_custom_command(TARGET integration-test-apply-orbit-file-op
        POST_BUILD
        COMMAND mkdir -p ${CMAKE_BINARY_DIR}/test_integration/goods
        COMMAND rm -rf ${CMAKE_BINARY_DIR}/test_integration/goods/apply_orbit_file_op
        COMMAND ln -s ${CMAKE_CURRENT_LIST_DIR}/goods/apply_orbit_file_op
        ${CMAKE_BINARY_DIR}/test_integration/goods/apply_orbit_file_op
        )
#---------------------------topsar-deburst-op integration test-----------------------
list(APPEND TEST_SOURCES_TOPSAR_DEBURST_OP
        src/topsar_deburst_op_test.cc
        )
add_executable(integration-test-topsar-deburst-op ${TEST_SOURCES_TOPSAR_DEBURST_OP})

target_include_directories(integration-test-topsar-deburst-op PUBLIC ${TEST_INCLUDE})

target_link_libraries(integration-test-topsar-deburst-op
        PRIVATE
        Boost::filesystem
        Boost::iostreams
        gmock_main
        cuda_util-static
        gdal-util-static
        sentinel1-util-static
        topsar-deburst-op-static
        crypto
        )

add_custom_command(TARGET integration-test-topsar-deburst-op
        POST_BUILD
        COMMAND mkdir -p ${CMAKE_BINARY_DIR}/test_integration/goods
        COMMAND rm -rf ${CMAKE_BINARY_DIR}/test_integration/goods/topsar_deburst_op
        COMMAND ln -s ${CMAKE_CURRENT_LIST_DIR}/goods/topsar_deburst_op
        ${CMAKE_BINARY_DIR}/test_integration/goods/topsar_deburst_op
        )
#--------------------------sentinel1 product reader integration test-----------------------
list(APPEND TEST_SOURCES_SENTINEL1_PRODUCT_READER
        src/test_sentinel1_product_reader.cc
        )
add_executable(integration-sentinel1_product_reader ${TEST_SOURCES_SENTINEL1_PRODUCT_READER})

target_include_directories(integration-sentinel1_product_reader PUBLIC ${TEST_INCLUDE})

#todo: probably not everything used what is declared, check these over
target_link_libraries(integration-sentinel1_product_reader
        PRIVATE
        Boost::filesystem
        Boost::iostreams
        gmock_main
        cuda_util-static
        gdal-util-static
        sentinel1-util-static
        snap-engine-static
        crypto
        staticZipper
        )

add_custom_command(TARGET integration-sentinel1_product_reader
        POST_BUILD
        COMMAND mkdir -p ${CMAKE_BINARY_DIR}/test_integration/goods
        COMMAND rm -rf ${CMAKE_BINARY_DIR}/test_integration/goods/sentinel1_product_reader
        COMMAND cp -al ${CMAKE_CURRENT_LIST_DIR}/goods/sentinel1_product_reader ${CMAKE_BINARY_DIR}/test_integration/goods/sentinel1_product_reader
        COMMAND cd ${CMAKE_BINARY_DIR}/test_integration/goods/sentinel1_product_reader && unzip -qn ${CMAKE_BINARY_DIR}/test_integration/goods/sentinel1_product_reader/S1A_IW_SLC__1SDV_20180815T154813_20180815T154840_023259_028747_4563.zip
        )

#--------------------------sentinel1 calibrate integration test-----------------------
list(APPEND TEST_SOURCES_SENTINEL1_CALIBRATE
        src/sentinel1_calibrate_test.cc
        )
add_executable(integration-test-sentinel1-calibrate ${TEST_SOURCES_SENTINEL1_CALIBRATE})

target_include_directories(integration-test-sentinel1-calibrate PUBLIC ${TEST_INCLUDE})

target_link_libraries(integration-test-sentinel1-calibrate
        PRIVATE
        crypto
        Boost::filesystem
        Boost::iostreams
        gmock_main
        sentinel1-calibrate-static
        cuda_util-static
        gdal-util-static
        )

add_custom_command(TARGET integration-test-sentinel1-calibrate
        POST_BUILD
        COMMAND mkdir -p ${CMAKE_BINARY_DIR}/test_integration/goods/sentinel1_calibrate

        COMMAND cp -arn ${CMAKE_CURRENT_LIST_DIR}/goods/sentinel1_product_reader/S1A_IW_SLC__1SDV_20180815T154813_20180815T154840_023259_028747_4563.zip
        ${CMAKE_BINARY_DIR}/test_integration/goods/sentinel1_calibrate/.

        COMMAND cd ${CMAKE_BINARY_DIR}/test_integration/goods/sentinel1_calibrate &&
        unzip -oq S1A_IW_SLC__1SDV_20180815T154813_20180815T154840_023259_028747_4563.zip
        )