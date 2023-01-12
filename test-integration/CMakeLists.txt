
add_library(alus-gmock-main "" src/alus_gmock_main.cc)
target_link_libraries(alus-gmock-main
        PUBLIC
        gmock
        common-util-static
        gdal-util-static
        )

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/egm96)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/srtm3)

list(APPEND TEST_SOURCES_COHERENCE_CUDA
        src/coherence_cuda_test.cc
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

list(APPEND TEST_SOURCES_TOPSAR_SPLIT
        src/topsar_split_test.cc
        )

list(APPEND TEST_SOURCES_COREGISTRATION
        src/coregistration_test.cc
        )

list(APPEND TEST_INCLUDE
        ${CMAKE_CURRENT_LIST_DIR}/include
        )

set(INTEGRATION_TEST_DATA_FOLDER ${CMAKE_BINARY_DIR}/test-integration/goods)
#----------------Prepare all data for integration tests-----------------------------
add_custom_target(integration-test-prepare-data ALL
        COMMAND echo "Creating integration test data"

        COMMAND ${ALUS_LFS_INVOKE}

        # Apply orbit file data
        COMMAND test -d ${INTEGRATION_TEST_DATA_FOLDER}/apply_orbit_file_op || unzip -qn ${ALUS_LFS_GOODS}/apply_orbit_file_op.zip -d ${INTEGRATION_TEST_DATA_FOLDER}

        # Prepare DEM files
        COMMAND test -f ${INTEGRATION_TEST_DATA_FOLDER}/srtm_41_01.tif || unzip -qn ${ALUS_LFS_GOODS}/srtm_41_01.zip -d ${INTEGRATION_TEST_DATA_FOLDER}
        COMMAND test -f ${INTEGRATION_TEST_DATA_FOLDER}/srtm_42_01.tif || unzip -qn ${ALUS_LFS_GOODS}/srtm_42_01.zip -d ${INTEGRATION_TEST_DATA_FOLDER}
        COMMAND test -f ${INTEGRATION_TEST_DATA_FOLDER}/srtm_43_06.tif || unzip -qn ${ALUS_LFS_GOODS}/srtm_43_06.zip -d ${INTEGRATION_TEST_DATA_FOLDER}
        COMMAND test -f ${INTEGRATION_TEST_DATA_FOLDER}/srtm_44_06.tif || unzip -qn ${ALUS_LFS_GOODS}/srtm_44_06.zip -d ${INTEGRATION_TEST_DATA_FOLDER}

        # Prepare Beirut SAFE files
        COMMAND mkdir -p ${INTEGRATION_TEST_DATA_FOLDER}/beirut_images
        COMMAND test -d ${INTEGRATION_TEST_DATA_FOLDER}/beirut_images/S1A_IW_SLC__1SDV_20200805T034334_20200805T034401_033766_03E9F9_52F6.SAFE || unzip -qn ${ALUS_LFS_GOODS}/S1A_IW_SLC__1SDV_20200805T034334_20200805T034401_033766_03E9F9_52F6.zip -d ${INTEGRATION_TEST_DATA_FOLDER}/beirut_images
        COMMAND test -d ${INTEGRATION_TEST_DATA_FOLDER}/beirut_images/S1B_IW_SLC__1SDV_20200730T034254_20200730T034321_022695_02B131_E8DD.SAFE || unzip -qn ${ALUS_LFS_GOODS}/S1B_IW_SLC__1SDV_20200730T034254_20200730T034321_022695_02B131_E8DD.zip -d ${INTEGRATION_TEST_DATA_FOLDER}/beirut_images
        COMMAND cp -n ${ALUS_LFS_GOODS}/S1B_IW_SLC__1SDV_20200730T034254_20200730T034321_022695_02B131_E8DD.zip ${INTEGRATION_TEST_DATA_FOLDER}/beirut_images/

        # Coregistration data
        COMMAND test -f ${INTEGRATION_TEST_DATA_FOLDER}/coregistration_strips.txt || cp ${CMAKE_CURRENT_LIST_DIR}/goods/coregistration_strips.txt ${INTEGRATION_TEST_DATA_FOLDER}/coregistration_strips.txt
        COMMAND test -f ${INTEGRATION_TEST_DATA_FOLDER}/coregistration_strips_cut.txt || cp ${CMAKE_CURRENT_LIST_DIR}/goods/coregistration_strips_cut.txt ${INTEGRATION_TEST_DATA_FOLDER}/coregistration_strips_cut.txt

        COMMAND ln -rfns ${CMAKE_CURRENT_LIST_DIR}/goods/ww15mgh_b.grd ${INTEGRATION_TEST_DATA_FOLDER}/.
        COMMAND ln -rfns ${CMAKE_CURRENT_LIST_DIR}/goods/master_metadata.dim ${INTEGRATION_TEST_DATA_FOLDER}/.
        COMMAND ln -rfns ${CMAKE_CURRENT_LIST_DIR}/goods/slave_metadata.dim ${INTEGRATION_TEST_DATA_FOLDER}/.
        COMMAND ln -fns ${CMAKE_CURRENT_LIST_DIR}/goods/egm96TestData.txt ${INTEGRATION_TEST_DATA_FOLDER}/.
        COMMAND ln -fns ${CMAKE_CURRENT_LIST_DIR}/goods/tileFormatTestData.txt ${INTEGRATION_TEST_DATA_FOLDER}/.
        COMMAND ln -fns ${CMAKE_CURRENT_LIST_DIR}/goods/altitudeTestData.txt ${INTEGRATION_TEST_DATA_FOLDER}/.

        COMMAND cp -nr ${CMAKE_CURRENT_LIST_DIR}/goods/coherence ${INTEGRATION_TEST_DATA_FOLDER}/.
        COMMAND test -f ${INTEGRATION_TEST_DATA_FOLDER}/coherence/4_bands.tif || cp ${ALUS_LFS_GOODS}/4_bands.tif ${INTEGRATION_TEST_DATA_FOLDER}/coherence/

        # Prepare TC Metadata test data
        COMMAND mkdir -p ${INTEGRATION_TEST_DATA_FOLDER}/terrain_correction
        COMMAND ln -fs ${CMAKE_CURRENT_LIST_DIR}/goods/terrain_correction/longitude.img ${INTEGRATION_TEST_DATA_FOLDER}/terrain_correction/longitude.img
        COMMAND ln -fs ${CMAKE_CURRENT_LIST_DIR}/goods/terrain_correction/longitude.hdr ${INTEGRATION_TEST_DATA_FOLDER}/terrain_correction/longitude.hdr
        COMMAND ln -fs ${CMAKE_CURRENT_LIST_DIR}/goods/terrain_correction/latitude.img ${INTEGRATION_TEST_DATA_FOLDER}/terrain_correction/latitude.img
        COMMAND ln -fs ${CMAKE_CURRENT_LIST_DIR}/goods/terrain_correction/latitude.hdr ${INTEGRATION_TEST_DATA_FOLDER}/terrain_correction/latitude.hdr

        # Prepare TC Beirut integration test data
        COMMAND unzip -oq ${ALUS_LFS_GOODS}/Beirut_IW1_6_VH_orb_stack_cor_deb_coh_data.zip -d ${INTEGRATION_TEST_DATA_FOLDER}/terrain_correction

        # Prepare TC Saaremaa test data
        COMMAND unzip -oq ${ALUS_LFS_GOODS}/S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb_data.zip -d ${INTEGRATION_TEST_DATA_FOLDER}/terrain_correction

        # Prepare backgeocoding test data
        COMMAND rm -rf ${INTEGRATION_TEST_DATA_FOLDER}/backgeocoding
        COMMAND ln -fsn ${CMAKE_CURRENT_LIST_DIR}/goods/backgeocoding ${INTEGRATION_TEST_DATA_FOLDER}/backgeocoding

        # Topsar deburst data
        COMMAND test -d ${INTEGRATION_TEST_DATA_FOLDER}/topsar_deburst_op || unzip -qn ${ALUS_LFS_GOODS}/topsar_deburst_op.zip -d ${INTEGRATION_TEST_DATA_FOLDER}

        # Topsar merge data
        COMMAND mkdir -p ${INTEGRATION_TEST_DATA_FOLDER}/topsar-merge
        COMMAND unzip -oq ${ALUS_LFS_GOODS}/topsar_merge_test_data.zip -d ${INTEGRATION_TEST_DATA_FOLDER}/topsar-merge

        VERBATIM
        )

add_executable(integration-test-topsar-split ${TEST_SOURCES_TOPSAR_SPLIT})
target_link_libraries(integration-test-topsar-split
        PRIVATE
        alus-gmock-main
        cuda-util-static
        topsar-split-static
        exotic-operations-static
        app-util-static
        apply-orbit-file-op-static
        )

add_dependencies(integration-test-topsar-split integration-test-prepare-data)

#----------------------Coregistration integration test--------------------------------

add_executable(integration-test-coregistration ${TEST_SOURCES_COREGISTRATION})
target_include_directories(integration-test-coregistration
        PUBLIC
        ${TEST_INCLUDE}
        )
target_link_libraries(integration-test-coregistration
        PRIVATE
        alus-gmock-main
        cuda-util-static
        topsar-split-static
        exotic-operations-static
        app-util-static
        backgeocoding-static
        apply-orbit-file-op-static
        coregistration-static
        )

add_dependencies(integration-test-coregistration integration-test-prepare-data)

#----------------------------Snap engine integration test----------------------------------

add_executable(integration-test-snap-engine ${TEST_SOURCES_SNAP_ENGINE})
target_include_directories(integration-test-snap-engine
        PUBLIC
        ${TEST_INCLUDE}
        ${CMAKE_CURRENT_LIST_DIR}/srtm3/include
        )
target_link_libraries(integration-test-snap-engine
        PRIVATE
        Boost::filesystem
        alus-gmock-main
        egm96-test-static
        srtm3-test-util-static
        snap-engine-static
        gdal-util-static
        cuda-util-static
        exotic-operations-static
        sentinel1-util-static
        )

add_dependencies(integration-test-snap-engine integration-test-prepare-data)

#----------------------------CUDA coherence integration test----------------------------------

add_executable(integration-test-coherence-cuda ${TEST_SOURCES_COHERENCE_CUDA})
target_include_directories(integration-test-coherence-cuda PUBLIC ${TEST_INCLUDE})
target_link_libraries(integration-test-coherence-cuda
        PRIVATE
        alus-gmock-main
        Boost::filesystem
        Boost::iostreams
        cuda-util-static
        gdal-util-static
        sentinel1-util-static
        coherence-cuda-static
        crypto
        )

add_dependencies(integration-test-coherence-cuda integration-test-prepare-data)

#----------------------------terrain correction integration test----------------------------------
add_executable(integration-test-tc ${TEST_SOURCES_TC})
target_include_directories(integration-test-tc PUBLIC ${TEST_INCLUDE})
target_link_libraries(integration-test-tc
        PRIVATE
        Boost::filesystem
        Boost::iostreams
        alus-gmock-main
        terrain-correction-static
        crypto
        )

add_dependencies(integration-test-tc integration-test-prepare-data)

add_executable(integration-test-backgeocoding ${TEST_SOURCES_BACKGEOCODING})
target_include_directories(integration-test-backgeocoding
        PUBLIC
        ${TEST_INCLUDE}
        )
# Only because of thrust::device_vector include
target_compile_options(integration-test-backgeocoding
        PRIVATE
       $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter, -Wno-unknown-pragmas>
       )
target_link_libraries(integration-test-backgeocoding
        PRIVATE
        alus-gmock-main
        backgeocoding-static
        sentinel1-util-static
        snap-engine-static
        gdal-util-static
        exotic-operations-static
        app-util-static
        )

add_dependencies(integration-test-backgeocoding integration-test-prepare-data)

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
        alus-gmock-main
        cuda-util-static
        gdal-util-static
        sentinel1-util-static
        apply-orbit-file-op-static
        crypto
        )

add_dependencies(integration-test-apply-orbit-file-op integration-test-prepare-data)

#---------------------------topsar-deburst-op integration test-----------------------
list(APPEND TEST_SOURCES_TOPSAR_DEBURST_OP
        src/topsar_deburst_op_test.cc
        )
add_executable(integration-test-topsar-deburst-op ${TEST_SOURCES_TOPSAR_DEBURST_OP})

target_include_directories(integration-test-topsar-deburst-op PUBLIC ${TEST_INCLUDE})
target_compile_options(integration-test-topsar-deburst-op
        PRIVATE
       $<$<COMPILE_LANGUAGE:CXX>:-Wno-unknown-pragmas>
       )

target_link_libraries(integration-test-topsar-deburst-op
        PRIVATE
        Boost::filesystem
        Boost::iostreams
        alus-gmock-main
        cuda-util-static
        gdal-util-static
        sentinel1-util-static
        topsar-deburst-op-static
        crypto
        )

add_dependencies(integration-test-topsar-deburst-op integration-test-prepare-data)

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
        alus-gmock-main
        cuda-util-static
        gdal-util-static
        sentinel1-util-static
        snap-engine-static
        crypto
        staticZipper
        )

add_dependencies(integration-sentinel1_product_reader integration-test-prepare-data)

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
        alus-gmock-main
        sentinel1-calibrate-static
        topsar-split-static
        cuda-util-static
        gdal-util-static
        )

add_dependencies(integration-test-sentinel1-calibrate integration-test-prepare-data)

#----------------TOPSAR Merge integration test--------------------------------------
list(APPEND TEST_SOURCES_TOPSAR_MERGE
        src/topsar_merge_test.cc
        )

add_executable(integration-test-topsar-merge ${TEST_SOURCES_TOPSAR_MERGE})

target_include_directories(integration-test-topsar-merge PUBLIC ${TEST_INCLUDE})

target_link_libraries(integration-test-topsar-merge
        PRIVATE
        crypto
        common-util-static
        topsar-merge-static
        alus-gmock-main
        Boost::filesystem
        sentinel1-util-static
        terrain-correction-static
        )

add_dependencies(integration-test-topsar-merge integration-test-prepare-data)

#----------------TNR integration test--------------------------------------
list(APPEND TEST_SOURCES_TNR
        src/thermal_noise_removal_test.cc
        )

add_executable(integration-test-tnr ${TEST_SOURCES_TNR})

target_include_directories(integration-test-tnr PUBLIC ${TEST_INCLUDE})

target_link_libraries(integration-test-tnr
        PRIVATE
        crypto
        common-util-static
        thermal-noise-static
        alus-gmock-main
        Boost::filesystem
        sentinel1-util-static
        terrain-correction-static
        topsar-split-static
        )

add_dependencies(integration-test-tnr integration-test-prepare-data)

#----------------Build all integration-tests----------------------------------------
add_custom_target(integration-test-all)
add_dependencies(integration-test-all
        integration-test-topsar-split
        integration-test-coregistration
        integration-test-snap-engine
        integration-test-coherence-cuda
        integration-test-tc
        integration-test-backgeocoding
        integration-test-apply-orbit-file-op
        integration-test-topsar-deburst-op
        integration-sentinel1_product_reader
        integration-test-sentinel1-calibrate
        integration-test-prepare-data
        integration-test-topsar-merge
        )
