
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

list(APPEND TEST_SOURCES_BACKGEOCODING_FULL
        src/backgeocoding/backgeocoding_threads_test.cc
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

#----------------Prepare all data for integration tests-----------------------------
add_custom_target(integration-test-prepare-data ALL
        COMMAND echo "Creating test data"

        COMMAND mkdir -p ${CMAKE_BINARY_DIR}/test_integration/goods/beirut_images

        # Apply orbit file data
        COMMAND rm -rf ${CMAKE_BINARY_DIR}/test_integration/goods/apply_orbit_file_op
        COMMAND ln -s ${CMAKE_CURRENT_LIST_DIR}/goods/apply_orbit_file_op ${CMAKE_BINARY_DIR}/test_integration/goods/apply_orbit_file_op

        # Prepare DEM files
        COMMAND test -d ${CMAKE_BINARY_DIR}/test_integration/goods/srtm_41_01.tif || unzip -qn ${CMAKE_CURRENT_LIST_DIR}/goods/srtm_41_01.zip -d ${CMAKE_BINARY_DIR}/test_integration/goods/
        COMMAND test -d ${CMAKE_BINARY_DIR}/test_integration/goods/srtm_42_01.tif || unzip -qn ${CMAKE_CURRENT_LIST_DIR}/goods/srtm_42_01.zip -d ${CMAKE_BINARY_DIR}/test_integration/goods/
        COMMAND test -d ${CMAKE_BINARY_DIR}/test_integration/goods/srtm_43_06.tif || unzip -qn ${CMAKE_CURRENT_LIST_DIR}/goods/srtm_43_06.zip -d ${CMAKE_BINARY_DIR}/test_integration/goods/
        COMMAND test -d ${CMAKE_BINARY_DIR}/test_integration/goods/srtm_44_06.tif || unzip -qn ${CMAKE_CURRENT_LIST_DIR}/goods/srtm_44_06.zip -d ${CMAKE_BINARY_DIR}/test_integration/goods/

        # Prepare Beirut thinned SAFE files
        COMMAND test -d ${CMAKE_BINARY_DIR}/test_integration/goods/beirut_images/S1A_IW_SLC__1SDV_20200805T034334_20200805T034401_033766_03E9F9_52F6_thin.SAFE || unzip -qn ${CMAKE_CURRENT_LIST_DIR}/goods/beirut_images/S1A_IW_SLC__1SDV_20200805T034334_20200805T034401_033766_03E9F9_52F6_thin.SAFE.zip -d ${CMAKE_BINARY_DIR}/test_integration/goods/beirut_images
        COMMAND test -d ${CMAKE_BINARY_DIR}/test_integration/goods/beirut_images/S1B_IW_SLC__1SDV_20200730T034254_20200730T034321_022695_02B131_E8DD_thin.SAFE || unzip -qn ${CMAKE_CURRENT_LIST_DIR}/goods/beirut_images/S1B_IW_SLC__1SDV_20200730T034254_20200730T034321_022695_02B131_E8DD_thin.SAFE.zip -d ${CMAKE_BINARY_DIR}/test_integration/goods/beirut_images

        # Coregistration data
        COMMAND test -f ${CMAKE_BINARY_DIR}/test_integration/goods/coregistration_strips.txt || cp ${CMAKE_CURRENT_LIST_DIR}/goods/coregistration_strips.txt ${CMAKE_BINARY_DIR}/test_integration/goods/coregistration_strips.txt
        COMMAND test -f ${CMAKE_BINARY_DIR}/test_integration/goods/coregistration_strips_cut.txt || cp ${CMAKE_CURRENT_LIST_DIR}/goods/coregistration_strips_cut.txt ${CMAKE_BINARY_DIR}/test_integration/goods/coregistration_strips_cut.txt

        COMMAND ln -rfns ${CMAKE_CURRENT_LIST_DIR}/goods/ww15mgh_b.grd ${CMAKE_BINARY_DIR}/test_integration/goods/.
        COMMAND ln -rfns ${CMAKE_CURRENT_LIST_DIR}/goods/master_metadata.dim ${CMAKE_BINARY_DIR}/test_integration/goods/.
        COMMAND ln -rfns ${CMAKE_CURRENT_LIST_DIR}/goods/slave_metadata.dim ${CMAKE_BINARY_DIR}/test_integration/goods/.
        COMMAND ln -fns ${CMAKE_CURRENT_LIST_DIR}/goods/egm96TestData.txt ${CMAKE_BINARY_DIR}/test_integration/goods/.
        COMMAND ln -fns ${CMAKE_CURRENT_LIST_DIR}/goods/tileFormatTestData.txt ${CMAKE_BINARY_DIR}/test_integration/goods/.
        COMMAND ln -fns ${CMAKE_CURRENT_LIST_DIR}/goods/altitudeTestData.txt ${CMAKE_BINARY_DIR}/test_integration/goods/.

        COMMAND rm -rf ${CMAKE_BINARY_DIR}/test_integration/goods/coherence
        COMMAND ln -s ${CMAKE_CURRENT_LIST_DIR}/goods/coherence ${CMAKE_BINARY_DIR}/test_integration/goods/coherence

        COMMAND mkdir -p ${CMAKE_BINARY_DIR}/test_integration/goods/terrain_correction

        # Prepare TC Metadata test data
        COMMAND ln -fs ${CMAKE_CURRENT_LIST_DIR}/goods/terrain_correction/longitude.img ${CMAKE_BINARY_DIR}/test_integration/goods/terrain_correction/longitude.img
        COMMAND ln -fs ${CMAKE_CURRENT_LIST_DIR}/goods/terrain_correction/longitude.hdr ${CMAKE_BINARY_DIR}/test_integration/goods/terrain_correction/longitude.hdr
        COMMAND ln -fs ${CMAKE_CURRENT_LIST_DIR}/goods/terrain_correction/latitude.img ${CMAKE_BINARY_DIR}/test_integration/goods/terrain_correction/latitude.img
        COMMAND ln -fs ${CMAKE_CURRENT_LIST_DIR}/goods/terrain_correction/latitude.hdr ${CMAKE_BINARY_DIR}/test_integration/goods/terrain_correction/latitude.hdr

        # Prepare TC Beirut integration test data
        COMMAND unzip -oq ${CMAKE_CURRENT_LIST_DIR}/goods/terrain_correction/Beirut_IW1_6_VH_orb_stack_cor_deb_coh_data.zip
        -d ${CMAKE_BINARY_DIR}/test_integration/goods/terrain_correction

        # Prepare TC Saaremaa test data
        COMMAND unzip -oq ${CMAKE_CURRENT_LIST_DIR}/goods/terrain_correction/S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb_data.zip
        -d ${CMAKE_BINARY_DIR}/test_integration/goods/terrain_correction

        # Prepare backgeocoding test data
        COMMAND rm -rf ${CMAKE_BINARY_DIR}/test_integration/goods/backgeocoding COMMAND ln -fsn ${CMAKE_CURRENT_LIST_DIR}/goods/backgeocoding ${CMAKE_BINARY_DIR}/test_integration/goods/backgeocoding

        # Topsar deburst data
        COMMAND rm -rf ${CMAKE_BINARY_DIR}/test_integration/goods/topsar_deburst_op
        COMMAND ln -s ${CMAKE_CURRENT_LIST_DIR}/goods/topsar_deburst_op
        ${CMAKE_BINARY_DIR}/test_integration/goods/topsar_deburst_op

        COMMAND ln -fsn ${CMAKE_CURRENT_LIST_DIR}/goods/beirut_images/S1B_IW_SLC__1SDV_20200730T034254_20200730T034321_022695_02B131_E8DD_thin.SAFE.zip
        ${CMAKE_BINARY_DIR}/test_integration/goods/beirut_images/S1B_IW_SLC__1SDV_20200730T034254_20200730T034321_022695_02B131_E8DD_thin.SAFE.zip

        VERBATIM
        )

add_executable(backgeocoding-full ${TEST_SOURCES_BACKGEOCODING_FULL})
target_include_directories(backgeocoding-full
        PUBLIC
        ${TEST_INCLUDE}
        )

target_link_libraries(backgeocoding-full
        PRIVATE
        alus-gmock-main
        cuda_util-static
        backgeocoding-static
        exotic_operations-static
        app-util-static
        )

add_dependencies(backgeocoding-full integration-test-prepare-data)

add_executable(integration-test-topsar-split ${TEST_SOURCES_TOPSAR_SPLIT})
target_link_libraries(integration-test-topsar-split
        PRIVATE
        alus-gmock-main
        cuda_util-static
        topsar-split-static
        exotic_operations-static
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
        cuda_util-static
        topsar-split-static
        exotic_operations-static
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
        cuda_util-static
        exotic_operations-static
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
        cuda_util-static
        gdal-util-static
        sentinel1-util-static
        coherence-cuda-static
        crypto
        )
target_compile_options(integration-test-coherence-cuda
        PRIVATE
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
target_link_libraries(integration-test-backgeocoding
        PRIVATE
        alus-gmock-main
        backgeocoding-static
        sentinel1-util-static
        snap-engine-static
        gdal-util-static
        exotic_operations-static
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
        cuda_util-static
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

target_link_libraries(integration-test-topsar-deburst-op
        PRIVATE
        Boost::filesystem
        Boost::iostreams
        alus-gmock-main
        cuda_util-static
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
        cuda_util-static
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
        cuda_util-static
        gdal-util-static
        )

add_dependencies(integration-test-sentinel1-calibrate integration-test-prepare-data)

#----------------Build all integration-tests----------------------------------------
add_custom_target(integration-test-all)
add_dependencies(integration-test-all
        backgeocoding-full
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
        )