
include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.8.1
)

FetchContent_GetProperties(googletest)

if(NOT googletest_POPULATED)
  FetchContent_Populate(googletest)

  # Prevent GoogleTest from overriding our compiler/linker options
  # when building with Visual Studio
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  # Prevent GoogleTest from using PThreads
  set(gtest_disable_pthreads ON CACHE BOOL "" FORCE)

  # adds the targers: gtest, gtest_main, gmock, gmock_main
  add_subdirectory(
    ${googletest_SOURCE_DIR}
    ${googletest_BINARY_DIR}
    )

  # Silence std::tr1 warning on MSVC
  if(MSVC)
    foreach(_tgt gtest gtest_main gmock gmock_main)
      target_compile_definitions(${_tgt}
        PRIVATE
          "_SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING"
        )
    endforeach()
  endif()
endif()

list(APPEND TEST_SOURCES
        src/dataset_test.cpp
        src/terrain_correction_test.cc
        src/terrain_correction_test_kernels.cu
        src/dem_test.cpp
        src/orbit_state_vectors_test.cc
        src/tests_common.cpp
        src/sentinel1_test.cc
        src/backgeocoding_test.cc
        src/egm96_test.cc
        src/geo_utils_test.cpp
        src/product_data_test.cpp
        src/sar_geocoding_test.cc
        src/srtm3_altitude_test.cc
        src/srtm3_formating_test.cc
        src/tie_point_grid_test.cc
        src/range_doppler_geocoding_test.cc
        src/zero_doppler_time_test.cc
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
        )

add_executable(unit_test ${TEST_SOURCES})
target_link_libraries(unit_test
        PRIVATE
        gmock_main
        gdal-util-static
        terrain-correction-static
        sentinel1-util-static
        exotic_operations-static
        backgeocoding-static
        snap-engine-static
        egm96-test-static
        srtm3-test-util-static
        cuda_util-static
        tiePointGrid-test-static
        range-doppler-geocoding-test-static
        sar-geocoding-test-static
        coherence-static
        )

add_executable(map_cut util/map_cut.cpp)
target_link_libraries(map_cut
        PRIVATE
        gdal-util-static
        )

add_custom_command(TARGET unit_test
        POST_BUILD
        COMMAND cp -r ${CMAKE_CURRENT_LIST_DIR}/goods ${CMAKE_BINARY_DIR}/test/.
        COMMAND cd ${CMAKE_BINARY_DIR}/test/goods && unzip -oq ${CMAKE_BINARY_DIR}/test/goods/\\*.zip
        )
