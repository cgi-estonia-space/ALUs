list(APPEND SENTINEL1_UTIL_SOURCES-GPU
        ${CMAKE_CURRENT_LIST_DIR}/src/orbit_state_vectors.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/sar_geocoding.cu
        )
list(APPEND SENTINEL1_UTIL_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/sentinel1_utils.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/subswath_info.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/point.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/s1tbx-commons/sar_utils.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/poly_utils.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/orbit.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/orbit_state_vectors.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/io/orbits/sentinel1/sentinel_p_o_d_orbit_file.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/jlinda-core/ellipsoid.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/jlinda-core/geopoint.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/s1tbx-io/sentinel1/sentinel1_product_reader.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/s1tbx-io/sentinel1/sentinel1_level1_directory.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/s1tbx-io/sentinel1/sentinel1_product_reader_plug_in.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/s1tbx-io/geotiffxml/geo_tiff_utils.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/s1tbx-commons/io/abstract_product_directory.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/s1tbx-commons/io/x_m_l_product_directory.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/s1tbx-commons/io/sar_reader.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/s1tbx-commons/io/image_i_o_file.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/s1tbx-commons/io/band_info.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/s1tbx-commons/test/metadata_validator.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/s1tbx-commons/test/reader_test.cc
        )

list(APPEND SENTINEL1_UTIL_INCLUDE_DIRS
        #sentinel1
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/include/io/orbits/sentinel1
        ${CMAKE_CURRENT_LIST_DIR}/include/jlinda-core
        ${CMAKE_CURRENT_LIST_DIR}/include/s1tbx-io/sentinel1
        ${CMAKE_CURRENT_LIST_DIR}/include/s1tbx-io/geotiffxml
        ${CMAKE_CURRENT_LIST_DIR}/include/s1tbx-commons/io
        #snap-engine
        ${CMAKE_CURRENT_LIST_DIR}/../snap-engine/include
        ${CMAKE_CURRENT_LIST_DIR}/../snap-engine/include/custom
        ${CMAKE_CURRENT_LIST_DIR}/../snap-engine/include/ceres-core
        ${CMAKE_CURRENT_LIST_DIR}/../snap-engine/include/snap-core/datamodel
        ${CMAKE_CURRENT_LIST_DIR}/../snap-engine/include/snap-core/datamodel/quicklooks
        ${CMAKE_CURRENT_LIST_DIR}/../snap-engine/include/snap-core/dataio
        ${CMAKE_CURRENT_LIST_DIR}/../snap-engine/include/snap-core/util
        ${CMAKE_CURRENT_LIST_DIR}/../snap-engine/include/snap-core/util/math
        ${CMAKE_CURRENT_LIST_DIR}/../snap-engine/include/snap-core/util/io
        ${CMAKE_CURRENT_LIST_DIR}/../snap-engine/include/snap-core/subset
        ${CMAKE_CURRENT_LIST_DIR}/../snap-engine/include/snap-engine-utilities
        ${CMAKE_CURRENT_LIST_DIR}/../snap-engine/include/snap-engine-utilities/util
        ${CMAKE_CURRENT_LIST_DIR}/../snap-engine/include/snap-engine-utilities/datamodel
        ${CMAKE_CURRENT_LIST_DIR}/../snap-engine/include/snap-engine-utilities/datamodel/metadata
        ${CMAKE_CURRENT_LIST_DIR}/../snap-engine/include/snap-engine-utilities/eo
        ${CMAKE_CURRENT_LIST_DIR}/../snap-engine/include/snap-engine-utilities/gpf
        #other
        ${CMAKE_CURRENT_LIST_DIR}/../cuda/include
        ${CMAKE_CURRENT_LIST_DIR}/../common/include
        )
#/*********************************************/
add_library(sentinel1-util-obj-cuda
        OBJECT
        ${SENTINEL1_UTIL_SOURCES-GPU}
        )

target_include_directories(sentinel1-util-obj-cuda
        PRIVATE
        ${SENTINEL1_UTIL_INCLUDE_DIRS}
        )

target_link_libraries(sentinel1-util-obj-cuda
        PUBLIC
        gdal
        snap-engine-static
        exotic_operations-static
        Eigen3::Eigen
        staticZipper
        )

add_library(sentinel1-util-obj
        OBJECT
        ${SENTINEL1_UTIL_SOURCES}
        )

target_include_directories(sentinel1-util-obj
        PRIVATE
        ${SENTINEL1_UTIL_INCLUDE_DIRS}
        )

target_link_libraries(sentinel1-util-obj
        PUBLIC
        gdal
        snap-engine-static
        exotic_operations-static
        Eigen3::Eigen
        staticZipper
        )

add_library(sentinel1-util-static
        STATIC
        $<TARGET_OBJECTS:sentinel1-util-obj>
        $<TARGET_OBJECTS:sentinel1-util-obj-cuda>
        )

target_include_directories(sentinel1-util-static
        PUBLIC
        ${SENTINEL1_UTIL_INCLUDE_DIRS}
        )

set_target_properties(sentinel1-util-static
        PROPERTIES
        OUTPUT_NAME sentinel1-util
        )

target_link_libraries(sentinel1-util-static
        PUBLIC
        gdal
        snap-engine-static
        exotic_operations-static
        Eigen3::Eigen
        staticZipper
        )

