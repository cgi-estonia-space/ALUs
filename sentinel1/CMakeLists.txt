list(APPEND SENTINEL1_UTIL_SOURCES-GPU
        s1tbx-commons/orbit_state_vectors.cu
        s1tbx-commons/sar_geocoding.cu
        )
list(APPEND SENTINEL1_UTIL_SOURCES
        s1tbx-commons/sentinel1_utils.cc
        s1tbx-commons/subswath_info.cc
        jlinda/jlinda-core/point.cc
        s1tbx-commons/sar_utils.cc
        jlinda/jlinda-core/utils/poly_utils.cc
        s1tbx-commons/orbit.cc
        s1tbx-commons/orbit_state_vectors.cc
        io/orbits/sentinel1/sentinel_p_o_d_orbit_file.cc
        jlinda/jlinda-core/ellipsoid.cc
        jlinda/jlinda-core/geopoint.cc
        s1tbx-io/sentinel1/sentinel1_product_reader.cc
        s1tbx-io/sentinel1/sentinel1_level1_directory.cc
        s1tbx-io/sentinel1/sentinel1_product_reader_plug_in.cc
        s1tbx-io/geotiffxml/geo_tiff_utils.cc
        s1tbx-commons/io/abstract_product_directory.cc
        s1tbx-commons/io/x_m_l_product_directory.cc
        s1tbx-commons/io/sar_reader.cc
        s1tbx-commons/io/image_i_o_file.cc
        s1tbx-commons/io/band_info.cc
        s1tbx-commons/test/metadata_validator.cc
        s1tbx-commons/test/reader_test.cc
        jlinda/jlinda-core/utils/date_utils.cc
        )

list(APPEND SENTINEL1_UTIL_INCLUDE_DIRS
        #sentinel1
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/io/orbits/sentinel1
        ${CMAKE_CURRENT_LIST_DIR}/jlinda/jlinda-core
        ${CMAKE_CURRENT_LIST_DIR}/s1tbx-io/sentinel1
        ${CMAKE_CURRENT_LIST_DIR}/s1tbx-io/geotiffxml
        ${CMAKE_CURRENT_LIST_DIR}/s1tbx-commons/io
        #snap-engine
        ${CMAKE_CURRENT_LIST_DIR}/../snap-engine
        ${CMAKE_CURRENT_LIST_DIR}/../snap-engine/custom
        ${CMAKE_CURRENT_LIST_DIR}/../snap-engine/ceres-core
        ${CMAKE_CURRENT_LIST_DIR}/../snap-engine/snap-core/core/datamodel
        ${CMAKE_CURRENT_LIST_DIR}/../snap-engine/snap-core/core/datamodel/quicklooks
        ${CMAKE_CURRENT_LIST_DIR}/../snap-engine/snap-core/core/dataio
        ${CMAKE_CURRENT_LIST_DIR}/../snap-engine/snap-core/core/util
        ${CMAKE_CURRENT_LIST_DIR}/../snap-engine/snap-core/core/util/math
        ${CMAKE_CURRENT_LIST_DIR}/../snap-engine/snap-core/core/util/io
        ${CMAKE_CURRENT_LIST_DIR}/../snap-engine/snap-core/core/subset
        ${CMAKE_CURRENT_LIST_DIR}/../snap-engine/snap-engine-utilities
        ${CMAKE_CURRENT_LIST_DIR}/../snap-engine/snap-engine-utilities/engine-utilities/util
        ${CMAKE_CURRENT_LIST_DIR}/../snap-engine/snap-engine-utilities/engine-utilities/datamodel
        ${CMAKE_CURRENT_LIST_DIR}/../snap-engine/snap-engine-utilities/engine-utilities/datamodel/metadata
        ${CMAKE_CURRENT_LIST_DIR}/../snap-engine/snap-engine-utilities/engine-utilities/eo
        ${CMAKE_CURRENT_LIST_DIR}/../snap-engine/snap-engine-utilities/engine-utilities/gpf
        #other
        ${CMAKE_CURRENT_LIST_DIR}/../util/cuda/include
        ${CMAKE_CURRENT_LIST_DIR}/../util/common/include
        )
#/*********************************************/
add_library(sentinel1-util-static
        STATIC
        ${SENTINEL1_UTIL_SOURCES}
        ${SENTINEL1_UTIL_SOURCES-GPU}
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
        common-util-static
        )

