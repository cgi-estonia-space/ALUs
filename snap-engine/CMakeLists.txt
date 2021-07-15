
list(APPEND SNAP-ENGINE_SOURCES_CU
        GeoUtils.cu
        earth_gravitational_model96.cc
        srtm3_elevation_model.cc
        srtm3_formatter.cu
        tie_point_grid.cu
        tie_point_geocoding.cc
        crs_geocoding.cc
        triangular_interpolation.cu
        )
list(APPEND SNAP-ENGINE_SOURCES
        snap-core/core/datamodel/data_node.cc
        snap-core/core/dataio/product_subset_def.cc
        snap-core/core/dataio/abstract_product_reader.cc
        snap-core/core/dataio/product_i_o.cc
        snap-core/core/dataio/product_subset_builder.cc
        snap-core/core/dataio/abstract_product_builder.cc
        guardian.cc
        snap-core/core/datamodel/geo_pos.cc
        snap-engine-utilities/engine-utilities/datamodel/metadata/abstract_metadata.cc
        snap-engine-utilities/engine-utilities/datamodel/metadata/abstract_metadata_i_o.cc
        snap-core/core/datamodel/metadata_attribute.cc
        snap-core/core/datamodel/metadata_element.cc
        snap-core/core/datamodel/parse_exception.cc
        snap-core/core/datamodel/product.cc
        snap-core/core/datamodel/product_data.cc
        snap-core/core/datamodel/product_data_ascii.cc
        snap-core/core/datamodel/product_data_byte.cc
        snap-core/core/datamodel/product_data_double.cc
        snap-core/core/datamodel/product_data_float.cc
        snap-core/core/datamodel/product_data_int.cc
        snap-core/core/datamodel/product_data_long.cc
        snap-core/core/datamodel/product_data_short.cc
        snap-core/core/datamodel/product_data_ubyte.cc
        snap-core/core/datamodel/product_data_uint.cc
        snap-core/core/datamodel/product_data_ushort.cc
        snap-core/core/datamodel/product_data_utc.cc
        snap-core/core/datamodel/product_data_utc.cc
        snap-core/core/datamodel/product_node.cc
        snap-core/core/datamodel/product_node_list.cc
        snap-core/core/datamodel/product_node_group.cc
        snap-core/core/datamodel/abstract_band.cc
        snap-core/core/datamodel/abstract_geo_coding.cc
        snap-core/core/datamodel/raster_data_node.cc
        snap-core/core/datamodel/sample_coding.cc
        snap-core/core/datamodel/tie_point_grid.cc
        snap-core/core/datamodel/tie_point_geo_coding.cc
        snap-core/core/datamodel/virtual_band.cc
        snap-core/core/datamodel/band.cc
        snap-core/core/datamodel/flag_coding.cc
        snap-core/core/datamodel/index_coding.cc
        snap-core/core/datamodel/quicklooks/quicklook.cc
        snap-core/core/datamodel/mask.cc
        snap-core/core/datamodel/pugixml_meta_data_reader.cc
        snap-core/core/datamodel/pugixml_meta_data_writer.cc
        snap-core/core/subset/abstract_subset_region.cc
        snap-core/core/subset/pixel_subset_region.cc
        snap-core/core/util/product_utils.cc
        snap-core/core/util/string_utils.cc
        snap-core/core/util/math/f_x_y_sum.cc
        snap-core/core/util/math/f_x_y_sum_linear.cc
        snap-core/core/util/math/f_x_y_sum_bi_linear.cc
        snap-core/core/util/math/f_x_y_sum_cubic.cc
        snap-core/core/util/math/f_x_y_sum_bi_cubic.cc
        snap-core/core/util/math/f_x_y_sum_quadric.cc
        snap-core/core/util/math/f_x_y_sum_bi_quadric.cc
        snap-core/core/util/math/i_f_x.cc
        snap-core/core/util/math/i_f_x_y.cc
        snap-core/core/util/math/approximator.cc
        snap-core/core/util/geo_utils.cc
        snap-core/core/util/io/file_utils.cc
        snap-core/core/util/math/math_utils.cc
        snap-engine-utilities/engine-utilities/util/maths.cc
        snap-engine-utilities/engine-utilities/util/test_utils.cc
        snap-engine-utilities/engine-utilities/util/zip_utils.cc
        snap-engine-utilities/engine-utilities/datamodel/metadata/orbit_vector.cc
        snap-engine-utilities/engine-utilities/gpf/input_product_validator.cc
        snap-engine-utilities/engine-utilities/gpf/operator_utils.cc
        snap-engine-utilities/engine-utilities/gpf/stack_utils.cc
        snap-engine-utilities/engine-utilities/gpf/reader_utils.cc
        snap-engine-utilities/engine-utilities/datamodel/metadata/unit.cc
        ${CMAKE_CURRENT_LIST_DIR}/ceres-core/core/i_virtual_dir.cc
        ${CMAKE_CURRENT_LIST_DIR}/ceres-core/core/zip.cc
        ${CMAKE_CURRENT_LIST_DIR}/ceres-core/core/dir.cc
        custom/gdal_image_reader.cc
        custom/gdal_image_writer.cc
        custom/io_tile.cc
        )

list(APPEND SNAP-ENGINE_INCLUDES
        #snap-engine
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/custom
        ${CMAKE_CURRENT_LIST_DIR}/ceres-core
        ${CMAKE_CURRENT_LIST_DIR}/snap-core/core/datamodel
        ${CMAKE_CURRENT_LIST_DIR}/snap-core/core/datamodel/quicklooks
        ${CMAKE_CURRENT_LIST_DIR}/snap-core/core/dataio
        ${CMAKE_CURRENT_LIST_DIR}/snap-core/core/util
        ${CMAKE_CURRENT_LIST_DIR}/snap-core/core/util/math
        ${CMAKE_CURRENT_LIST_DIR}/snap-core/core/util/io
        ${CMAKE_CURRENT_LIST_DIR}/snap-core/core/subset
        ${CMAKE_CURRENT_LIST_DIR}/snap-engine-utilities
        ${CMAKE_CURRENT_LIST_DIR}/snap-engine-utilities/engine-utilities/util
        ${CMAKE_CURRENT_LIST_DIR}/snap-engine-utilities/engine-utilities/datamodel
        ${CMAKE_CURRENT_LIST_DIR}/snap-engine-utilities/engine-utilities/datamodel/metadata
        ${CMAKE_CURRENT_LIST_DIR}/snap-engine-utilities/engine-utilities/eo
        ${CMAKE_CURRENT_LIST_DIR}/snap-engine-utilities/engine-utilities/gpf
        ${CMAKE_CURRENT_LIST_DIR}/sentinel1
        #        ${CMAKE_CURRENT_LIST_DIR}/../sentinel1/
        #        ${CMAKE_CURRENT_LIST_DIR}/../sentinel1/io/orbits/sentinel1
        #        ${CMAKE_CURRENT_LIST_DIR}/../sentinel1/jlinda-core
        #        ${CMAKE_CURRENT_LIST_DIR}/../sentinel1/s1tbx-io/sentinel1
        #        ${CMAKE_CURRENT_LIST_DIR}/../sentinel1/s1tbx-io/geotiffxml
        #        ${CMAKE_CURRENT_LIST_DIR}/../sentinel1/s1tbx-commons/io
        #other
        ${CMAKE_CURRENT_LIST_DIR}/../util/cuda/include
        ${CMAKE_CURRENT_LIST_DIR}/../util/common/include
        ${CMAKE_CURRENT_LIST_DIR}/../external/pugixml/include
        )


add_library(snap-engine-static
        STATIC
        ${SNAP-ENGINE_SOURCES}
        ${SNAP-ENGINE_SOURCES_CU}
        )


target_include_directories(snap-engine-static
        PUBLIC
        ${SNAP-ENGINE_INCLUDES}
        )

set_target_properties(snap-engine-static
        PROPERTIES
        OUTPUT_NAME snap-engine
        )

target_link_libraries(snap-engine-static
        PUBLIC
        Boost::filesystem
        Boost::iostreams
        exotic_operations-static
        gdal-util-static
        pugixml-static
        staticZipper
        common-util-static
        )