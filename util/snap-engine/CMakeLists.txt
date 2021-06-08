
list(APPEND SNAP-ENGINE_SOURCES_CU
        ${CMAKE_CURRENT_LIST_DIR}/src/GeoUtils.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/earth_gravitational_model96.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/srtm3_elevation_model.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/srtm3_formatter.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/tie_point_grid.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/tie_point_geocoding.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/crs_geocoding.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/triangular_interpolation.cu
        )
list(APPEND SNAP-ENGINE_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/data_node.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/dataio/product_subset_def.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/dataio/abstract_product_reader.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/dataio/product_i_o.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/dataio/product_subset_builder.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/dataio/abstract_product_builder.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/date_utils.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/guardian.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/geo_pos.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-engine-utilities/datamodel/metadata/abstract_metadata.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-engine-utilities/datamodel/metadata/abstract_metadata_i_o.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/metadata_attribute.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/metadata_element.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/parse_exception.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/product.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/product_data.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/product_data_ascii.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/product_data_byte.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/product_data_double.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/product_data_float.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/product_data_int.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/product_data_long.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/product_data_short.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/product_data_ubyte.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/product_data_uint.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/product_data_ushort.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/product_data_utc.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/product_data_utc.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/product_node.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/product_node_list.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/product_node_group.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/abstract_band.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/abstract_geo_coding.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/raster_data_node.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/sample_coding.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/tie_point_grid.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/tie_point_geo_coding.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/virtual_band.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/band.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/flag_coding.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/index_coding.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/quicklooks/quicklook.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/mask.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/pugixml_meta_data_reader.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/pugixml_meta_data_writer.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/subset/abstract_subset_region.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/subset/pixel_subset_region.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/util/product_utils.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/util/string_utils.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/util/math/f_x_y_sum.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/util/math/f_x_y_sum_linear.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/util/math/f_x_y_sum_bi_linear.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/util/math/f_x_y_sum_cubic.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/util/math/f_x_y_sum_bi_cubic.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/util/math/f_x_y_sum_quadric.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/util/math/f_x_y_sum_bi_quadric.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/util/math/i_f_x.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/util/math/i_f_x_y.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/util/math/approximator.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/util/geo_utils.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/util/io/file_utils.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/util/math/math_utils.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-engine-utilities/util/maths.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-engine-utilities/util/test_utils.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-engine-utilities/util/zip_utils.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-engine-utilities/orbit_vector.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-engine-utilities/gpf/input_product_validator.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-engine-utilities/gpf/operator_utils.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-engine-utilities/gpf/stack_utils.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-engine-utilities/gpf/reader_utils.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-engine-utilities/datamodel/unit.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/ceres-core/i_virtual_dir.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/ceres-core/zip.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/ceres-core/dir.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/custom/gdal_image_reader.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/custom/gdal_image_writer.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/custom/io_tile.cc
        )

list(APPEND SNAP-ENGINE_INCLUDES
        #snap-engine
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/include/custom
        ${CMAKE_CURRENT_LIST_DIR}/include/ceres-core
        ${CMAKE_CURRENT_LIST_DIR}/include/snap-core/datamodel
        ${CMAKE_CURRENT_LIST_DIR}/include/snap-core/datamodel/quicklooks
        ${CMAKE_CURRENT_LIST_DIR}/include/snap-core/dataio
        ${CMAKE_CURRENT_LIST_DIR}/include/snap-core/util
        ${CMAKE_CURRENT_LIST_DIR}/include/snap-core/util/math
        ${CMAKE_CURRENT_LIST_DIR}/include/snap-core/util/io
        ${CMAKE_CURRENT_LIST_DIR}/include/snap-core/subset
        ${CMAKE_CURRENT_LIST_DIR}/include/snap-engine-utilities
        ${CMAKE_CURRENT_LIST_DIR}/include/snap-engine-utilities/util
        ${CMAKE_CURRENT_LIST_DIR}/include/snap-engine-utilities/datamodel
        ${CMAKE_CURRENT_LIST_DIR}/include/snap-engine-utilities/datamodel/metadata
        ${CMAKE_CURRENT_LIST_DIR}/include/snap-engine-utilities/eo
        ${CMAKE_CURRENT_LIST_DIR}/include/snap-engine-utilities/gpf
        sentinel1
        #        ${CMAKE_CURRENT_LIST_DIR}/../sentinel1/include
        #        ${CMAKE_CURRENT_LIST_DIR}/../sentinel1/include/io/orbits/sentinel1
        #        ${CMAKE_CURRENT_LIST_DIR}/../sentinel1/include/jlinda-core
        #        ${CMAKE_CURRENT_LIST_DIR}/../sentinel1/include/s1tbx-io/sentinel1
        #        ${CMAKE_CURRENT_LIST_DIR}/../sentinel1/include/s1tbx-io/geotiffxml
        #        ${CMAKE_CURRENT_LIST_DIR}/../sentinel1/include/s1tbx-commons/io
        #other
        ${CMAKE_CURRENT_LIST_DIR}/../cuda/include
        ${CMAKE_CURRENT_LIST_DIR}/../common/include
        ${CMAKE_CURRENT_LIST_DIR}/../../external/pugixml/include
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