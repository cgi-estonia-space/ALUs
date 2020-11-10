
list(APPEND SNAP-ENGINE_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/GeoUtils.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/earth_gravitational_model96.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/srtm3_elevation_model.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/srtm3_formatter.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/tie_point_grid.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/tie_point_geocoding.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/crs_geocoding.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/triangular_interpolation.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/data_node.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/date_utils.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/guardian.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/i_meta_data_reader.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/meta_data_node_names.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/metadata_attribute.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/metadata_element.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/parse_exception.cc
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
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/product_node.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/pugixml_meta_data_reader.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/string_utils.cc
        )

list(APPEND SNAP-ENGINE_INCLUDES
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/include/snap-core/datamodel
        ${CMAKE_CURRENT_LIST_DIR}/../cuda/include
        ${CMAKE_CURRENT_LIST_DIR}/../common/include
        )

add_library(snap-engine-static
        STATIC
        ${SNAP-ENGINE_SOURCES}
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
        exotic_operations-static
        gdal-util-static
        pugixml-static
        )
