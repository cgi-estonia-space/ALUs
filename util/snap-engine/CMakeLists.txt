
list(APPEND SNAP-ENGINE_SOURCES_CU
        ${CMAKE_CURRENT_LIST_DIR}/src/GeoUtils.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/earth_gravitational_model96.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/srtm3_elevation_model.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/srtm3_formatter.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/tie_point_grid.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/tie_point_geocoding.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/crs_geocoding.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/triangular_interpolation.cu
        )
list(APPEND SNAP-ENGINE_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/data_node.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/date_utils.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/guardian.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/meta_data_node_names.cc
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
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/product_node.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/raster_data_node.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/sample_coding.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/tie_point_grid.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/virtual_band.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/band.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/flag_coding.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/index_coding.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/mask.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/pugixml_meta_data_reader.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/datamodel/pugixml_meta_data_writer.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-core/util/product_utils.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-engine-utilities/math_utils.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-engine-utilities/util/maths.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/snap-engine-utilities/orbit_vector.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/string_utils.cc
        )

list(APPEND SNAP-ENGINE_INCLUDES
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/include/snap-core/datamodel
        ${CMAKE_CURRENT_LIST_DIR}/include/snap-core/util
        ${CMAKE_CURRENT_LIST_DIR}/include/snap-core/util/math
        ${CMAKE_CURRENT_LIST_DIR}/include/ceres-core
        ${CMAKE_CURRENT_LIST_DIR}/include/snap-engine-utilities
        ${CMAKE_CURRENT_LIST_DIR}/include/snap-engine-utilities/util
        ${CMAKE_CURRENT_LIST_DIR}/../cuda/include
        ${CMAKE_CURRENT_LIST_DIR}/../common/include
        ${CMAKE_CURRENT_LIST_DIR}/../../external/pugixml/include
        )

add_library(snapengine-obj-cu
        OBJECT
        ${SNAP-ENGINE_SOURCES_CU}
        )
target_include_directories(snapengine-obj-cu
        PRIVATE
        ${SNAP-ENGINE_INCLUDES}
        )

add_library(snapengine-obj
        OBJECT
        ${SNAP-ENGINE_SOURCES}
        )
target_include_directories(snapengine-obj
        PRIVATE
        ${SNAP-ENGINE_INCLUDES}
        )

target_link_libraries(snapengine-obj-cu
        PUBLIC
        exotic_operations-static
        gdal-util-static
        pugixml-static
        )
target_link_libraries(snapengine-obj
        PUBLIC
        Boost::filesystem
        Boost::iostreams
        exotic_operations-static
        gdal-util-static
        pugixml-static
        )

add_library(snap-engine-static
        STATIC
        $<TARGET_OBJECTS:snapengine-obj>
        $<TARGET_OBJECTS:snapengine-obj-cu>
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
        )