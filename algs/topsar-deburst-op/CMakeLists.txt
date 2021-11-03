list(APPEND TOPSAR_DEBURST_OP_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/topsar_deburst_op.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/topsar_deburst_rectangle_generator.cc
        )

list(APPEND TOPSAR_DEBURST_OP_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/../algorithm_bond/include
        ${CMAKE_CURRENT_LIST_DIR}/../../app/include
        #      SENTINEL1
        ${CMAKE_CURRENT_LIST_DIR}/../../sentinel1
        ${CMAKE_CURRENT_LIST_DIR}/../../sentinel1/io/orbits/sentinel1
        ${CMAKE_CURRENT_LIST_DIR}/../../sentinel1/jlinda/jlinda-core
        ${CMAKE_CURRENT_LIST_DIR}/../../sentinel1/s1tbx-io/sentinel1
        ${CMAKE_CURRENT_LIST_DIR}/../../sentinel1/s1tbx-commons/io
        #      SNAP-ENGINE
        ${CMAKE_CURRENT_LIST_DIR}/../../snap-engine
        ${CMAKE_CURRENT_LIST_DIR}/../../snap-engine/ceres-core
        ${CMAKE_CURRENT_LIST_DIR}/../../snap-engine/snap-core/core/datamodel
        ${CMAKE_CURRENT_LIST_DIR}/../../snap-engine/snap-core/core/dataio
        ${CMAKE_CURRENT_LIST_DIR}/../../snap-engine/snap-core/core/util
        ${CMAKE_CURRENT_LIST_DIR}/../../snap-engine/snap-core/core/util/math
        ${CMAKE_CURRENT_LIST_DIR}/../../snap-engine/snap-core/core/subset
        ${CMAKE_CURRENT_LIST_DIR}/../../snap-engine/snap-engine-utilities
        ${CMAKE_CURRENT_LIST_DIR}/../../snap-engine/snap-engine-utilities/engine-utilities/util
        ${CMAKE_CURRENT_LIST_DIR}/../../snap-engine/snap-engine-utilities/engine-utilities/datamodel
        #      OTHER
        ${CMAKE_CURRENT_LIST_DIR}/../../util/exotic_operations/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/cuda/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/common/include
        ${CMAKE_CURRENT_LIST_DIR}/../../external/pugixml/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/gdal/include
        ${CMAKE_CURRENT_LIST_DIR}/../../common/include
        )

add_library(topsar-deburst-op-obj
        OBJECT
        ${TOPSAR_DEBURST_OP_SOURCES}
        )
target_include_directories(topsar-deburst-op-obj
        PRIVATE
        ${TOPSAR_DEBURST_OP_INCLUDE_DIRS}
        )

add_library(topsar-deburst-op-static
        STATIC
        $<TARGET_OBJECTS:topsar-deburst-op-obj>
        )
target_include_directories(topsar-deburst-op-static
        PUBLIC
        ${TOPSAR_DEBURST_OP_INCLUDE_DIRS}
        )
set_target_properties(topsar-deburst-op-static
        PROPERTIES
        OUTPUT_NAME topsar-deburst-op
        )
target_link_libraries(topsar-deburst-op-static
        PUBLIC
        gdal
        snap-engine-static
        sentinel1-util-static
        pugixml-static
        )

