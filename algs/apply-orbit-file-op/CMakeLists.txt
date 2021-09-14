list(APPEND APPLY_ORBIT_FILE_OP_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/apply_orbit_file_op.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/gdal_data_copy.cc
        )

list(APPEND APPLY_ORBIT_FILE_OP_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/../algorithm_bond/include
        ${CMAKE_CURRENT_LIST_DIR}/../../app/include
        #      SENTINEL1
        ${CMAKE_CURRENT_LIST_DIR}/../../util/sentinel1/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/sentinel1/include/io/orbits/sentinel1
        ${CMAKE_CURRENT_LIST_DIR}/../../util/sentinel1/include/jlinda-core
        ${CMAKE_CURRENT_LIST_DIR}/../../util/sentinel1/include/s1tbx-io/sentinel1
        ${CMAKE_CURRENT_LIST_DIR}/../../util/sentinel1/include/s1tbx-commons/io
        #      SNAP-ENGINE
        ${CMAKE_CURRENT_LIST_DIR}/../../util/snap-engine/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/snap-engine/include/ceres-core
        ${CMAKE_CURRENT_LIST_DIR}/../../util/snap-engine/include/snap-core/datamodel
        ${CMAKE_CURRENT_LIST_DIR}/../../util/snap-engine/include/snap-core/dataio
        ${CMAKE_CURRENT_LIST_DIR}/../../util/snap-engine/include/snap-core/util
        ${CMAKE_CURRENT_LIST_DIR}/../../util/snap-engine/include/snap-core/util/math
        ${CMAKE_CURRENT_LIST_DIR}/../../util/snap-engine/include/snap-core/subset
        ${CMAKE_CURRENT_LIST_DIR}/../../util/snap-engine/include/snap-engine-utilities
        ${CMAKE_CURRENT_LIST_DIR}/../../util/snap-engine/include/snap-engine-utilities/util
        ${CMAKE_CURRENT_LIST_DIR}/../../util/snap-engine/include/snap-engine-utilities/datamodel
        #      OTHER
        ${CMAKE_CURRENT_LIST_DIR}/../../external/pugixml/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/gdal/include
        ${CMAKE_CURRENT_LIST_DIR}/../../common/include
        )

add_library(apply-orbit-file-op-static
        STATIC
        ${APPLY_ORBIT_FILE_OP_SOURCES}
        )
target_include_directories(apply-orbit-file-op-static
        PUBLIC
        ${APPLY_ORBIT_FILE_OP_INCLUDE_DIRS}
        )
set_target_properties(apply-orbit-file-op-static
        PROPERTIES
        OUTPUT_NAME apply-orbit-file-op
        )
target_link_libraries(apply-orbit-file-op-static
        PUBLIC
        gdal
        snap-engine-static
        sentinel1-util-static
        exotic_operations-static
        pugixml-static
        common-util-static
        )

