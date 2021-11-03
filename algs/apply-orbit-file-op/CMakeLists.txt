list(APPEND APPLY_ORBIT_FILE_OP_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/apply_orbit_file_op.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/gdal_data_copy.cc
        )

list(APPEND APPLY_ORBIT_FILE_OP_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include
        #      SENTINEL1
        ${CMAKE_CURRENT_LIST_DIR}/../../sentinel1/
        ${CMAKE_CURRENT_LIST_DIR}/../../sentinel1/io/orbits/sentinel1
        ${CMAKE_CURRENT_LIST_DIR}/../../sentinel1/jlinda-core
        ${CMAKE_CURRENT_LIST_DIR}/../../sentinel1/s1tbx-io/sentinel1
        ${CMAKE_CURRENT_LIST_DIR}/../../sentinel1/s1tbx-commons/io
        #      SNAP-ENGINE
        ${CMAKE_CURRENT_LIST_DIR}/../../snap-engine/ceres-core
        ${CMAKE_CURRENT_LIST_DIR}/../../snap-engine/snap-core/datamodel
        ${CMAKE_CURRENT_LIST_DIR}/../../snap-engine/snap-core/dataio
        ${CMAKE_CURRENT_LIST_DIR}/../../snap-engine/snap-core/util
        ${CMAKE_CURRENT_LIST_DIR}/../../snap-engine/snap-core/util/math
        ${CMAKE_CURRENT_LIST_DIR}/../../snap-engine/snap-core/subset
        ${CMAKE_CURRENT_LIST_DIR}/../../snap-engine/snap-engine-utilities
        ${CMAKE_CURRENT_LIST_DIR}/../../snap-engine/snap-engine-utilities/util
        ${CMAKE_CURRENT_LIST_DIR}/../../snap-engine/snap-engine-utilities/datamodel
        #      OTHER
        ${CMAKE_CURRENT_LIST_DIR}/../../external/pugixml/include
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
        exotic-operations-static
        pugixml-static
        common-util-static
        )

