
list(APPEND BACKGEOCODING_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/backgeocoding.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/backgeocoding_controller.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/backgeocoding_worker.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/bilinear.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/deramp_demod.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/slave_pixpos.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/extended_amount.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/burst_offset.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/elevation_mask.cu
        )

list(APPEND BACKGEOCODING_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/gdal/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/common/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/exotic-operations/include
        ${CMAKE_CURRENT_LIST_DIR}/../../sentinel1
        ${CMAKE_CURRENT_LIST_DIR}/../../snap-engine
        ${CMAKE_CURRENT_LIST_DIR}/../../snap-engine/snap-core/core/datamodel
        ${CMAKE_CURRENT_LIST_DIR}/../../util/cuda/include
        ${CMAKE_CURRENT_LIST_DIR}/../../app/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/common/include
        )

add_library(backgeocoding-static
        STATIC
        ${BACKGEOCODING_SOURCES}
        )
target_include_directories(backgeocoding-static
        PUBLIC
        ${BACKGEOCODING_INCLUDE_DIRS}
        )
set_target_properties(backgeocoding-static
        PROPERTIES
        OUTPUT_NAME backgeocoding
        )
target_link_libraries(backgeocoding-static
        PUBLIC
        common-util-static
        dem-util-static
        gdal-util-static
        exotic-operations-static
        sentinel1-util-static
        cuda-util-static
        snap-engine-static
        Boost::program_options
        )

