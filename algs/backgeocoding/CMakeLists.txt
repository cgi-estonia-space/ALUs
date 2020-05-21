
list(APPEND BACKGEOCODING_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/backgeocoding.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/backgeocoding_controller.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/bilinear.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/deramp_demod.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/slave_pixpos.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/dem_formatter.cu
        )

list(APPEND BACKGEOCODING_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/gdal/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/common/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/exotic_operations/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/sentinel1/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/snap-engine/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/cuda/include
        )

add_library(backgeocoding-obj
        OBJECT
        ${BACKGEOCODING_SOURCES}
        )
target_include_directories(backgeocoding-obj
        PRIVATE
        ${BACKGEOCODING_INCLUDE_DIRS}
        )

add_library(backgeocoding-static
        STATIC
        $<TARGET_OBJECTS:backgeocoding-obj>
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
        gdal-util-static
        exotic_operations-static
        sentinel1-util-static
        cuda_util-static
        )
