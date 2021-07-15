list(APPEND COREGISTRATION_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/coregistration_controller.cc
        )

list(APPEND COREGISTRATION_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/../algorithm_bond/include
        ${CMAKE_CURRENT_LIST_DIR}/../backgeocoding/include
        ${CMAKE_CURRENT_LIST_DIR}/../topsar_split/include
        ${CMAKE_CURRENT_LIST_DIR}/../apply-orbit-file-op/include
        ${CMAKE_CURRENT_LIST_DIR}/../../app/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/gdal/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/cuda/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/exotic_operations/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/common/include
        ${CMAKE_CURRENT_LIST_DIR}/../../sentinel1
        ${CMAKE_CURRENT_LIST_DIR}/../../snap-engine
        ${CMAKE_CURRENT_LIST_DIR}/../../external/pugixml/include
        )

add_library(coregistration-obj
        OBJECT
        ${COREGISTRATION_SOURCES}
        )
target_include_directories(coregistration-obj
        PRIVATE
        ${COREGISTRATION_INCLUDE_DIRS}
        )

add_library(coregistration-static
        STATIC
        $<TARGET_OBJECTS:coregistration-obj>
        )
target_include_directories(coregistration-static
        PUBLIC
        ${COREGISTRATION_INCLUDE_DIRS}
        )
set_target_properties(coregistration-static
        PROPERTIES
        OUTPUT_NAME coregistration
        )
target_link_libraries(coregistration-static
        PUBLIC
        gdal
        gdal-util-static
        backgeocoding-static
        topsar-split-static
        apply-orbit-file-op-static
        exotic_operations-static
        sentinel1-util-static
        snap-engine-static
        Boost::program_options
        )

add_library(coregistration-shared
        SHARED
        $<TARGET_OBJECTS:coregistration-obj>
        )
target_include_directories(coregistration-shared
        PUBLIC
        ${COREGISTRATION_INCLUDE_DIRS}
        )
set_target_properties(coregistration-shared
        PROPERTIES
        OUTPUT_NAME coregistration
        LIBRARY_OUTPUT_DIRECTORY ${ALUS_BINARY_OUTPUT_DIR}
        )

target_link_libraries(coregistration-shared
        PUBLIC
        exotic_operations-static
        sentinel1-util-static
        snap-engine-static
        backgeocoding-static
        Boost::program_options
        )