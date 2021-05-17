
list(APPEND GDAL_UTIL_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/dataset.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/dem.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/target_dataset.cc
        )

list(APPEND GDAL_UTIL_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/../common/include
        ${CMAKE_CURRENT_LIST_DIR}/../cuda/include
        )

add_library(gdal-util-obj
        OBJECT
        ${GDAL_UTIL_SOURCES}
        include/transform_constants.h)
target_include_directories(gdal-util-obj
        PRIVATE
        ${GDAL_UTIL_INCLUDE_DIRS}
        )

add_library(gdal-util-static
        STATIC
        $<TARGET_OBJECTS:gdal-util-obj>
        )
target_include_directories(gdal-util-static
        PUBLIC
        ${GDAL_UTIL_INCLUDE_DIRS}
        )
set_target_properties(gdal-util-static
        PROPERTIES
        OUTPUT_NAME gdal-util
        )
target_link_libraries(gdal-util-static
        PUBLIC
        gdal
        cuda_util-static
        Boost::filesystem
        )
