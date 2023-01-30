
list(APPEND GDAL_UTIL_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/dataset.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/target_dataset.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/c16_dataset.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/gdal_util.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/gdal_management.cc
        )

list(APPEND GDAL_UTIL_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include
        )

add_library(gdal-util-static
        STATIC
        ${GDAL_UTIL_SOURCES}
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
        cuda-util-static
        common-util-static
        Boost::filesystem
        )
