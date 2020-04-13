
list(APPEND GDAL_UTIL_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/dataset.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/dem.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/target_dataset.cpp
        )

list(APPEND GDAL_UTIL_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/../common/include
        )

add_library(gdal-util-obj
        OBJECT
        ${GDAL_UTIL_SOURCES}
        )
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
        )
