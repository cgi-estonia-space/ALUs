
list(APPEND DEM_UTIL_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/dem_assistant.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/dem_egm96.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/copdem_cog_30m.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/copdem_cog_30m_calc.cu
        )

list(APPEND DEM_UTIL_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include
        )

add_library(dem-util-static
        STATIC
        ${DEM_UTIL_SOURCES}
        )
target_include_directories(dem-util-static
        PUBLIC
        ${DEM_UTIL_INCLUDE_DIRS}
        )
set_target_properties(dem-util-static
        PROPERTIES
        OUTPUT_NAME dem-util
        )
target_link_libraries(dem-util-static
        PUBLIC
        gdal
        exotic-operations-static
        snap-engine-static
        common-util-static
        )

if (ENABLE_TESTS)
    add_subdirectory(unit-test)
endif ()

