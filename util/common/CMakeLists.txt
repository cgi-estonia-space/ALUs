
list(APPEND COMMON_UTIL_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include
        )

list(APPEND COMMON_UTIL_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/src/alus_log.cc
    )

add_library(common-util-static
        STATIC
        ${COMMON_UTIL_SOURCES}
        )
target_include_directories(common-util-static
        PUBLIC
        ${COMMON_UTIL_INCLUDE_DIRS}
        )
set_target_properties(common-util-static
        PROPERTIES
        OUTPUT_NAME common-util
        )
target_link_libraries(common-util-static
        PUBLIC
        Boost::log
        Boost::log_setup
        staticZipper
        gdal-util-static
        )

