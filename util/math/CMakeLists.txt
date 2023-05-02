
list(APPEND MATH_UTIL_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include)

add_library(alus-math-static INTERFACE)

target_include_directories(alus-math-static
        INTERFACE
        $<BUILD_INTERFACE:${MATH_UTIL_INCLUDE_DIRS}>
        $<INSTALL_INTERFACE:include>)