
list(APPEND MATH_UTIL_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include)

add_library(math-util-static INTERFACE)

target_include_directories(math-util-static
        INTERFACE
        $<BUILD_INTERFACE:${MATH_UTIL_INCLUDE_DIRS}>
        $<INSTALL_INTERFACE:include>)

if (ENABLE_TESTS)
    add_subdirectory(unit-test)
endif ()
