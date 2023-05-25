
list(APPEND MATH_UTIL_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include)

add_library(math-util-static INTERFACE)

target_include_directories(math-util-static
        INTERFACE
        $<BUILD_INTERFACE:${MATH_UTIL_INCLUDE_DIRS}>
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../cuda/include>
        $<INSTALL_INTERFACE:../cuda/include>)

list(APPEND MATH_UTIL_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/calc_kernels.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/filters.cu)

add_library(math-util-kernels-static
        STATIC
        ${MATH_UTIL_SOURCES})

target_include_directories(math-util-kernels-static
        PUBLIC
        ${MATH_UTIL_INCLUDE_DIRS}
        )

target_link_libraries(math-util-kernels-static
        PUBLIC
        cuda-util-static
        )

set_target_properties(math-util-kernels-static
        PROPERTIES
        OUTPUT_NAME math-util-kernels
        )

if (ENABLE_TESTS)
    add_subdirectory(unit-test)
endif ()
