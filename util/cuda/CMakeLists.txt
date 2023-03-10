list(APPEND CUDA_UTIL_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/cuda_device.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/cuda_device_init.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/memory_policy.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/cuda_util.cu
        )

list(APPEND CUDA_UTIL_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include
        )

add_library(cuda-util-static
        STATIC
        ${CUDA_UTIL_SOURCES}
        )
target_include_directories(cuda-util-static
        PUBLIC
        ${CUDA_UTIL_INCLUDE_DIRS}
        )
set_target_properties(cuda-util-static
        PROPERTIES
        OUTPUT_NAME cuda-util
        )
target_link_libraries(cuda-util-static
        PUBLIC
        )

