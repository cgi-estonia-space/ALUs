
list(APPEND CUDA_ALG_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/cuda_alg.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/cuda_alg.cu)

list(APPEND CUDA_ALG_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include)

add_library(cuda_alg-obj
        OBJECT
        ${CUDA_ALG_SOURCES})

target_include_directories(cuda_alg-obj
        PRIVATE
        ${CUDA_ALG_INCLUDE_DIRS})

add_library(cuda_alg-static
        STATIC
        $<TARGET_OBJECTS:cuda_alg-obj>
        )
target_include_directories(cuda_alg-static
        PUBLIC
        ${CUDA_ALG_INCLUDE_DIRS}
        )
set_target_properties(cuda_alg-static
        PROPERTIES
        OUTPUT_NAME cuda_alg
        )

#list(APPEND CUDA_ALG_CU_SOURCES
#        ${CMAKE_CURRENT_LIST_DIR}/src/cuda_alg.cu)
#
#add_library(cuda_alg_cu-obj
#        OBJECT
#        ${CUDA_ALG_CU_SOURCES}
#        )
#
#add_library(cuda_alg_cu-static
#        STATIC
#        $<TARGET_OBJECTS:cuda_alg_cu-obj>
#        )
#
#set_target_properties(cuda_alg_cu-static
#        PROPERTIES
#        OUTPUT_NAME cuda_alg_cu
#        )

