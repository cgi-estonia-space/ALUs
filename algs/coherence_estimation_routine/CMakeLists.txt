list(APPEND COHERENCE_ESTIMATION_ROUTINE_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/coherence_estimation_routine_execute.cc
        )

list(APPEND COHERENCE_ESTIMATION_ROUTINE_INCLUDES
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/../algorithm_bond/include
        ${CMAKE_CURRENT_LIST_DIR}/../../app/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/exotic_operations/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/cuda/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/gdal/include
        )

add_library(coherence-estimation-routine-obj
        OBJECT
        ${COHERENCE_ESTIMATION_ROUTINE_SOURCES}
        )

target_include_directories(coherence-estimation-routine-obj
        PRIVATE
        ${COHERENCE_ESTIMATION_ROUTINE_INCLUDES}
        )

add_library(coherence-estimation-routine-static
        STATIC
        $<TARGET_OBJECTS:coherence-estimation-routine-obj>
        )

target_include_directories(coherence-estimation-routine-static
        PUBLIC
        ${COHERENCE_ESTIMATION_ROUTINE_INCLUDES}
        )

target_link_libraries(coherence-estimation-routine-static
        PUBLIC
        alg-bond-static
        gdal-util-static
        )

set_target_properties(coherence-estimation-routine-static
        PROPERTIES
        OUTPUT_NAME coherence_estimation_routine
        )

add_library(coherence-estimation-routine-shared
        SHARED
        $<TARGET_OBJECTS:coherence-estimation-routine-obj>
        )

target_include_directories(coherence-estimation-routine-shared
        PUBLIC
        ${COHERENCE_ESTIMATION_ROUTINE_INCLUDES}
        )

target_link_libraries(coherence-estimation-routine-shared
        PUBLIC
        alg-bond-static
        gdal-util-static
        )

set_target_properties(coherence-estimation-routine-shared
        PROPERTIES
        OUTPUT_NAME coherence-estimation-routine
        LIBRARY_OUTPUT_DIRECTORY ${ALUS_BINARY_OUTPUT_DIR}
        )
