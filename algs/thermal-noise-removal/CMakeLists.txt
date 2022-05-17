list(APPEND THERMAL_NOISE_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/thermal_noise_utils.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/thermal_noise_kernel.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/thermal_noise_remover.cc
        )

list(APPEND THERMAL_NOISE_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include
        )

add_library(thermal-noise-static
        STATIC
        ${THERMAL_NOISE_SOURCES}
        )

target_include_directories(thermal-noise-static
        PUBLIC
        ${THERMAL_NOISE_INCLUDE_DIRS}
        )

set_target_properties(thermal-noise-static
        PROPERTIES
        OUTPUT_NAME thermal-noise
        )

target_link_libraries(thermal-noise-static
        PUBLIC
        sentinel1-util-static
        snap-engine-static
        Boost::iostreams
        )

if(ENABLE_TESTS)
    add_subdirectory(unit-test)
endif()