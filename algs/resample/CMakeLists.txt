
list(APPEND RESAMPLE_SOURCES
        src/cli_args.cc
        src/execute.cc
        src/projection.cc)

list(APPEND RESAMPLE_INCLUDES
        ${CMAKE_CURRENT_LIST_DIR}/include)

add_library(resample-static
        STATIC
        ${RESAMPLE_SOURCES})

target_include_directories(resample-static
        PUBLIC
        ${RESAMPLE_INCLUDES})

target_link_libraries(resample-static
        PUBLIC
        Boost::program_options
        app-util-static
        common-util-static)

set_target_properties(resample-static
        PROPERTIES
        OUTPUT_NAME resample)

target_compile_definitions(resample-static PRIVATE APP_NAME="Resample")

add_executable(alus-resa src/main.cc)

target_compile_definitions(alus-resa PRIVATE APP_NAME="Resample")

target_link_libraries(alus-resa
        PRIVATE
        resample-static
        )

set_target_properties(alus-resa
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${ALUS_BINARY_OUTPUT_DIR}
        )

if(ENABLE_TESTS)
    add_subdirectory(unit-test)
endif()
