
include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.8.1
)

FetchContent_GetProperties(googletest)

if(NOT googletest_POPULATED)
  FetchContent_Populate(googletest)

  # Prevent GoogleTest from overriding our compiler/linker options
  # when building with Visual Studio
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  # Prevent GoogleTest from using PThreads
  set(gtest_disable_pthreads ON CACHE BOOL "" FORCE)

  # adds the targers: gtest, gtest_main, gmock, gmock_main
  add_subdirectory(
    ${googletest_SOURCE_DIR}
    ${googletest_BINARY_DIR}
    )

  # Silence std::tr1 warning on MSVC
  if(MSVC)
    foreach(_tgt gtest gtest_main gmock gmock_main)
      target_compile_definitions(${_tgt}
        PRIVATE
          "_SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING"
        )
    endforeach()
  endif()
endif()

list(APPEND TEST_SOURCES
        src/dataset_test.cpp
        src/terrain_correction_test.cpp)

add_executable(unit_test ${TEST_SOURCES})
target_link_libraries(unit_test
        PRIVATE
        gmock_main
        gdal-util-static
        terrain-correction-static
        )

add_executable(map_cut util/map_cut.cpp)
target_link_libraries(map_cut
        PRIVATE
        gdal-util-static
        )

add_custom_command(TARGET unit_test
        POST_BUILD
        COMMAND cp -r ${CMAKE_CURRENT_LIST_DIR}/goods ${CMAKE_BINARY_DIR}/test/.
        )
