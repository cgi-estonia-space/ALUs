
list(APPEND TENSORFLOW_INCLUDES
        ${CMAKE_CURRENT_LIST_DIR}/../../tensorflow/ubuntu_1804/include
        ${CMAKE_CURRENT_LIST_DIR}/../../tensorflow/ubuntu_1804/include/external/com_google_absl
        # For error_codes.pb.h etc.
        ${CMAKE_CURRENT_LIST_DIR}/../../tensorflow/ubuntu_1804/include/bazel-out/host/bin
        # For port_def.inc etc.
        ${CMAKE_CURRENT_LIST_DIR}/../../tensorflow/ubuntu_1804/include/external/com_google_protobuf/src
        # For unsupported/Eigen/CXX11/Tensor
        ${CMAKE_CURRENT_LIST_DIR}/../../tensorflow/ubuntu_1804/include/external/eigen_archive
        )

list(APPEND TENSORFLOW_LIBS
        ${CMAKE_CURRENT_LIST_DIR}/../../tensorflow/ubuntu_1804/lib/libtensorflow_cc.so
        ${CMAKE_CURRENT_LIST_DIR}/../../tensorflow/ubuntu_1804/lib/libtensorflow_framework.so
        )

list(APPEND COHERENCE_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/io_tile.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/tf_algorithm_runner.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/gdal_tile_writer.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/gdal_tile_reader.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/point.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/coh_tiles_generator.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/coh_tile.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/meta_data.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/orbit.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/utils.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/coherence_calc.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/data_bands_buffer.cc
        )

list(APPEND COHERENCE_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/gdal/include
        ${TENSORFLOW_INCLUDES}
        )

add_library(coherence-obj
        OBJECT
        ${COHERENCE_SOURCES}
        )
target_include_directories(coherence-obj
        PRIVATE
        ${COHERENCE_INCLUDE_DIRS}
        )
target_compile_options(coherence-obj
        PRIVATE
        -Wno-pedantic
        -Wno-unused-parameter
        -Wno-ignored-qualifiers
        -Wno-unused-variable
        -Wno-narrowing
        )

add_library(coherence-static
        STATIC
        $<TARGET_OBJECTS:coherence-obj>
        )
target_include_directories(coherence-static
        PUBLIC
        ${COHERENCE_INCLUDE_DIRS}
        )
set_target_properties(coherence-static
        PROPERTIES
        OUTPUT_NAME coherence
        )
target_link_libraries(coherence-static
        PUBLIC
        gdal
        ${TENSORFLOW_LIBS}
        )
#this is main executable, no need to build object out of example.cc
add_executable(coherence src/example_coh.cc)
target_link_libraries(coherence
        PRIVATE
        coherence-static
        )
target_compile_options(coherence
        PRIVATE
        -Wno-pedantic
        -Wno-unused-parameter
        -Wno-ignored-qualifiers
        )