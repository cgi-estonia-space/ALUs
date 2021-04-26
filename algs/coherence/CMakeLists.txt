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
        ${CMAKE_CURRENT_LIST_DIR}/src/coherence_execute.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/io_tile.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/tf_algorithm_runner.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/gdal_tile_writer.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/gdal_tile_reader.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/coh_tiles_generator.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/coh_tile.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/meta_data.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/coherence_calc.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/data_bands_buffer.cc
        )

list(APPEND COHERENCE_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/../algorithm_bond/include
        ${CMAKE_CURRENT_LIST_DIR}/../../app/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/common/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/exotic_operations/include
        #      SENTINEL1
        ${CMAKE_CURRENT_LIST_DIR}/../../util/sentinel1/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/sentinel1/include/io/orbits/sentinel1
        ${CMAKE_CURRENT_LIST_DIR}/../../util/sentinel1/include/jlinda-core
        ${CMAKE_CURRENT_LIST_DIR}/../../util/sentinel1/include/s1tbx-io/sentinel1
        ${CMAKE_CURRENT_LIST_DIR}/../../util/sentinel1/include/s1tbx-commons/io
        #      SNAP-ENGINE
        ${CMAKE_CURRENT_LIST_DIR}/../../util/snap-engine/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/snap-engine/include/ceres-core
        ${CMAKE_CURRENT_LIST_DIR}/../../util/snap-engine/include/snap-core/datamodel
        ${CMAKE_CURRENT_LIST_DIR}/../../util/snap-engine/include/snap-core/dataio
        ${CMAKE_CURRENT_LIST_DIR}/../../util/snap-engine/include/snap-core/util
        ${CMAKE_CURRENT_LIST_DIR}/../../util/snap-engine/include/snap-core/util/math
        ${CMAKE_CURRENT_LIST_DIR}/../../util/snap-engine/include/snap-core/subset
        ${CMAKE_CURRENT_LIST_DIR}/../../util/snap-engine/include/snap-engine-utilities
        ${CMAKE_CURRENT_LIST_DIR}/../../util/snap-engine/include/snap-engine-utilities/util
        ${CMAKE_CURRENT_LIST_DIR}/../../util/snap-engine/include/snap-engine-utilities/datamodel
        #       other
        ${CMAKE_CURRENT_LIST_DIR}/../../external/pugixml/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/gdal/include
        ${CMAKE_CURRENT_LIST_DIR}/../../common/include
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
        -Wno-unused-but-set-variable
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
        Boost::filesystem
        Boost::iostreams
        gdal
        snap-engine-static
        sentinel1-util-static
        pugixml-static
        ${TENSORFLOW_LIBS}
        )

add_library(coherence-shared
        SHARED
        $<TARGET_OBJECTS:coherence-obj>
        ${CMAKE_CURRENT_LIST_DIR}/src/coherence_execute_shared.cc
        )
target_include_directories(coherence-shared
        PUBLIC
        ${COHERENCE_INCLUDE_DIRS}
        )
set_target_properties(coherence-shared
        PROPERTIES
        OUTPUT_NAME coherence
        LIBRARY_OUTPUT_DIRECTORY ${ALUS_BINARY_OUTPUT_DIR}
        )
target_link_libraries(coherence-shared
        PUBLIC
        Boost::filesystem
        Boost::iostreams
        gdal
        snap-engine-static
        sentinel1-util-static
        pugixml-static
        ${TENSORFLOW_LIBS}
        )
add_custom_command(TARGET coherence-shared
        POST_BUILD
        COMMAND cp -P ${CMAKE_CURRENT_LIST_DIR}/../../tensorflow/ubuntu_1804/lib/* ${ALUS_BINARY_OUTPUT_DIR}/.
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

