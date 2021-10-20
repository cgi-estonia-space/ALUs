list(APPEND COHERENCE_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/io_tile.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/cuda_algorithm_runner.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/gdal_tile_writer.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/gdal_tile_reader.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/coh_tiles_generator.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/coh_tile.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/meta_data.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/coherence_calc_cuda.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/coherence_computation.cu
        )

list(APPEND COHERENCE_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/../algorithm_bond/include
        ${CMAKE_CURRENT_LIST_DIR}/../../app/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/common/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/exotic_operations/include
        #      SENTINEL1
        ${CMAKE_CURRENT_LIST_DIR}/../../sentinel1
        ${CMAKE_CURRENT_LIST_DIR}/../../sentinel1/io/orbits/sentinel1
        ${CMAKE_CURRENT_LIST_DIR}/../../sentinel1/jlinda/jlinda-core
        ${CMAKE_CURRENT_LIST_DIR}/../../sentinel1/s1tbx-io/sentinel1
        ${CMAKE_CURRENT_LIST_DIR}/../../sentinel1/s1tbx-commons/io
        #      SNAP-ENGINE
        ${CMAKE_CURRENT_LIST_DIR}/../../snap-engine
        ${CMAKE_CURRENT_LIST_DIR}/../../snap-engine/ceres-core
        ${CMAKE_CURRENT_LIST_DIR}/../../snap-engine/snap-core/core/datamodel
        ${CMAKE_CURRENT_LIST_DIR}/../../snap-engine/snap-core/dataio
        ${CMAKE_CURRENT_LIST_DIR}/../../snap-engine/snap-core/util
        ${CMAKE_CURRENT_LIST_DIR}/../../snap-engine/snap-core/util/math
        ${CMAKE_CURRENT_LIST_DIR}/../../snap-engine/snap-core/subset
        ${CMAKE_CURRENT_LIST_DIR}/../../snap-engine/snap-engine-utilities
        ${CMAKE_CURRENT_LIST_DIR}/../../snap-engine/snap-engine-utilities/util
        ${CMAKE_CURRENT_LIST_DIR}/../../snap-engine/snap-engine-utilities/datamodel
        #       other
        ${CMAKE_CURRENT_LIST_DIR}/../../external/pugixml/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/gdal/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/cuda/include
        ${CMAKE_CURRENT_LIST_DIR}/../../common/include
        )

add_library(coherence-cuda-static
        STATIC
        ${COHERENCE_SOURCES}
        )
target_include_directories(coherence-cuda-static
        PUBLIC
        ${COHERENCE_INCLUDE_DIRS}
        )
set_target_properties(coherence-cuda-static
        PROPERTIES
        OUTPUT_NAME coherence-cuda
        )
target_link_libraries(coherence-cuda-static
        PUBLIC
        Boost::filesystem
        Boost::iostreams
        gdal
        snap-engine-static
        sentinel1-util-static
        pugixml-static
        cuda-util-static
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_cusolver_LIBRARY}
        )
