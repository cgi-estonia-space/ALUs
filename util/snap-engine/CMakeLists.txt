
list(APPEND SNAP-ENGINE_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/GeoUtils.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/earth_gravitational_model96.cc
        )

list(APPEND SNAP-ENGINE_INCLUDES
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/../cuda/include
        ${CMAKE_CURRENT_LIST_DIR}/../exotic_operations/include
        )

add_library(snap-engine-obj
        OBJECT
        ${SNAP-ENGINE_SOURCES}
        )
target_include_directories(snap-engine-obj
        PRIVATE
        ${SNAP-ENGINE_INCLUDES}
        )

add_library(snap-engine-static
        STATIC
        $<TARGET_OBJECTS:snap-engine-obj>
        )
target_include_directories(snap-engine-static
        PUBLIC
        ${SNAP-ENGINE_INCLUDES}
        )
set_target_properties(snap-engine-static
        PROPERTIES
        OUTPUT_NAME snap-engine
        )
target_link_libraries(snap-engine-static
        PUBLIC
        exotic_operations-static
        )
