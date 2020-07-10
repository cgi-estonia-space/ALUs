list(APPEND EXOTIC_OPERATIONS_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/comparators.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/allocators.cc
        )

list(APPEND EXOTIC_OPERATIONS_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include
        )

add_library(exotic_operations-obj
        OBJECT
        ${EXOTIC_OPERATIONS_SOURCES}
        )
target_include_directories(exotic_operations-obj
        PRIVATE
        ${EXOTIC_OPERATIONS_INCLUDE_DIRS}
        )

add_library(exotic_operations-static
        STATIC
        $<TARGET_OBJECTS:exotic_operations-obj>
        )
target_include_directories(exotic_operations-static
        PUBLIC
        ${EXOTIC_OPERATIONS_INCLUDE_DIRS}
        )
set_target_properties(exotic_operations-static
        PROPERTIES
        OUTPUT_NAME exotic_operations
        )
target_link_libraries(exotic_operations-static
        PUBLIC
        gdal
        )
