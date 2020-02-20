list(APPEND INVERTER_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/inverter_parallel.cpp)
list(APPEND INVERTER_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/ThreadController.cpp)
list(APPEND INVERTER_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/ThreadHolder.cpp)

list(APPEND INVERTER_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include)

add_library(inverter_parallel-obj
        OBJECT
        ${INVERTER_SOURCES})

target_include_directories(inverter_parallel-obj
        PRIVATE
        ${INVERTER_INCLUDE_DIRS})

add_library(inverter_parallel-static
        STATIC
        $<TARGET_OBJECTS:inverter_parallel-obj>
        )
target_include_directories(inverter_parallel-static
        PUBLIC
        ${INVERTER_INCLUDE_DIRS}
        )
set_target_properties(inverter_parallel-static
        PROPERTIES
        OUTPUT_NAME inverter_parallel
        )
target_link_libraries(inverter_parallel-static
        PUBLIC
        gdal)
