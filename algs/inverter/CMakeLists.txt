
list(APPEND INVERTER_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/inverter.cpp)

list(APPEND INVERTER_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include)

add_library(inverter-obj
        OBJECT
        ${INVERTER_SOURCES})

target_include_directories(inverter-obj
        PRIVATE
        ${INVERTER_INCLUDE_DIRS})

add_library(inverter-static
        STATIC
        $<TARGET_OBJECTS:inverter-obj>
        )
target_include_directories(inverter-static
        PUBLIC
        ${INVERTER_INCLUDE_DIRS}
        )
set_target_properties(inverter-static
        PROPERTIES
        OUTPUT_NAME inverter
        )
target_link_libraries(inverter-static
        PUBLIC
        gdal)
