list(APPEND ALGORITHM_BOND_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/alg_load.cc
        )

list(APPEND ALGORITHM_BOND_INCLUDES
        ${CMAKE_CURRENT_LIST_DIR}/include/
        ${CMAKE_CURRENT_LIST_DIR}/../../app/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/exotic-operations/include
        )

add_library(alg-bond-static
        STATIC
        ${ALGORITHM_BOND_SOURCES}
        )

target_include_directories(alg-bond-static
        PUBLIC
        ${ALGORITHM_BOND_INCLUDES}
        )

target_link_libraries(alg-bond-static
        PUBLIC
        common-util-static
        )

set_target_properties(alg-bond-static
        PROPERTIES
        OUTPUT_NAME alg-bond
        )

