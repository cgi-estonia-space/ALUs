
add_library(pugixml-static
        STATIC
        ${CMAKE_CURRENT_LIST_DIR}/pugixml.cpp
        )
target_include_directories(pugixml-static
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/include
        )
set_target_properties(pugixml-static
        PROPERTIES
        OUTPUT_NAME pugixml
        )