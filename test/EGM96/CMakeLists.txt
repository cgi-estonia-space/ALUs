list(APPEND EGM96_TEST_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/EGM96_test.cu
        )

list(APPEND EGM96_TEST_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/../../util/sentinel1/include
        )

add_library(egm96-test-obj
        OBJECT
        ${EGM96_TEST_SOURCES}
        )
target_include_directories(egm96-test-obj
        PRIVATE
        ${EGM96_TEST_INCLUDE_DIRS}
        )

add_library(egm96-test-static
        STATIC
        $<TARGET_OBJECTS:egm96-test-obj>
        )
target_include_directories(egm96-test-static
        PUBLIC
        ${EGM96_TEST_INCLUDE_DIRS}
        )
set_target_properties(egm96-test-static
        PROPERTIES
        OUTPUT_NAME egm96-test
        )
target_link_libraries(egm96-test-static
        PUBLIC
        sentinel1-util-static
        )
