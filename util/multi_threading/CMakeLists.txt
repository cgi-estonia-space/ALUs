list(APPEND MULTI_THREADING_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/alus_thread_pool.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/thread_worker.cc
        )

list(APPEND MULTI_THREADING_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include
        )

add_library(multi_threading-obj
        OBJECT
        ${MULTI_THREADING_SOURCES}
        )
target_include_directories(multi_threading-obj
        PRIVATE
        ${MULTI_THREADING_INCLUDE_DIRS}
)

add_library(multi_threading-static
        STATIC
        $<TARGET_OBJECTS:multi_threading-obj>
        )
target_include_directories(multi_threading-static
        PUBLIC
        ${MULTI_THREADING_INCLUDE_DIRS}
        )
set_target_properties(multi_threading-static
        PROPERTIES
        OUTPUT_NAME multi_threading
        )
target_link_libraries(multi_threading-static
        PUBLIC
        Threads::Threads
        )