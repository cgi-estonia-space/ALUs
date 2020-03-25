
list(APPEND APP_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/main.cpp)

list(APPEND APP_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include)

add_executable(aluserer ${APP_SOURCES})
target_link_libraries(aluserer PUBLIC inverter-static inverter_parallel-static cuda_alg-static terrain-correction-static)
