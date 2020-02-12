
list(APPEND APP_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/main.cpp)

list(APPEND APP_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include)

add_executable(aluserer ${APP_SOURCES} ${CUDA_ALG_SOURCES})
target_include_directories(aluserer PUBLIC ${APP_INCLUDE_DIRS} ${INVERTER_INCLUDE_DIRS} ${CUDA_ALG_INCLUDE_DIRS})
target_link_libraries(aluserer PUBLIC inverter-static cuda_alg-static)