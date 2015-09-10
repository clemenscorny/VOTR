project(NCC)

cmake_minimum_required(VERSION 2.6)

find_package(OpenCV REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_definitions ("-Wall")

add_executable(ncc ncc.cpp)

target_link_libraries(ncc ${OpenCV_LIBS})

