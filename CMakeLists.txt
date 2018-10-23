cmake_minimum_required(VERSION 3.10)
project(testDepth)

set(CMAKE_CXX_STANDARD 11)

find_package(PkgConfig REQUIRED)
find_package(Threads)

FILE(GLOB SOURCE_FILES main.cpp)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

pkg_check_modules(opencv REQUIRED opencv>=3.3.0)
target_include_directories(${PROJECT_NAME} PUBLIC ${opencv_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${opencv_LDFLAGS})
