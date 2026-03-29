cmake_minimum_required(VERSION 3.10)
project(mapeg2d)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# 源文件
set(SOURCES
    src/entity.cpp
    src/aircraft.cpp
    src/interceptor.cpp
    src/decoy.cpp
    src/missile.cpp
    src/battlefield.cpp
    src/simulation.cpp
    src/main.cpp
)

# 创建可执行文件
add_executable(mapeg2d ${SOURCES})

# Windows特定设置
if(WIN32)
    target_compile_definitions(mapeg2d PRIVATE WIN32_LEAN_AND_MEAN)
    set_property(TARGET mapeg2d PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# 设置输出目录
set_target_properties(mapeg2d PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)


# 添加调试信息
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "Include directories: ${CMAKE_CURRENT_SOURCE_DIR}/include")
message(STATUS "Source files: ${SOURCES}")