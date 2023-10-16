cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(example-app)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(src/)

file(GLOB_RECURSE SRCS "src/*.cpp")
file(GLOB_RECURSE HDRS "src/*.h")

message("${SRCS}")

add_executable(Exp ${SRCS})
target_link_libraries(Exp "${TORCH_LIBRARIES}" pthread)
set_property(TARGET Exp PROPERTY CXX_STANDARD 17)
