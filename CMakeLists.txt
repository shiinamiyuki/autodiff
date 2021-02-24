cmake_minimum_required(VERSION 3.12)

project(autodiff)
set(CMAKE_CXX_STANDARD 17)


add_executable(simple-test tests/simple_test.cpp)
target_include_directories(simple-test PUBLIC include/)