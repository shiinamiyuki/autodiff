cmake_minimum_required(VERSION 3.12)

project(autodiff)
set(CMAKE_CXX_STANDARD 17)


if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    add_executable(simple-test-cg tests/simple_test_cg.cpp)
    target_include_directories(simple-test-cg PUBLIC include/)
    add_custom_command(COMMAND simple-test-cg
                        OUTPUT tests/grad.cpp
                        WORKING_DIRECTORY tests/
                        DEPENDS simple-test-cg )
    add_executable(simple-test tests/simple_test.cpp tests/grad.cpp)
    target_include_directories(simple-test PUBLIC include/)
endif()

add_library(autodiff INTERFACE)
target_include_directories(autodiff INTERFACE include/)