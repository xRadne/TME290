cmake_minimum_required(VERSION 3.2)
project(helloworld)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

add_executable(${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/helloworld.cpp ${CMAKE_CURRENT_SOURCE_DIR}/prime-checker.cpp)

enable_testing()
add_executable(${PROJECT_NAME}-runner test-prime-checker.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/prime-checker.cpp)
add_test(NAME ${PROJECT_NAME}-runner COMMAND ${PROJECT_NAME}-runner)


