cmake_minimum_required(VERSION 3.21)

set(PROJECT_NAME engine)
set(TEST_NAME tests)

# Set C++ standard and compile options
set(CMAKE_CXX_COMPILER "clang-18")
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_COLOR_DIAGNOSTICS ON)
set(CMAKE_MESSAGE_LOG_LEVEL STATUS)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")

add_compile_options(
  -Wall
  -Wextra
  -Wpedantic
  -Wno-unused-variable
  -Wno-infinite-recursion
  -Werror=implicit-function-declaration
  -Wshadow
  -Wno-shadow-field-in-constructor-modified
  -Wno-shadow-ivar
  -Wuninitialized
  -Wunused-label
  -Wunused-lambda-capture
)

project(${PROJECT_NAME} VERSION 1.0)

option(AUTOFORMAT_SRC_ON_CONFIGURE
  "If enabled, clang-format will be used to format all sources in src/ during configuration"
  ON)
option(LIST_SOURCE_FILES
  "If enabled, during project configuration, cmake will print out all files included in build"
  ON)

# Include cmake module
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

# Set source files to include (consider explicitly listing files instead)
file(GLOB_RECURSE SOURCE_FILES "src/*.[ch]pp")
file(GLOB_RECURSE TEST_FILES "tests/*.[ch]pp")

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
add_executable(${TEST_NAME} ${TEST_FILES})

find_package(fmt CONFIG REQUIRED)
find_package(Catch2 3 REQUIRED)

# Add sanitizer flags only for Debug builds
target_compile_options(${PROJECT_NAME}
  PUBLIC
    $<$<CONFIG:Debug>:-fsanitize=address -fno-omit-frame-pointer>
)

target_link_options(${PROJECT_NAME}
  PUBLIC
    $<$<CONFIG:Debug>:-fsanitize=address>
)

target_link_libraries(${PROJECT_NAME}
  PRIVATE
    fmt::fmt
    fmt::fmt-header-only
    -lstdc++
    m
)

target_link_libraries(${TEST_NAME}
  PRIVATE
    Catch2::Catch2WithMain
    fmt::fmt
    fmt::fmt-header-only
    -lstdc++
    m
)

# ----------- [Optional] ------------ #

message(STATUS
    "Adding cmake directory for access: "
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(LIST_SOURCE_FILES MATCHES ON)
    include(print-source-files)
endif()

if(AUTOFORMAT_SRC_ON_CONFIGURE MATCHES ON)
    include(clang-format)
endif()
