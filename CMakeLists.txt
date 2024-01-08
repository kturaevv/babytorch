cmake_minimum_required(VERSION 3.21)

set(PROJECT_NAME engine)
set(TEST_NAME tests)

project(${PROJECT_NAME} VERSION 1.0)

option(
  AUTOFORMAT_SRC_ON_CONFIGURE
    "If enabled, clang-format will be used to format all sources in src/ during configuration"
        ON
)

option(
  LIST_SOURCE_FILES
    "If enabled, during project configuration, cmake will print out all files included in build"
        ON
)

add_compile_options(-Wall -Wextra -Werror)

# Set C++ standard and compile options
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_COLOR_DIAGNOSTICS ON)
set(CMAKE_MESSAGE_LOG_LEVEL STATUS)

# Include cmake module
list(APPEND CMAKE_MODULE_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/"
)

# Set source files to include
file(GLOB_RECURSE SOURCE_FILES "src/*.[ch]pp")
file(GLOB_RECURSE TEST_FILES "tests/*.[ch]pp")

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
add_executable(${TEST_NAME} ${TEST_FILES})

find_package(fmt CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(Boost REQUIRED)
find_package(Catch2 3 REQUIRED)

target_compile_options(
    ${PROJECT_NAME} 
        PUBLIC 
            -fsanitize=address 
            -fno-omit-frame-pointer 
            -g
)

target_link_libraries(
    ${PROJECT_NAME}
        PUBLIC
            -fsanitize=address
        PRIVATE 
            fmt::fmt
            fmt::fmt-header-only
            spdlog::spdlog_header_only
)

target_link_libraries(
    ${TEST_NAME} 
        PRIVATE 
            Catch2::Catch2WithMain
)

# ----------- [Optional] ------------ # 

message(STATUS 
    "Adding cmake directory for access: "
     ${CMAKE_CURRENT_SOURCE_DIR}
)

if (LIST_SOURCE_FILES MATCHES ON)
    include(print-source-files)
endif()

if (AUTOFORMAT_SRC_ON_CONFIGURE MATCHES ON)
    include(clang-format)
endif()
