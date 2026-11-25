cmake_minimum_required(VERSION 3.24)

project(mattress
    VERSION 1.0.0
    DESCRIPTION "Building the mattress shared library"
    LANGUAGES CXX)

# Defining the targets.
find_package(Python ${PYBIND11_PYTHON_VERSION} EXACT COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)

# pybind11 method:
pybind11_add_module(mattress
    src/tatami_utils.cpp
    src/tatami_multiply.cpp
    src/tatami_stats.cpp
    src/tatami_extract.cpp
    src/dense_matrix.cpp
    src/compressed_sparse_matrix.cpp
    src/fragmented_sparse_matrix.cpp
    src/delayed_binary_isometric_operation.cpp
    src/delayed_bind.cpp
    src/delayed_subset.cpp
    src/delayed_transpose.cpp
    src/delayed_unary_isometric_operation_simple.cpp
    src/delayed_unary_isometric_operation_with_args.cpp
    src/unknown_matrix.cpp
    src/init.cpp
)

target_include_directories(mattress PRIVATE "${ASSORTHEAD_INCLUDE_DIR}")
target_include_directories(mattress PRIVATE "../src/mattress/include")

set_property(TARGET mattress PROPERTY CXX_STANDARD 17)

target_link_libraries(mattress PRIVATE pybind11::pybind11)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(mattress PRIVATE -Wall -Wpedantic -Wextra)
endif()

option(FORCE_COPY "Force copy in tatami::fetch" OFF)
if(FORCE_COPY)
    target_compile_definitions(mattress PRIVATE TATAMI_DEBUG_FORCE_COPY=1)
endif()

set_target_properties(mattress PROPERTIES
    OUTPUT_NAME _lib_mattress
    PREFIX ""
)
