cmake_minimum_required(VERSION 3.24)
project(ppopp-tutorial-2026 VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

################################################################################
# MLIR Dependency
#
# We take a MLIR dependency through releases via EuDSL. Building MLIR is a
# whole thing and requires a big chunky machine which might not be readily
# available. We anyway don't want to make changes to MLIR itself and only want
# to use it as a library.
#
# It is easy to add mlir as a submodule and point the cmake directory to it,
# if you want.
# TODO: Add example of adding mlir as a submodule.
#
################################################################################

find_package(MLIR CONFIG)

if(NOT MLIR_FOUND)
  message(FATAL_ERROR
    "MLIR not found. Please specify CMAKE_PREFIX_PATH to the MLIR installation.\n"
    "If using mlir_wheel, run:\n"
    "  cmake -DCMAKE_PREFIX_PATH=$(python -m mlir_wheel --root-dir) -DLLVM_EXTERNAL_LIT=$(which lit) .."
  )
endif()

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

################################################################################
# tutorial-opt
################################################################################

add_subdirectory(tutorial)
add_subdirectory(test)
