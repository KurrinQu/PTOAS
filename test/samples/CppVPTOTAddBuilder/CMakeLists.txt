cmake_minimum_required(VERSION 3.20)

project(CppVPTOTAddBuilder LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(PTOAS_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../.." CACHE PATH
    "Path to the PTOAS source tree")
set(PTOAS_BUILD_DIR "${PTOAS_SOURCE_DIR}/build" CACHE PATH
    "Path to the PTOAS build tree")

if(NOT EXISTS "${PTOAS_SOURCE_DIR}/include/PTO/IR/PTO.h")
  message(FATAL_ERROR "Invalid PTOAS_SOURCE_DIR: ${PTOAS_SOURCE_DIR}")
endif()

if(NOT EXISTS "${PTOAS_BUILD_DIR}/CMakeCache.txt")
  message(FATAL_ERROR "Invalid PTOAS_BUILD_DIR: ${PTOAS_BUILD_DIR}")
endif()

if(NOT LLVM_DIR)
  message(FATAL_ERROR "LLVM_DIR is required. Use build.sh or pass -DLLVM_DIR=/path/to/llvm/cmake/dir.")
endif()

if(NOT MLIR_DIR)
  message(FATAL_ERROR "MLIR_DIR is required. Use build.sh or pass -DMLIR_DIR=/path/to/mlir/cmake/dir.")
endif()

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}" "${MLIR_CMAKE_DIR}")
include(AddLLVM)
include(AddMLIR)

add_executable(tadd_vpto_ir_builder main.cpp)

target_include_directories(tadd_vpto_ir_builder PRIVATE
  "${PTOAS_SOURCE_DIR}/include"
  "${PTOAS_BUILD_DIR}/include"
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)

target_compile_definitions(tadd_vpto_ir_builder PRIVATE ${LLVM_DEFINITIONS})

target_link_libraries(tadd_vpto_ir_builder PRIVATE
  "${PTOAS_BUILD_DIR}/lib/PTO/IR/libPTOIR.a"
  MLIRIR
  MLIRArithDialect
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRControlFlowInterfaces
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
  LLVMSupport
)
