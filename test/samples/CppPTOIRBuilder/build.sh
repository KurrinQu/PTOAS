#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PTOAS_SOURCE_DIR="${PTOAS_SOURCE_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
PTOAS_BUILD_DIR="${PTOAS_BUILD_DIR:-${PTOAS_SOURCE_DIR}/build}"
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build}"

CACHE_FILE="${PTOAS_BUILD_DIR}/CMakeCache.txt"
if [[ ! -f "${CACHE_FILE}" ]]; then
  echo "error: PTOAS build cache not found: ${CACHE_FILE}" >&2
  echo "hint: set PTOAS_BUILD_DIR=/path/to/PTOAS/build" >&2
  exit 1
fi

extract_cache_var() {
  local name="$1"
  sed -n "s#^${name}:[^=]*=##p" "${CACHE_FILE}" | head -n 1
}

LLVM_DIR="${LLVM_DIR:-$(extract_cache_var LLVM_DIR)}"
MLIR_DIR="${MLIR_DIR:-$(extract_cache_var MLIR_DIR)}"
CXX_COMPILER="${CXX:-$(extract_cache_var CMAKE_CXX_COMPILER)}"

if [[ -z "${LLVM_DIR}" || -z "${MLIR_DIR}" ]]; then
  echo "error: failed to extract LLVM_DIR or MLIR_DIR from ${CACHE_FILE}" >&2
  exit 1
fi

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-RelWithDebInfo}" \
  -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" \
  -DPTOAS_SOURCE_DIR="${PTOAS_SOURCE_DIR}" \
  -DPTOAS_BUILD_DIR="${PTOAS_BUILD_DIR}" \
  -DLLVM_DIR="${LLVM_DIR}" \
  -DMLIR_DIR="${MLIR_DIR}"

cmake --build "${BUILD_DIR}" -j "${JOBS:-$(nproc)}"

if [[ "${RUN_AFTER_BUILD:-1}" == "1" ]]; then
  "${BUILD_DIR}/vadd_pto_ir_builder"
fi
