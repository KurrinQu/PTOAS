#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CANN_HOME="${CANN_HOME:-/home/qukelin/tools/CANN_9.1/cann-9.1.T530}"
PTOAS_ROOT="${PTOAS_ROOT:-/home/qukelin/projects/PTOAS}"
PTOAS_BIN="${PTOAS_BIN:-${PTOAS_ROOT}/build/tools/ptoas/ptoas}"
PTOAS_ENV="${PTOAS_ENV:-${PTOAS_ROOT}/scripts/ptoas_env.sh}"
PTO_ARCH="${PTO_ARCH:-a5}"
PTO_AICORE_ARCH="${PTO_AICORE_ARCH:-dav-c310}"
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
ACL_DEVICE_ID="${ACL_DEVICE_ID:-0}"

ACTION="${1:-build}"
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build}"
KERNEL_PTO="${KERNEL_PTO:-${SCRIPT_DIR}/kernel.pto}"
LAUNCH_CPP="${LAUNCH_CPP:-${SCRIPT_DIR}/launch.cpp}"
MAIN_CPP="${MAIN_CPP:-${SCRIPT_DIR}/main.cpp}"
FATOBJ="${BUILD_DIR}/kernel.fatobj.o"
LAUNCH_OBJ="${BUILD_DIR}/launch.o"
KERNEL_SO="${BUILD_DIR}/lib_kernel.so"
HOST_EXE="${BUILD_DIR}/rmsnorm_pto_main"

if [[ ! -f "${KERNEL_PTO}" ]]; then
  echo "kernel PTO IR not found: ${KERNEL_PTO}" >&2
  exit 1
fi
if [[ ! -f "${LAUNCH_CPP}" ]]; then
  echo "launch source not found: ${LAUNCH_CPP}" >&2
  exit 1
fi
if [[ ! -f "${MAIN_CPP}" ]]; then
  echo "host source not found: ${MAIN_CPP}" >&2
  exit 1
fi
if [[ ! -x "${PTOAS_BIN}" ]]; then
  echo "ptoas not found or not executable: ${PTOAS_BIN}" >&2
  exit 1
fi
if [[ ! -f "${CANN_HOME}/set_env.sh" ]]; then
  echo "CANN set_env.sh not found: ${CANN_HOME}/set_env.sh" >&2
  exit 1
fi

export PTOAS_ENV_SKIP_SMOKE_TEST="${PTOAS_ENV_SKIP_SMOKE_TEST:-1}"
export ASCEND_HOME_PATH="${CANN_HOME}"
export ASCEND_RT_VISIBLE_DEVICES
export ACL_DEVICE_ID

source "${PTOAS_ENV}"
source "${CANN_HOME}/set_env.sh"

mkdir -p "${BUILD_DIR}"

BISHENG="${CANN_HOME}/aarch64-linux/bin/bisheng"
CXX_BIN="${CXX:-g++}"

build_all() {
  echo "[1/4] PTOAS compile: ${KERNEL_PTO}"
  "${PTOAS_BIN}" --pto-arch="${PTO_ARCH}" --pto-backend=vpto \
    "${KERNEL_PTO}" \
    -o "${FATOBJ}"

  echo "[2/4] Compile launch wrapper: ${LAUNCH_CPP}"
  "${BISHENG}" -c -fPIC -O2 -xcce \
    -Xhost-start -Xhost-end \
    -mllvm -cce-aicore-stack-size=0x8000 \
    -mllvm -cce-aicore-function-stack-size=0x8000 \
    -mllvm -cce-aicore-record-overflow=true \
    -mllvm -cce-aicore-addr-transform \
    -mllvm -cce-aicore-dcci-insert-for-scalar=false \
    --cce-aicore-arch="${PTO_AICORE_ARCH}" \
    -std=c++17 \
    -Wno-macro-redefined \
    -Wno-ignored-attributes \
    "${LAUNCH_CPP}" \
    -o "${LAUNCH_OBJ}"

  echo "[3/4] Link device shared library: ${KERNEL_SO}"
  "${BISHENG}" -fPIC -shared --cce-fatobj-link \
    -o "${KERNEL_SO}" \
    "${FATOBJ}" \
    "${LAUNCH_OBJ}" \
    -Wl,--no-as-needed

  echo "[4/4] Build host executable: ${HOST_EXE}"
  "${CXX_BIN}" -std=c++17 -O2 \
    "${MAIN_CPP}" \
    -I"${CANN_HOME}/include" \
    -L"${CANN_HOME}/lib64" \
    -Wl,-rpath,"${BUILD_DIR}" \
    -Wl,-rpath,"${CANN_HOME}/lib64" \
    -Wl,-rpath,"${CANN_HOME}/aarch64-linux/lib64" \
    -Wl,--allow-shlib-undefined \
    -Wl,--no-as-needed \
    -lascendcl \
    -Wl,--as-needed \
    -ldl \
    -lpthread \
    -o "${HOST_EXE}"

  ln -sf "${KERNEL_SO}" "${SCRIPT_DIR}/lib_kernel.rebuilt.so"
  ln -sf "${HOST_EXE}" "${SCRIPT_DIR}/rmsnorm_pto_main"
  echo "build done:"
  echo "  ${KERNEL_SO}"
  echo "  ${HOST_EXE}"
}

run_case() {
  export LD_LIBRARY_PATH="${BUILD_DIR}:${CANN_HOME}/lib64:${CANN_HOME}/aarch64-linux/lib64:${LD_LIBRARY_PATH:-}"
  "${HOST_EXE}" "${KERNEL_SO}"
}

case "${ACTION}" in
  build)
    build_all
    ;;
  run)
    build_all
    run_case
    ;;
  run-only)
    run_case
    ;;
  *)
    echo "usage: $0 [build|run|run-only]" >&2
    exit 1
    ;;
esac
