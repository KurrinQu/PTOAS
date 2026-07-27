#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PTOAS_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
SOURCE_DIR="${SCRIPT_DIR}/kernels"
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build}"
CANN_HOME="${CANN_HOME:-/home/qukelin/tools/CANN_9.1/cann-9.1.T530}"
PTOAS_BIN="${PTOAS_BIN:-${PTOAS_ROOT}/build/tools/ptoas/ptoas}"
BISHENG_BIN="${BISHENG_BIN:-${CANN_HOME}/bin/bisheng}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -x "${PTOAS_BIN}" ]] || die "PTOAS_BIN is not executable: ${PTOAS_BIN}"
[[ -x "${BISHENG_BIN}" ]] || die "BISHENG_BIN is not executable: ${BISHENG_BIN}"
[[ -f "${CANN_HOME}/set_env.sh" ]] || die "missing CANN environment: ${CANN_HOME}/set_env.sh"

set +u
source "${CANN_HOME}/set_env.sh" >/dev/null 2>&1
set -u

mkdir -p "${BUILD_DIR}"
BUILD_DIR="$(cd "${BUILD_DIR}" && pwd)"

build_shape() {
  local d="$1"
  local source_dir="${SOURCE_DIR}/d${d}"
  local output_dir="${BUILD_DIR}/d${d}"

  mkdir -p "${output_dir}"
  echo "[build] d=${d}: PTOAS"
  "${PTOAS_BIN}" \
    --pto-arch=a5 \
    --pto-backend=vpto \
    "${source_dir}/kernel.pto" \
    -o "${output_dir}/kernel.fatobj.o" \
    >"${output_dir}/ptoas.log" 2>&1

  echo "[build] d=${d}: launch wrapper"
  "${BISHENG_BIN}" \
    -c -fPIC -xcce -fenable-matrix --cce-aicore-enable-tl \
    -Xhost-start -Xhost-end \
    -mllvm -cce-aicore-stack-size=0x8000 \
    -mllvm -cce-aicore-function-stack-size=0x8000 \
    -mllvm -cce-aicore-record-overflow=true \
    -mllvm -cce-aicore-addr-transform \
    -mllvm -cce-aicore-dcci-insert-for-scalar=false \
    --cce-aicore-arch=dav-c310 \
    -DREGISTER_BASE -std=c++17 \
    -Wno-macro-redefined -Wno-ignored-attributes \
    -I "${CANN_HOME}/include" \
    -I "${CANN_HOME}/pkg_inc" \
    -I "${CANN_HOME}/pkg_inc/profiling" \
    -I "${CANN_HOME}/pkg_inc/runtime/runtime" \
    "${source_dir}/launch.cpp" \
    -o "${output_dir}/launch.o" \
    >"${output_dir}/bisheng-launch.log" 2>&1

  echo "[build] d=${d}: shared library"
  "${BISHENG_BIN}" \
    -fPIC -s -Wl,-z,relro -Wl,-z,now --cce-fatobj-link \
    -shared -Wl,-soname,libkernel.so \
    -L "${CANN_HOME}/lib64" \
    -Wl,-rpath,"${CANN_HOME}/lib64" \
    -o "${output_dir}/libkernel.so" \
    "${output_dir}/kernel.fatobj.o" \
    "${output_dir}/launch.o" \
    -Wl,--no-as-needed -lruntime \
    >"${output_dir}/bisheng-link.log" 2>&1
}

for d in 4096 5120 7168; do
  build_shape "${d}"
done

echo "[build] ACL sequence runner"
"${BISHENG_BIN}" \
  -xc++ -include stdint.h -include stddef.h -std=c++17 \
  "${SCRIPT_DIR}/host/sequence_main.cpp" \
  -I "${CANN_HOME}/include" \
  -L "${CANN_HOME}/lib64" \
  -Wl,-rpath,"${CANN_HOME}/lib64" \
  -Wl,--allow-shlib-undefined \
  -lruntime -lstdc++ -lascendcl -lm -ldl -lpthread \
  -o "${BUILD_DIR}/sequence_runner" \
  >"${BUILD_DIR}/bisheng-host.log" 2>&1

echo "Built source-only reproducer into ${BUILD_DIR}"
