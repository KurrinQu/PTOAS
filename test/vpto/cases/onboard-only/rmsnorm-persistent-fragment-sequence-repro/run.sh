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
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build}"
CANN_HOME="${CANN_HOME:-/home/qukelin/tools/CANN_9.1/cann-9.1.T530}"
RUNNER="${BUILD_DIR}/sequence_runner"

[[ -x "${RUNNER}" ]] || {
  echo "ERROR: missing ${RUNNER}; run ./build.sh first" >&2
  exit 2
}
for d in 4096 5120 7168; do
  [[ -f "${BUILD_DIR}/d${d}/libkernel.so" ]] || {
    echo "ERROR: missing ${BUILD_DIR}/d${d}/libkernel.so; run ./build.sh first" >&2
    exit 2
  }
done

set +u
source "${CANN_HOME}/set_env.sh" >/dev/null 2>&1
set -u

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export ACL_DEVICE_ID="${ACL_DEVICE_ID:-0}"
export LD_LIBRARY_PATH="${BUILD_DIR}/d4096:${BUILD_DIR}/d5120:${BUILD_DIR}/d7168:${CANN_HOME}/lib64:${LD_LIBRARY_PATH:-}"

exec "${RUNNER}" \
  "${BUILD_DIR}/d4096/libkernel.so" \
  "${BUILD_DIR}/d5120/libkernel.so" \
  "${BUILD_DIR}/d7168/libkernel.so"
