#!/usr/bin/env bash
# 在同一进程内连续启动两个独立 device program：
#   k1_inlineasm（SIMT 内空 inline asm，VF_SIMT code-size 应为 0xffff）
#   k2_plain（普通 SIMT tid 读）
# 预期：k1 通过；k2 的首次 SIMT dispatch 取指异常（507035）或输出错误。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build}"
CANN_HOME="${CANN_HOME:-/home/qukelin/tools/CANN_9.1/cann-9.1.T530}"
RUNNER="${BUILD_DIR}/sequence_runner"

[[ -x "${RUNNER}" ]] || { echo "ERROR: run ./build.sh first" >&2; exit 2; }
[[ -f "${BUILD_DIR}/k1_inlineasm/libkernel.so" ]] || { echo "ERROR: run ./build.sh first" >&2; exit 2; }
[[ -f "${BUILD_DIR}/k2_plain/libkernel.so" ]] || { echo "ERROR: run ./build.sh first" >&2; exit 2; }

set +u
source "${CANN_HOME}/set_env.sh" >/dev/null 2>&1
set -u

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export ACL_DEVICE_ID="${ACL_DEVICE_ID:-0}"
export LD_LIBRARY_PATH="${BUILD_DIR}/k1_inlineasm:${BUILD_DIR}/k2_plain:${CANN_HOME}/lib64:${LD_LIBRARY_PATH:-}"

set +e
"${RUNNER}" \
  "${BUILD_DIR}/k1_inlineasm/libkernel.so" \
  "${BUILD_DIR}/k2_plain/libkernel.so"
rc=$?
echo "sequence_runner exit=${rc}（复现成功时 k2 失败、退出码非 0）"
exit 0
