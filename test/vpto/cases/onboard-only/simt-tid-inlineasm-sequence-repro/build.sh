#!/usr/bin/env bash
# 构建 kernel1（SIMT 内空 inline asm 传递 tid）与 kernel2（普通 tid）的
# 独立 device program SO，以及 host 连续启动器。不依赖 TileLang/PTOAS。
# fatobj 打包流程复刻自 ptoas（strace 捕获）。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="${SCRIPT_DIR}/kernels"
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build}"
CANN_HOME="${CANN_HOME:-/home/qukelin/tools/CANN_9.1/cann-9.1.T530}"
BISHENG_BIN="${BISHENG_BIN:-${CANN_HOME}/bin/bisheng}"
BISHENG_CC1="${BISHENG_CC1:-${CANN_HOME}/tools/bisheng_compiler/bin/bisheng}"
LD_LLD="${LD_LLD:-${CANN_HOME}/bin/ld.lld}"
CLANG_RES="${CLANG_RES:-${CANN_HOME}/tools/bisheng_compiler/lib/clang/15.0.5}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

for t in "${BISHENG_BIN}" "${BISHENG_CC1}" "${LD_LLD}"; do
  [[ -x "${t}" ]] || die "not executable: ${t}"
done
[[ -f "${CANN_HOME}/set_env.sh" ]] || die "missing CANN environment: ${CANN_HOME}/set_env.sh"

set +u
source "${CANN_HOME}/set_env.sh" >/dev/null 2>&1
set -u

mkdir -p "${BUILD_DIR}"
BUILD_DIR="$(cd "${BUILD_DIR}" && pwd)"

build_kernel() {
  local name="$1"       # k1_inlineasm
  local kernel="$2"     # tid_asm_kernel（逻辑 kernel 名；IR 中函数为 ${kernel}_mix_aiv）
  local module_id="$3"
  local source_dir="${SOURCE_DIR}/${name}"
  local output_dir="${BUILD_DIR}/${name}"
  mkdir -p "${output_dir}"

  echo "[build] ${name}: LLVM IR -> device object"
  "${BISHENG_BIN}" \
    --cce-aicore-arch=dav-c310-vec \
    --cce-aicore-only \
    -O2 \
    --cce-generic-addrspace=off \
    -cce-bitcode-is-aicore \
    -Wno-override-module \
    -dc \
    --cce-long-scbz=true \
    -mllvm -cce-dyn-kernel-stack-size=true \
    -mllvm --cce-aicore-vec-misched=0 \
    -c -x ir "${source_dir}/${kernel}.ll" \
    -o "${output_dir}/kernel.device.o" \
    >"${output_dir}/bisheng-device.log" 2>&1

  echo "[build] ${name}: merge device object"
  "${LD_LLD}" -m aicorelinux -Ttext 0 \
    "${output_dir}/kernel.device.o" \
    -o "${output_dir}/kernel.merged.o" \
    -r --allow-multiple-definition \
    >>"${output_dir}/bisheng-device.log" 2>&1

  echo "[build] ${name}: host stub + fatobj"
  cat >"${output_dir}/stub.cpp" <<EOF
#ifndef AICORE
#define AICORE [aicore]
#endif

extern "C" __global__ AICORE void ${kernel}(__gm__ void * arg0) {}
EOF
  "${BISHENG_CC1}" \
    -cc1 \
    -triple aarch64-unknown-linux-gnu \
    -target-cpu generic \
    -fcce-aicpu-legacy-launch \
    -fcce-is-host \
    -cce-enable-mix \
    -mllvm -enable-mix=true \
    -cce-launch-with-flagv2-impl \
    -fcce-aicore-arch dav-c310 \
    -fcce-fatobj-compile \
    -emit-obj \
    --mrelax-relocations \
    -disable-free \
    -clear-ast-before-backend \
    -disable-llvm-verifier \
    -discard-value-names \
    -main-file-name stub.cpp \
    -mrelocation-model pic \
    -pic-level 2 \
    -fhalf-no-semantic-interposition \
    -mframe-pointer=none \
    -fmath-errno \
    -ffp-contract=on \
    -fno-rounding-math \
    -mconstructor-aliases \
    -funwind-tables=2 \
    -fallow-half-arguments-and-returns \
    -mllvm -treat-scalable-fixed-error-as-warning \
    -fcoverage-compilation-dir=. \
    -resource-dir "${CLANG_RES}" \
    -internal-isystem "${CLANG_RES}/include" \
    -include __clang_cce_runtime_wrapper.h \
    -D _FORTIFY_SOURCE=2 \
    -D REGISTER_BASE \
    -O2 \
    -Wno-macro-redefined \
    -Wno-ignored-attributes \
    -std=c++17 \
    -fdeprecated-macro \
    -fdebug-compilation-dir=. \
    -ferror-limit 19 \
    -stack-protector 2 \
    -fno-signed-char \
    -fgnuc-version=4.2.1 \
    -fcxx-exceptions \
    -fexceptions \
    -vectorize-loops \
    -vectorize-slp \
    -mllvm -cce-aicore-stack-size=0x8000 \
    -mllvm -cce-aicore-function-stack-size=0x8000 \
    -mllvm -cce-aicore-record-overflow=true \
    -mllvm -cce-aicore-addr-transform \
    -mllvm -cce-aicore-dcci-insert-for-scalar=false \
    -fcce-include-aibinary "${output_dir}/kernel.merged.o" \
    -fcce-device-module-id "${module_id}" \
    -faddrsig \
    -D__GCC_HAVE_DWARF2_CFI_ASM=1 \
    -o "${output_dir}/kernel.fatobj.o" \
    -x cce "${output_dir}/stub.cpp" \
    >"${output_dir}/bisheng-fatobj.log" 2>&1

  echo "[build] ${name}: launch wrapper"
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

  echo "[build] ${name}: shared library"
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

build_kernel k1_inlineasm tid_asm_kernel ptoas_module_0
build_kernel k2_plain tid_plain_kernel ptoas_module_1

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

echo "Built into ${BUILD_DIR}"
echo "Check VF_SIMT code-size field:"
echo "  ${CANN_HOME}/bin/llvm-objdump -d ${BUILD_DIR}/k1_inlineasm/kernel.device.o | grep -i vf_simt"
echo "  ${CANN_HOME}/bin/llvm-objdump -d ${BUILD_DIR}/k2_plain/kernel.device.o | grep -i vf_simt"
