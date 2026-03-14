#!/usr/bin/env bash
set -euo pipefail

SOC_VERSION="${SOC_VERSION:-Ascend910}"
BUILD_DIR="${BUILD_DIR:-build}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${ROOT_DIR}"
python3 "${ROOT_DIR}/golden.py"

if [[ -z "${ASCEND_HOME_PATH:-}" && -f "/usr/local/Ascend/cann/set_env.sh" ]]; then
  set +e; set +u; set +o pipefail
  source "/usr/local/Ascend/cann/set_env.sh" || true
  set -o pipefail; set -u; set -e
elif [[ -z "${ASCEND_HOME_PATH:-}" && -f "/usr/local/Ascend/ascend-toolkit/latest/set_env.sh" ]]; then
  set +e; set +u; set +o pipefail
  source "/usr/local/Ascend/ascend-toolkit/latest/set_env.sh" || true
  set -o pipefail; set -u; set -e
fi

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  echo "[ERROR] ASCEND_HOME_PATH is not set; please source CANN env first." >&2
  exit 2
fi

if [[ -z "${PTO_ISA_ROOT:-}" ]]; then
  search_dir="${ROOT_DIR}"
  for _ in {1..8}; do
    if [[ -d "${search_dir}/pto-isa/include" && -d "${search_dir}/pto-isa/tests/common" ]]; then
      PTO_ISA_ROOT="${search_dir}/pto-isa"
      break
    fi
    [[ "${search_dir}" == "/" ]] && break
    search_dir="$(dirname "${search_dir}")"
  done
  export PTO_ISA_ROOT="${PTO_ISA_ROOT:-}"
fi

if [[ -z "${PTO_ISA_ROOT:-}" ]]; then
  echo "[ERROR] PTO_ISA_ROOT is not set and auto-detect failed." >&2
  exit 2
fi

mkdir -p "${ROOT_DIR}/${BUILD_DIR}"
cmake -S "${ROOT_DIR}" -B "${ROOT_DIR}/${BUILD_DIR}" \
  -DSOC_VERSION="${SOC_VERSION}" \
  -DENABLE_SIM_GOLDEN=OFF \
  -DPTO_ISA_ROOT="${PTO_ISA_ROOT}"
cmake --build "${ROOT_DIR}/${BUILD_DIR}" --parallel

export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"
"${ROOT_DIR}/${BUILD_DIR}/assemble"
python3 "${ROOT_DIR}/compare.py"
