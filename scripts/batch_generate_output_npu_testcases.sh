#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

SRC_ROOT="${SRC_ROOT:-${PTOAS_OUT_DIR:-${REPO_ROOT}/build/output}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/build/output_npu_validation}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/build/output_npu_validation_log}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_MODE="${RUN_MODE:-npu}"
SOC_VERSION="${SOC_VERSION:-Ascend950}"
CASE_FILTER="${CASE_FILTER:-}"
SAMPLE_FILTER="${SAMPLE_FILTER:-}"

usage() {
	cat <<EOF
批量对 build/output 下的所有 .cpp 执行 generate_testcase.py。

用法:
  $0 [--src-root <dir>] [--output-root <dir>] [--log-root <dir>]
     [--python <python>] [--run-mode <npu|sim>] [--soc-version <soc>]
     [--sample <sample>] [--case <case>]

参数:
  --src-root     输入 .cpp 根目录，默认: \$PTOAS_OUT_DIR 或 ${REPO_ROOT}/build/output
  --output-root  generate_testcase.py 输出目录，默认: ${REPO_ROOT}/build/output_npu_validation
  --log-root     每个 case 的日志目录，默认: ${REPO_ROOT}/build/output_npu_validation_log
  --python       Python 解释器，默认: python3
  --run-mode     传给 generate_testcase.py 的运行模式，默认: npu
  --soc-version  传给 generate_testcase.py 的 SoC，默认: Ascend950
  --sample       仅处理指定 sample 目录，例如 Abs
  --case         仅处理指定 case 名，例如 abs 或 abs-pto
  -h, --help     显示帮助

示例:
  $0
  $0 --sample TMuls
  $0 --case tmuls_half_63x128_63x64_63x64
EOF
}

log() {
	printf '[%s] %s\n' "$(date +'%F %T')" "$*"
}

normalize_case_name() {
	local name="$1"
	name="${name%.cpp}"
	name="${name%-pto}"
	name="${name%_pto}"
	printf '%s\n' "${name}"
}

matches_filter() {
	local value="$1"
	local filter="$2"
	[[ -z "${filter}" ]] && return 0
	[[ "${value}" == "${filter}" ]]
}

while [[ $# -gt 0 ]]; do
	case "$1" in
	--src-root)
		[[ $# -ge 2 ]] || {
			echo "[ERROR] --src-root 缺少参数" >&2
			exit 2
		}
		SRC_ROOT="$2"
		shift 2
		;;
	--output-root)
		[[ $# -ge 2 ]] || {
			echo "[ERROR] --output-root 缺少参数" >&2
			exit 2
		}
		OUTPUT_ROOT="$2"
		shift 2
		;;
	--log-root)
		[[ $# -ge 2 ]] || {
			echo "[ERROR] --log-root 缺少参数" >&2
			exit 2
		}
		LOG_ROOT="$2"
		shift 2
		;;
	--python)
		[[ $# -ge 2 ]] || {
			echo "[ERROR] --python 缺少参数" >&2
			exit 2
		}
		PYTHON_BIN="$2"
		shift 2
		;;
	--run-mode)
		[[ $# -ge 2 ]] || {
			echo "[ERROR] --run-mode 缺少参数" >&2
			exit 2
		}
		RUN_MODE="$2"
		shift 2
		;;
	--soc-version)
		[[ $# -ge 2 ]] || {
			echo "[ERROR] --soc-version 缺少参数" >&2
			exit 2
		}
		SOC_VERSION="$2"
		shift 2
		;;
	--sample)
		[[ $# -ge 2 ]] || {
			echo "[ERROR] --sample 缺少参数" >&2
			exit 2
		}
		SAMPLE_FILTER="$2"
		shift 2
		;;
	--case)
		[[ $# -ge 2 ]] || {
			echo "[ERROR] --case 缺少参数" >&2
			exit 2
		}
		CASE_FILTER="$2"
		shift 2
		;;
	-h | --help)
		usage
		exit 0
		;;
	*)
		echo "[ERROR] 未知参数: $1" >&2
		usage >&2
		exit 2
		;;
	esac
done

[[ -d "${SRC_ROOT}" ]] || {
	echo "[ERROR] 输入目录不存在: ${SRC_ROOT}" >&2
	exit 1
}
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
	echo "[ERROR] Python 不可用: ${PYTHON_BIN}" >&2
	exit 1
}

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}" || {
	echo "[ERROR] 创建输出目录失败" >&2
	exit 1
}

declare -a CPP_FILES=()
while IFS= read -r -d '' file; do
	[[ "$(basename -- "${file}")" == ._* ]] && continue
	if [[ "${file}" == *"/npu_validation/"* ]]; then
		continue
	fi
	sample_name="$(basename -- "$(dirname -- "${file}")")"
	case_base_name="$(basename -- "${file}" .cpp)"
	case_name="$(normalize_case_name "$(basename -- "${file}")")"
	matches_filter "${sample_name}" "${SAMPLE_FILTER}" || continue
	if [[ -n "${CASE_FILTER}" ]] && [[ "${case_name}" != "${CASE_FILTER}" ]] && [[ "${case_base_name}" != "${CASE_FILTER}" ]]; then
		continue
	fi
	CPP_FILES+=("${file}")
done < <(find "${SRC_ROOT}" -type f -name "*.cpp" -print0 | sort -z)

TOTAL_COUNT="${#CPP_FILES[@]}"
if [[ "${TOTAL_COUNT}" -eq 0 ]]; then
	echo "[ERROR] 未找到可处理的 .cpp case: ${SRC_ROOT}" >&2
	exit 1
fi

log "SRC_ROOT=${SRC_ROOT}"
log "OUTPUT_ROOT=${OUTPUT_ROOT}"
log "LOG_ROOT=${LOG_ROOT}"
log "RUN_MODE=${RUN_MODE}"
log "SOC_VERSION=${SOC_VERSION}"
log "TOTAL_CASES=${TOTAL_COUNT}"

status_file="$(mktemp -t output_npu_generate.XXXXXX)"
trap 'rm -f "${status_file}"' EXIT

ok_count=0
fail_count=0
idx=0

for cpp in "${CPP_FILES[@]}"; do
	idx=$((idx + 1))
	sample_name="$(basename -- "$(dirname -- "${cpp}")")"
	case_base="$(basename -- "${cpp}")"
	testcase="$(normalize_case_name "${case_base}")"
	case_log_dir="${LOG_ROOT}/${sample_name}"
	case_log_path="${case_log_dir}/${testcase}.log"

	mkdir -p "${case_log_dir}" || {
		printf '%s\t%s\t%s\n' "FAIL" "${sample_name}/${testcase}" "mkdir log dir failed" >>"${status_file}"
		fail_count=$((fail_count + 1))
		continue
	}

	echo
	log "[${idx}/${TOTAL_COUNT}] GENERATE ${sample_name}/${testcase}"
	log "INPUT=${cpp}"
	log "LOG=${case_log_path}"

	{
		echo "[INFO] generate_testcase: ${cpp}"
		"${PYTHON_BIN}" "${REPO_ROOT}/test/npu_validation/scripts/generate_testcase.py" \
			--input "${cpp}" \
			--testcase "${testcase}" \
			--output-root "${OUTPUT_ROOT}" \
			--run-mode "${RUN_MODE}" \
			--soc-version "${SOC_VERSION}"
	} 2>&1 | tee "${case_log_path}"
	gen_rc=${PIPESTATUS[0]}

	if [[ ${gen_rc} -eq 0 ]]; then
		log "OK ${sample_name}/${testcase}"
		printf '%s\t%s\t%s\n' "OK" "${sample_name}/${testcase}" "${case_log_path}" >>"${status_file}"
		ok_count=$((ok_count + 1))
	else
		log "FAIL generate_testcase: ${sample_name}/${testcase} (exit=${gen_rc})"
		printf '%s\t%s\t%s\n' "FAIL" "${sample_name}/${testcase}" "${case_log_path}" >>"${status_file}"
		fail_count=$((fail_count + 1))
	fi
done

echo
echo "========== GENERATE TESTCASE SUMMARY =========="
echo "TOTAL=${TOTAL_COUNT}  OK=${ok_count}  FAIL=${fail_count}"

if [[ ${fail_count} -gt 0 ]]; then
	echo "---------- FAIL CASES ----------"
	while IFS=$'\t' read -r st case_name log_path; do
		[[ "${st}" == "FAIL" ]] || continue
		echo "${case_name}  FAIL"
		echo "log: ${log_path}"
	done <"${status_file}"
	exit 1
fi

echo "All testcases generated."
exit 0
