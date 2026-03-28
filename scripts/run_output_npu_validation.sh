#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/build/output_npu_validation}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/build/output_npu_validation_log}"
RUN_MODE="${RUN_MODE:-npu}"
SOC_VERSION="${SOC_VERSION:-Ascend950}"
CASE_FILTER="${CASE_FILTER:-}"
SAMPLE_FILTER="${SAMPLE_FILTER:-}"

usage() {
	cat <<EOF
串行执行 ${OUTPUT_ROOT} 下所有 testcase 目录中的 run.sh，并统计 pass/fail 数量与 fail 列表。

用法:
  $0 [--output-root <dir>] [--log-root <dir>]
     [--run-mode <npu|sim>] [--soc-version <soc>]
     [--sample <sample>] [--case <case>]

参数:
  --output-root  testcase 根目录，默认: ${REPO_ROOT}/build/output_npu_validation
  --log-root     每个 case 的日志目录，默认: ${REPO_ROOT}/build/output_npu_validation_log
  --run-mode     传给 run.sh 的 RUN_MODE，默认: npu
  --soc-version  传给 run.sh 的 SOC_VERSION，默认: Ascend950
  --sample       仅运行指定 sample 目录，例如 TAdd
  --case         仅运行指定 case 名，例如 tadd_float_16x32_16x64_16x32_16x31
  -h, --help     显示帮助

示例:
  $0
  $0 --sample TAdd
  $0 --case tadd_float_16x32_16x64_16x32_16x31
  $0 --run-mode sim --soc-version Ascend910
EOF
}

log() {
	printf '[%s] %s\n' "$(date +'%F %T')" "$*"
}

interrupted=0

handle_interrupt() {
	interrupted=1
	log "收到中断信号，停止继续调度后续 case"
}

matches_filter() {
	local value="$1"
	local filter="$2"
	[[ -z "${filter}" ]] && return 0
	[[ "${value}" == "${filter}" ]]
}

while [[ $# -gt 0 ]]; do
	case "$1" in
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

[[ -d "${OUTPUT_ROOT}" ]] || {
	echo "[ERROR] output_npu_validation 目录不存在: ${OUTPUT_ROOT}" >&2
	exit 1
}
mkdir -p "${LOG_ROOT}" || {
	echo "[ERROR] 创建日志目录失败: ${LOG_ROOT}" >&2
	exit 1
}

declare -a RUN_SCRIPTS=()
while IFS= read -r -d '' run_script; do
	case_dir="$(dirname -- "${run_script}")"
	case_name="$(basename -- "${case_dir}")"
	sample_name="$(basename -- "$(dirname -- "${case_dir}")")"
	matches_filter "${sample_name}" "${SAMPLE_FILTER}" || continue
	matches_filter "${case_name}" "${CASE_FILTER}" || continue
	RUN_SCRIPTS+=("${run_script}")
done < <(find "${OUTPUT_ROOT}" -type f -name "run.sh" -print0 | sort -z)

TOTAL_COUNT="${#RUN_SCRIPTS[@]}"
if [[ "${TOTAL_COUNT}" -eq 0 ]]; then
	echo "[ERROR] 未找到匹配的 run.sh: ${OUTPUT_ROOT} (sample=${SAMPLE_FILTER:-*}, case=${CASE_FILTER:-*})" >&2
	exit 1
fi

log "OUTPUT_ROOT=${OUTPUT_ROOT}"
log "LOG_ROOT=${LOG_ROOT}"
log "RUN_MODE=${RUN_MODE}"
log "SOC_VERSION=${SOC_VERSION}"
log "TOTAL_CASES=${TOTAL_COUNT}"

status_file="$(mktemp -t output_npu_validation.XXXXXX)"
trap 'rm -f "${status_file}"' EXIT
trap 'handle_interrupt' INT TERM

ok_count=0
fail_count=0
idx=0

for run_script in "${RUN_SCRIPTS[@]}"; do
	if [[ ${interrupted} -ne 0 ]]; then
		break
	fi

	idx=$((idx + 1))
	case_dir="$(dirname -- "${run_script}")"
	case_name="$(basename -- "${case_dir}")"
	sample_name="$(basename -- "$(dirname -- "${case_dir}")")"
	case_id="${sample_name}/${case_name}"
	case_log_dir="${LOG_ROOT}/${sample_name}"
	case_log_path="${case_log_dir}/${case_name}.log"

	mkdir -p "${case_log_dir}" || {
		printf '%s\t%s\t%s\n' "FAIL" "${case_id}" "mkdir log dir failed" >>"${status_file}"
		fail_count=$((fail_count + 1))
		continue
	}

	echo
	log "[${idx}/${TOTAL_COUNT}] CASE ${case_id}"
	log "RUN_SH=${run_script}"
	log "LOG=${case_log_path}"

	(
		cd "${case_dir}" || exit 1
		env RUN_MODE="${RUN_MODE}" SOC_VERSION="${SOC_VERSION}" ./run.sh
	) 2>&1 | tee "${case_log_path}"
	run_rc=${PIPESTATUS[0]}

	if [[ ${interrupted} -ne 0 || ${run_rc} -eq 130 || ${run_rc} -eq 143 ]]; then
		interrupted=1
		log "中断 ${case_id}，退出批量执行"
		break
	fi

	if [[ ${run_rc} -eq 0 ]]; then
		log "PASS ${case_id}"
		printf '%s\t%s\t%s\n' "PASS" "${case_id}" "${case_log_path}" >>"${status_file}"
		ok_count=$((ok_count + 1))
	else
		log "FAIL ${case_id} (exit=${run_rc})"
		printf '%s\t%s\t%s\n' "FAIL" "${case_id}" "${case_log_path}" >>"${status_file}"
		fail_count=$((fail_count + 1))
	fi
done

if [[ ${interrupted} -ne 0 ]]; then
	exit 130
fi

echo
echo "========== NPU VALIDATION SUMMARY =========="
echo "TOTAL=${TOTAL_COUNT}  PASS=${ok_count}  FAIL=${fail_count}"

if [[ ${fail_count} -gt 0 ]]; then
	echo "---------- FAIL CASES ----------"
	while IFS=$'\t' read -r st case_id log_path; do
		[[ "${st}" == "FAIL" ]] || continue
		echo "${case_id}"
		echo "log: ${log_path}"
	done <"${status_file}"
	exit 1
fi

echo "All cases passed."
exit 0
