#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

MODE="all"
JOBS="${JOBS:-$(nproc)}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || true)}"
SKIP_PY_DEPS=0
DRY_RUN=0

WORKSPACE_DIR="${WORKSPACE_DIR:-$HOME/llvm-workspace}"
LLVM_SOURCE_DIR="${LLVM_SOURCE_DIR:-${WORKSPACE_DIR}/llvm-project}"
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-${LLVM_SOURCE_DIR}/build-shared}"
PTO_SOURCE_DIR="${PTO_SOURCE_DIR:-${REPO_ROOT}}"
PTO_INSTALL_DIR="${PTO_INSTALL_DIR:-${PTO_SOURCE_DIR}/install}"

SET_WORKSPACE_DIR=0
SET_LLVM_SOURCE_DIR=0
SET_LLVM_BUILD_DIR=0
SET_PTO_SOURCE_DIR=0
SET_PTO_INSTALL_DIR=0

LLVM_REPO_URL="${LLVM_REPO_URL:-https://github.com/llvm/llvm-project.git}"
PTO_REPO_URL="${PTO_REPO_URL:-https://github.com/zhangstevenunity/PTOAS.git}"
LLVM_BRANCH="${LLVM_BRANCH:-release/19.x}"

LLVM_ENABLE_PROJECTS="${LLVM_ENABLE_PROJECTS:-mlir;clang}"
LLVM_TARGETS_TO_BUILD="${LLVM_TARGETS_TO_BUILD:-host}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"

usage() {
	cat <<EOF
按照 README 的构建流程引导搭建 PTOAS 环境。

用法:
  scripts/bootstrap_ptoas.sh [选项]

核心选项:
  --mode <all|prepare|llvm-only|ptoas-only|llvm-configure|llvm-build|ptoas-configure|ptoas-build|ptoas-install>
      all            执行 prepare + llvm-only + ptoas-only (默认)
      prepare        创建目录，按需 clone llvm/PTOAS
      llvm-only      仅执行 LLVM configure + build
      ptoas-only     仅执行 PTOAS configure + build + install
      llvm-configure 仅配置 LLVM
      llvm-build     仅编译 LLVM
      ptoas-configure仅配置 PTOAS
      ptoas-build    仅编译 PTOAS
      ptoas-install  仅安装 PTOAS

  --workspace-dir <path>   工作目录 (默认: \$HOME/llvm-workspace)
  --llvm-source-dir <path> LLVM 源码目录 (默认: <workspace>/llvm-project)
  --llvm-build-dir <path>  LLVM 构建目录 (默认: <llvm-source>/build-shared)
  --pto-source-dir <path>  PTOAS 源码目录 (默认: 当前仓库)
  --pto-install-dir <path> PTOAS 安装目录 (默认: <pto-source>/install)
  --python-bin <path>      Python 可执行文件 (默认: python3)
  --jobs <N>               并行编译任务数 (默认: nproc)
  --skip-python-deps       跳过 pip 安装 pybind11/numpy
  --dry-run                仅打印将执行的命令，不实际执行
  --help, -h               显示帮助

说明:
  1) 你可以先运行 --mode llvm-only，之后再运行 --mode ptoas-only。
  2) 默认不会覆盖已有源码目录；目录不存在时才会 clone。

示例:
  scripts/bootstrap_ptoas.sh --mode llvm-only
  scripts/bootstrap_ptoas.sh --mode ptoas-only --jobs 32
  scripts/bootstrap_ptoas.sh --mode all --workspace-dir /data/llvm-workspace
EOF
}

log() {
	echo "[bootstrap] $*"
}

die() {
	echo "[bootstrap][ERROR] $*" >&2
	exit 1
}

run_cmd() {
	log "RUN: $*"
	if [[ ${DRY_RUN} -eq 1 ]]; then
		return 0
	fi
	"$@"
}

need_cmd() {
	if [[ ${DRY_RUN} -eq 1 ]]; then
		return 0
	fi
	command -v "$1" >/dev/null 2>&1 || die "未找到命令: $1"
}

parse_args() {
	while [[ $# -gt 0 ]]; do
		case "$1" in
		--mode)
			[[ $# -ge 2 ]] || die "--mode 缺少参数"
			MODE="$2"
			shift 2
			;;
		--workspace-dir)
			[[ $# -ge 2 ]] || die "--workspace-dir 缺少参数"
			WORKSPACE_DIR="$2"
			SET_WORKSPACE_DIR=1
			shift 2
			;;
		--llvm-source-dir)
			[[ $# -ge 2 ]] || die "--llvm-source-dir 缺少参数"
			LLVM_SOURCE_DIR="$2"
			SET_LLVM_SOURCE_DIR=1
			shift 2
			;;
		--llvm-build-dir)
			[[ $# -ge 2 ]] || die "--llvm-build-dir 缺少参数"
			LLVM_BUILD_DIR="$2"
			SET_LLVM_BUILD_DIR=1
			shift 2
			;;
		--pto-source-dir)
			[[ $# -ge 2 ]] || die "--pto-source-dir 缺少参数"
			PTO_SOURCE_DIR="$2"
			SET_PTO_SOURCE_DIR=1
			shift 2
			;;
		--pto-install-dir)
			[[ $# -ge 2 ]] || die "--pto-install-dir 缺少参数"
			PTO_INSTALL_DIR="$2"
			SET_PTO_INSTALL_DIR=1
			shift 2
			;;
		--python-bin)
			[[ $# -ge 2 ]] || die "--python-bin 缺少参数"
			PYTHON_BIN="$2"
			shift 2
			;;
		--jobs)
			[[ $# -ge 2 ]] || die "--jobs 缺少参数"
			JOBS="$2"
			shift 2
			;;
		--skip-python-deps)
			SKIP_PY_DEPS=1
			shift
			;;
		--dry-run)
			DRY_RUN=1
			shift
			;;
		--help | -h)
			usage
			exit 0
			;;
		*)
			die "未知参数: $1"
			;;
		esac
	done

	if [[ ${SET_WORKSPACE_DIR} -eq 1 && ${SET_LLVM_SOURCE_DIR} -eq 0 ]]; then
		LLVM_SOURCE_DIR="${WORKSPACE_DIR}/llvm-project"
	fi
	if [[ ${SET_WORKSPACE_DIR} -eq 1 && ${SET_LLVM_BUILD_DIR} -eq 0 && ${SET_LLVM_SOURCE_DIR} -eq 0 ]]; then
		LLVM_BUILD_DIR="${LLVM_SOURCE_DIR}/build-shared"
	fi
	if [[ ${SET_LLVM_SOURCE_DIR} -eq 1 && ${SET_LLVM_BUILD_DIR} -eq 0 ]]; then
		LLVM_BUILD_DIR="${LLVM_SOURCE_DIR}/build-shared"
	fi
	if [[ ${SET_PTO_SOURCE_DIR} -eq 1 && ${SET_PTO_INSTALL_DIR} -eq 0 ]]; then
		PTO_INSTALL_DIR="${PTO_SOURCE_DIR}/install"
	fi
}

validate_env() {
	if [[ ${DRY_RUN} -eq 0 ]]; then
		[[ -n "${PYTHON_BIN}" ]] || die "无法找到 python3，请通过 --python-bin 指定"
		[[ -x "${PYTHON_BIN}" ]] || die "Python 不可执行: ${PYTHON_BIN}"
	fi
	need_cmd git
	need_cmd cmake
	need_cmd ninja
}

prepare_repos() {
	run_cmd mkdir -p "${WORKSPACE_DIR}"

	if [[ ! -d "${LLVM_SOURCE_DIR}" ]]; then
		log "LLVM 源码目录不存在，开始 clone"
		run_cmd git clone --depth=1 --branch "${LLVM_BRANCH}" --single-branch "${LLVM_REPO_URL}" "${LLVM_SOURCE_DIR}"
	else
		log "LLVM 源码目录已存在，跳过 clone: ${LLVM_SOURCE_DIR}"
	fi

	if [[ ! -d "${PTO_SOURCE_DIR}" ]]; then
		log "PTOAS 源码目录不存在，开始 clone"
		run_cmd git clone "${PTO_REPO_URL}" "${PTO_SOURCE_DIR}"
	else
		log "PTOAS 源码目录已存在，跳过 clone: ${PTO_SOURCE_DIR}"
	fi
}

install_python_deps() {
	if [[ ${SKIP_PY_DEPS} -eq 1 ]]; then
		log "按参数要求跳过 Python 依赖安装"
		return 0
	fi
	run_cmd "${PYTHON_BIN}" -m pip install pybind11 numpy
}

checkout_llvm_branch() {
	if [[ ${DRY_RUN} -eq 0 && ! -d "${LLVM_SOURCE_DIR}/.git" ]]; then
		die "LLVM 目录不是 git 仓库: ${LLVM_SOURCE_DIR}"
	fi
	run_cmd git -C "${LLVM_SOURCE_DIR}" fetch --all --tags
	run_cmd git -C "${LLVM_SOURCE_DIR}" checkout "${LLVM_BRANCH}"
}

configure_llvm() {
	run_cmd cmake -G Ninja \
		-S "${LLVM_SOURCE_DIR}/llvm" \
		-B "${LLVM_BUILD_DIR}" \
		-DLLVM_ENABLE_PROJECTS="${LLVM_ENABLE_PROJECTS}" \
		-DBUILD_SHARED_LIBS=ON \
		-DMLIR_ENABLE_BINDINGS_PYTHON=ON \
		-DPython3_EXECUTABLE="${PYTHON_BIN}" \
		-DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
		-DLLVM_TARGETS_TO_BUILD="${LLVM_TARGETS_TO_BUILD}"
}

build_llvm() {
	run_cmd ninja -C "${LLVM_BUILD_DIR}" -j "${JOBS}"
}

configure_ptoas() {
	if [[ ${DRY_RUN} -eq 0 ]]; then
		[[ -d "${PTO_SOURCE_DIR}" ]] || die "PTOAS 源码目录不存在: ${PTO_SOURCE_DIR}"
		[[ -d "${LLVM_BUILD_DIR}/lib/cmake/llvm" ]] || die "未找到 LLVM_DIR: ${LLVM_BUILD_DIR}/lib/cmake/llvm"
		[[ -d "${LLVM_BUILD_DIR}/lib/cmake/mlir" ]] || die "未找到 MLIR_DIR: ${LLVM_BUILD_DIR}/lib/cmake/mlir"
	fi

	local pybind11_cmake_dir
	if [[ ${DRY_RUN} -eq 1 ]]; then
		pybind11_cmake_dir="<pybind11-cmake-dir>"
	else
		pybind11_cmake_dir="$("${PYTHON_BIN}" -m pybind11 --cmakedir)" || die "获取 pybind11 cmake dir 失败，请检查 pybind11 是否安装"
	fi

	run_cmd cmake -G Ninja \
		-S "${PTO_SOURCE_DIR}" \
		-B "${PTO_SOURCE_DIR}/build" \
		-DLLVM_DIR="${LLVM_BUILD_DIR}/lib/cmake/llvm" \
		-DMLIR_DIR="${LLVM_BUILD_DIR}/lib/cmake/mlir" \
		-DPython3_EXECUTABLE="${PYTHON_BIN}" \
		-DPython3_FIND_STRATEGY=LOCATION \
		-Dpybind11_DIR="${pybind11_cmake_dir}" \
		-DMLIR_ENABLE_BINDINGS_PYTHON=ON \
		-DMLIR_PYTHON_PACKAGE_DIR="${LLVM_BUILD_DIR}/tools/mlir/python_packages/mlir_core" \
		-DCMAKE_INSTALL_PREFIX="${PTO_INSTALL_DIR}"
}

build_ptoas() {
	run_cmd ninja -C "${PTO_SOURCE_DIR}/build" -j "${JOBS}"
}

install_ptoas() {
	run_cmd ninja -C "${PTO_SOURCE_DIR}/build" install
}

print_summary() {
	cat <<EOF

[bootstrap] 完成。关键路径如下:
  WORKSPACE_DIR=${WORKSPACE_DIR}
  LLVM_SOURCE_DIR=${LLVM_SOURCE_DIR}
  LLVM_BUILD_DIR=${LLVM_BUILD_DIR}
  PTO_SOURCE_DIR=${PTO_SOURCE_DIR}
  PTO_INSTALL_DIR=${PTO_INSTALL_DIR}

[bootstrap] 运行时环境可通过以下命令加载:
  cd ${PTO_SOURCE_DIR}
  source scripts/ptoas_env.sh
EOF

	if [[ ${DRY_RUN} -eq 1 ]]; then
		echo
		echo "[bootstrap] 当前为 dry-run 模式，以上命令均未实际执行。"
	fi
}

main() {
	parse_args "$@"
	validate_env

	case "${MODE}" in
	all)
		prepare_repos
		install_python_deps
		checkout_llvm_branch
		configure_llvm
		build_llvm
		configure_ptoas
		build_ptoas
		install_ptoas
		;;
	prepare)
		prepare_repos
		install_python_deps
		;;
	llvm-only)
		prepare_repos
		install_python_deps
		checkout_llvm_branch
		configure_llvm
		build_llvm
		;;
	ptoas-only)
		install_python_deps
		configure_ptoas
		build_ptoas
		install_ptoas
		;;
	llvm-configure)
		configure_llvm
		;;
	llvm-build)
		build_llvm
		;;
	ptoas-configure)
		install_python_deps
		configure_ptoas
		;;
	ptoas-build)
		build_ptoas
		;;
	ptoas-install)
		install_ptoas
		;;
	*)
		die "不支持的 --mode: ${MODE}"
		;;
	esac

	print_summary
}

main "$@"
