import os
import lit.formats

config.name = "PTOAS"
config.test_format = lit.formats.ShTest(execute_external=True)

# Keep discovery focused on lit-style tests.
config.suffixes = [".mlir"]
config.excludes = [
    "CMakeLists.txt",
    "README.md",
    "lit.cfg.py",
    "resources",
]

config.test_source_root = os.path.dirname(__file__)


def _resolve_build_root():
    env_build_dir = os.environ.get("PTOAS_BUILD_DIR")
    if env_build_dir:
        return os.path.abspath(env_build_dir)

    repo_root = os.path.abspath(os.path.join(config.test_source_root, ".."))
    return os.path.join(repo_root, "build")


build_root = _resolve_build_root()
config.test_exec_root = os.path.join(build_root, "test")
os.makedirs(config.test_exec_root, exist_ok=True)


def _sanitize_lit_test_times(exec_root):
    times_path = os.path.join(exec_root, ".lit_test_times.txt")
    if not os.path.isfile(times_path):
        return

    cleaned_lines = []
    saw_invalid_line = False
    with open(times_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line:
                saw_invalid_line = True
                continue

            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                saw_invalid_line = True
                continue

            try:
                float(parts[0])
            except ValueError:
                saw_invalid_line = True
                continue

            cleaned_lines.append(line)

    if not saw_invalid_line:
        return

    with open(times_path, "w", encoding="utf-8") as handle:
        for line in cleaned_lines:
            handle.write(line)
            handle.write("\n")


_sanitize_lit_test_times(config.test_exec_root)


def _resolve_llvm_bin_dir():
    env_build_dir = os.environ.get("LLVM_BUILD_DIR")
    candidates = []
    if env_build_dir:
        candidates.append(os.path.join(os.path.abspath(env_build_dir), "bin"))

    repo_root = os.path.abspath(os.path.join(config.test_source_root, ".."))
    candidates.append(
        os.path.abspath(
            os.path.join(repo_root, "..", "llvm-project", "build-shared", "bin")
        )
    )

    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return ""


def _resolve_ptoas_bin():
    env_bin = os.environ.get("PTOAS_BIN")
    if env_bin:
        return env_bin

    candidate = os.path.join(build_root, "tools", "ptoas", "ptoas")
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate

    return "ptoas"


def _prepend_path(path_var, entry):
    if not entry:
        return path_var
    if not path_var:
        return entry
    return entry + os.pathsep + path_var


ptoas_bin = _resolve_ptoas_bin()
ptoas_dir = os.path.dirname(ptoas_bin) if os.path.isabs(ptoas_bin) else ""
llvm_bin_dir = _resolve_llvm_bin_dir()

path_env = config.environment.get("PATH", os.environ.get("PATH", ""))
if llvm_bin_dir:
    path_env = _prepend_path(path_env, llvm_bin_dir)
if ptoas_dir:
    path_env = _prepend_path(path_env, ptoas_dir)
config.environment["PATH"] = path_env

# Keep RUN lines using bare `ptoas` stable regardless of shell cwd.
if os.path.isabs(ptoas_bin):
    config.substitutions.append(("ptoas", ptoas_bin))
