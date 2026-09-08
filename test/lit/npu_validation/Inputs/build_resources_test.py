# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check build parallelism against container memory limits before compilation."""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BASH = shutil.which("bash")
SOURCE = Path(sys.argv[1]).resolve()


def check_jobs(memory_kib, requested, expected):
    with tempfile.TemporaryDirectory(prefix="ptoas build jobs ") as directory:
        root = Path(directory)
        source = (SOURCE / "build.sh").read_text()
        (root / "functions.sh").write_text(source.removesuffix('main "$@"\n'))
        script = root / "check.sh"
        script.write_text('''source ./functions.sh
nproc() { echo 128; }
build_memory_available_kib() { printf '%s\\n' "$TEST_MEMORY"; }
resolve_build_jobs
[[ "$JOBS" == "$TEST_EXPECTED" ]]
[[ "$CMAKE_BUILD_PARALLEL_LEVEL" == "$TEST_EXPECTED" ]]
''')
        env = dict(os.environ, TEST_MEMORY=str(memory_kib), JOBS=requested, TEST_EXPECTED=str(expected))
        result = subprocess.run([BASH, str(script)], cwd=root, env=env, text=True,
                                capture_output=True, timeout=10)
        if expected is None:
            assert result.returncode != 0 and "positive integer" in result.stderr, result
        else:
            assert result.returncode == 0, result.stdout + result.stderr
        print(f"PASS: memory_kib={memory_kib}, requested={requested!r}, jobs={expected}")


def check_memory_controller(version):
    with tempfile.TemporaryDirectory(prefix="ptoas cgroup ") as directory:
        root = Path(directory)
        (root / "functions.sh").write_text((SOURCE / "build.sh").read_text().removesuffix('main "$@"\n'))
        proc = root / "proc"
        proc.mkdir()
        (proc / "meminfo").write_text("MemAvailable:   67108864 kB\n")
        cgroup = root / "cgroup"
        controller = cgroup / "memory" if version == 1 else cgroup
        controller.mkdir(parents=True)
        limit = controller / ("memory.limit_in_bytes" if version == 1 else "memory.max")
        usage = controller / ("memory.usage_in_bytes" if version == 1 else "memory.current")
        usage.write_text(str(2 * 1024**3) + "\n")
        script = root / "check.sh"
        script.write_text('source ./functions.sh\nbuild_memory_available_kib ./proc ./cgroup\n')
        for value, expected in [(str(10 * 1024**3), [67108864, 8388608]),
                                ("max", [67108864]), ("9223372036854771712", [67108864]),
                                ("1024", [67108864, 0])]:
            limit.write_text(value + "\n")
            result = subprocess.run([BASH, str(script)], cwd=root, text=True, capture_output=True, timeout=10)
            assert result.returncode == 0, result.stderr
            assert [int(line) for line in result.stdout.splitlines()] == expected, result.stdout
        print(f"PASS: cgroup v{version} finite, unlimited and exhausted memory budgets")


for memory, requested, expected in [(64 * 1024**2, "", 16), (18 * 1024**2, "", 8),
                                    (2 * 1024**2, "", 1), (0, "", 1), ("", "", 16),
                                    (18 * 1024**2, "4", 4), (18 * 1024**2, "64", 8),
                                    (256 * 1024**2, "64", 64)]:
    check_jobs(memory, requested, expected)
for invalid in ("0", "-1", "hello", "01", "99999999999999999999"):
    check_jobs(64 * 1024**2, invalid, None)
check_memory_controller(1)
check_memory_controller(2)
