---
name: ptoas-abs-hivm-bitcode
description: Export the Abs sample through the A5VM backend as LLVM bitcode, verify the output is real bitcode instead of MLIR text, and compile it with bisheng. Use when the user wants Abs as LLVM bitcode, wants the relevant ptoas option for bitcode emission, or wants to validate the bisheng handoff for dav-c310-vec.
---

# PTOAS Abs HiVM Bitcode

Use this skill when the task is specifically about:
- exporting `test/samples/Abs/abs.py` as LLVM bitcode
- finding or using the `ptoas` bitcode emission option for the A5VM backend
- validating the generated HiVM/LLVM bitcode with LLVM tools
- compiling the exported bitcode with `bisheng`

## Canonical Option

`ptoas` already supports this path:

```bash
--a5vm-emit-hivm-bc
```

This option only works with:

```bash
--pto-backend=a5vm
```

## Canonical Commands

### 1. Build `ptoas` if needed

```bash
CCACHE_DISABLE=1 ninja -C build ptoas
```

### 2. Prepare the runtime environment

Always load the repo environment before running the sample:

```bash
source env.sh
```

### 3. Export `Abs` as LLVM bitcode

```bash
source env.sh
PTOAS_BIN="$PWD/build/tools/ptoas/ptoas" \
PTOAS_OUT_DIR=/tmp/ptoas-abs-hivm-bc \
PTOAS_FLAGS='--pto-arch a5 --pto-backend=a5vm --a5vm-emit-hivm-bc' \
./test/samples/runop.sh -t Abs
```

Expected outputs:
- `/tmp/ptoas-abs-hivm-bc/Abs/abs-pto-ir.pto`
- `/tmp/ptoas-abs-hivm-bc/Abs/abs-pto.cpp`

Important:
- `runop.sh` still writes the bitcode payload to `abs-pto.cpp`
- that file is not C++ source in this mode; it is binary LLVM bitcode

### 4. Verify the output is bitcode

```bash
file /tmp/ptoas-abs-hivm-bc/Abs/abs-pto.cpp
xxd -l 16 /tmp/ptoas-abs-hivm-bc/Abs/abs-pto.cpp
source env.sh
"$LLVM_ROOT/bin/llvm-dis" /tmp/ptoas-abs-hivm-bc/Abs/abs-pto.cpp -o - | sed -n '1,80p'
```

Sanity checks:
- `file` should report `LLVM IR bitcode`
- the header should start with `42 43 c0 de`
- `llvm-dis` should show HiVM intrinsics such as `@llvm.hivm.vabs`

### 5. Compile the bitcode with `bisheng`

Preferred: force the language explicitly with `-x ir`, because the sample script still leaves the bitcode payload in a file named `abs-pto.cpp`.

```bash
bisheng \
  --target=hiipu64-hisilicon-cce \
  -march=dav-c310-vec \
  --cce-aicore-arch=dav-c310-vec \
  -c -x ir /tmp/ptoas-abs-hivm-bc/Abs/abs-pto.cpp \
  -o /tmp/ptoas-abs-hivm-bc/Abs/abs-pto.o
```

Alternative: rename or copy it to `.bc` so `bisheng` recognizes it as bitcode by suffix.

```bash
cp /tmp/ptoas-abs-hivm-bc/Abs/abs-pto.cpp /tmp/ptoas-abs-hivm-bc/Abs/abs-pto.bc
bisheng \
  --target=hiipu64-hisilicon-cce \
  -march=dav-c310-vec \
  --cce-aicore-arch=dav-c310-vec \
  -c /tmp/ptoas-abs-hivm-bc/Abs/abs-pto.bc \
  -o /tmp/ptoas-abs-hivm-bc/Abs/abs-pto.o
```

Optional verification:

```bash
file /tmp/ptoas-abs-hivm-bc/Abs/abs-pto.o
```

Expected result:
- an ELF relocatable object is produced
- `bisheng` may warn that it is overriding the module target triple; that is not a hard failure

## Failure Modes

If export fails, report the first concrete blocker:
- `--a5vm-emit-hivm-bc` used without `--pto-backend=a5vm`
- `env.sh` was not sourced
- `bisheng` not found in `PATH`
- `llvm-dis` not available under `$LLVM_ROOT/bin`

If `bisheng` reports UTF-8 or C++ parse errors on the bitcode file:
- the input is being treated as source because it still has a suffix such as `.cpp`
- rerun with `-x ir`, or copy/rename it to `.bc`

## Reporting Back

When you use this skill, report:
- the exact `ptoas` flags used
- the exact output path that contains the bitcode payload
- whether `file` confirmed `LLVM IR bitcode`
- whether `llvm-dis` shows HiVM/LLVM content
- whether `bisheng` produced an object file, and whether that used `-x ir` or a `.bc` rename/copy
