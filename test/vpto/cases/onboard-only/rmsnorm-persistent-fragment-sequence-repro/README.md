# RMSNorm Persistent Fragment Sequence Reproducer

This temporary, hardware-only directory preserves the source boundary of the
state/order-dependent failure seen when three different persistent-fragment
RMSNorm kernels run in one process. It is intentionally self-contained so it
can be removed as one directory after the root cause is understood.

No TileLang checkout is required. No generated object, shared library, runner,
or compiler log is tracked; all build output goes to the ignored `build/`
directory (or to `BUILD_DIR` when overridden).

## Source Layout

- `kernels/d*/kernel.ptodsl.py`: TileLang-generated PTODSL source retained for
  frontend comparison. It is not invoked by `build.sh`.
- `kernels/d*/kernel.pto`: textual PTO IR compiled by the current PTOAS.
- `kernels/d*/kernel_vpto.ll`: VPTO LLVM IR emitted from the corresponding
  `kernel.pto` with the current PTOAS.
- `kernels/d*/launch.cpp`: AOT launch wrapper and dynamic-UB contract.
- `host/sequence_main.cpp`: one ACL process, one device, one stream, and the
  ordered kernel launches.

The launch contract is:

| `d` | kernel symbol | grid | SIMT threads | dynamic UB bytes |
| ---: | --- | ---: | ---: | ---: |
| 4096 | `rmsnorm_d4096_kernel` | 64 | 128 | 82432 |
| 5120 | `rmsnorm_d5120_kernel` | 64 | 256 | 152576 |
| 7168 | `rmsnorm_d7168_kernel` | 64 | 256 | 160768 |

These are the persistent-fragment UB sizes. Do not replace them with the
non-persistent baseline sizes (`82496`, `152640`, and `160832`).

## Build

From this directory:

```bash
CANN_HOME=/path/to/cann \
PTOAS_BIN=/path/to/PTOAS/build/tools/ptoas/ptoas \
./build.sh
```

By default, `PTOAS_BIN` resolves to the `build/tools/ptoas/ptoas` binary in
this PTOAS checkout, and `CANN_HOME` resolves to the local CANN 9.1 T530
installation used for the original reproduction.

To keep every generated file outside the checkout, use the same `BUILD_DIR`
for both build and run:

```bash
BUILD_DIR=/tmp/rmsnorm-persistent-sequence ./build.sh
BUILD_DIR=/tmp/rmsnorm-persistent-sequence \
  ASCEND_RT_VISIBLE_DEVICES=0 ACL_DEVICE_ID=0 \
  RMSNORM_SEQUENCE_REPEATS=1 ./run.sh
```

To regenerate the checked-in LLVM IR without producing device binaries:

```bash
PTOAS_BIN=/home/qukelin/projects/PTOAS/build/tools/ptoas/ptoas
for d in 4096 5120 7168; do
  "${PTOAS_BIN}" --pto-arch=a5 --pto-backend=vpto \
    --emit-vpto-llvm-ir "kernels/d${d}/kernel.pto" \
    -o "kernels/d${d}/kernel_vpto.ll"
done
```

## Run On Hardware

The captured host defaults to two launches of 4096, two launches of 5120, and
one launch of 7168. This was the stronger state/order-dependent reproduction:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 ACL_DEVICE_ID=0 ./run.sh
```

To run exactly one launch of each shape in the same process:

```bash
RMSNORM_SEQUENCE_REPEATS=1 ./run.sh
```

Use fresh processes for the single-kernel controls:

```bash
RMSNORM_ONLY=4096 RMSNORM_SEQUENCE_REPEATS=1 ./run.sh
RMSNORM_ONLY=5120 RMSNORM_SEQUENCE_REPEATS=1 ./run.sh
RMSNORM_ONLY=7168 RMSNORM_SEQUENCE_REPEATS=1 ./run.sh
```

Each launch is followed by `aclrtSynchronizeStream`. The failing sequence has
reported runtime error `507035` and vector GM read error `355`; each shape has
also passed when launched alone. The failing symbol is state/order dependent
and is not necessarily the final 7168 kernel. A reproduced fault exits with a
nonzero status; a completed sequence prints sampled 7168 `rstd` values.

On 2026-07-27, a one-launch sequence built with the current PTOAS checkout
reproduced the fault after these markers:

```text
launch 4096 repeat=0
launch 5120 repeat=0
fault kernel_name=rmsnorm_d5120_kernel
runtime result = 507035
errcode:(355) The address for VEC to read GM is out of bounds (exceeding 48 bits)
```

Run this reproducer on a real device because camodel does not preserve
persistent SIMT register state across SIMTVF exits.
