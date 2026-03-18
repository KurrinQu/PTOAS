#!/usr/bin/env bash
set -euo pipefail

ptoas_bin="./build/tools/ptoas/ptoas"

if [[ ! -x "${ptoas_bin}" ]]; then
  echo "error: missing ./build/tools/ptoas/ptoas" >&2
  exit 1
fi

backend_report="$(mktemp)"
cleanup() {
  rm -f "${backend_report}"
}
trap cleanup EXIT

echo "phase1 check: a5vm_vec_type.mlir"
"${ptoas_bin}" test/phase1/a5vm_vec_type.mlir 2>&1 | FileCheck test/phase1/a5vm_vec_type.mlir

echo "phase1 check: a5vm_load_op.mlir"
"${ptoas_bin}" test/phase1/a5vm_load_op.mlir -o - | FileCheck test/phase1/a5vm_load_op.mlir

echo "phase1 check: a5vm_abs_op.mlir"
"${ptoas_bin}" test/phase1/a5vm_abs_op.mlir -o - | FileCheck test/phase1/a5vm_abs_op.mlir

echo "phase1 check: a5vm_store_op.mlir"
"${ptoas_bin}" test/phase1/a5vm_store_op.mlir -o - | FileCheck test/phase1/a5vm_store_op.mlir

echo "phase1 check: a5vm_backend_switch.mlir"
"${ptoas_bin}" --pto-backend=a5vm --a5vm-allow-unresolved --a5vm-unresolved-report="${backend_report}" \
  test/phase1/a5vm_backend_switch.mlir -o - | FileCheck test/phase1/a5vm_backend_switch.mlir

test -s "$backend_report"

echo "phase1 check: a5vm_backend_switch.mlir (intrinsics)"
"${ptoas_bin}" --pto-backend=a5vm --a5vm-print-intrinsics --a5vm-allow-unresolved \
  --a5vm-unresolved-report="${backend_report}" test/phase1/a5vm_backend_switch.mlir -o - 2>&1 | \
  rg "A5VM intrinsic:"

echo "phase1 check: a5vm_shared_dialects.mlir"
"${ptoas_bin}" --pto-backend=a5vm --a5vm-print-ir test/phase1/a5vm_shared_dialects.mlir -o /dev/null 2>&1 | \
  FileCheck test/phase1/a5vm_shared_dialects.mlir
