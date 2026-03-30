#!/usr/bin/env bash
set -euo pipefail

ptoas_bin="./build/tools/ptoas/ptoas"
vpto_ops_td="include/PTO/IR/VPTOOps.td"
filecheck_candidates=("FileCheck" "FileCheck-19" "/usr/lib/llvm-19/bin/FileCheck")
filecheck_bin=""

for candidate in "${filecheck_candidates[@]}"; do
  if command -v "${candidate}" >/dev/null 2>&1; then
    filecheck_bin="$(command -v "${candidate}")"
    break
  fi
  if [[ "${candidate}" = /* && -x "${candidate}" ]]; then
    filecheck_bin="${candidate}"
    break
  fi
done

if [[ -z "${filecheck_bin}" ]]; then
  echo "error: missing FileCheck; checked: ${filecheck_candidates[*]}" >&2
  exit 1
fi

if [[ ! -x "${ptoas_bin}" ]]; then
  echo "error: missing ./build/tools/ptoas/ptoas" >&2
  exit 1
fi

for required in CopyGmToUbuf CopyUbufToGm Vlds Vabs Vsts; do
  rg -n "def PTO_${required}Op" "${vpto_ops_td}" >/dev/null
done

if rg -n 'pto\.(load|store|abs)\b' "${vpto_ops_td}" >/dev/null; then
  echo "error: legacy pseudo-op names detected in ${vpto_ops_td}" >&2
  exit 1
fi

if rg -n 'pto\.(load|store|abs)\b|tabs_precheck\.mlir' test/phase2/*.mlir >/dev/null; then
  echo "error: obsolete Phase 2 fixture content detected" >&2
  exit 1
fi

if ! rg -n 'cce_aiv_loop_hint|pto\.vecscope' test/phase2/tabs_abs_loop_shape.mlir >/dev/null; then
  echo "error: tabs_abs_loop_shape.mlir must require explicit vecscope markers" >&2
  exit 1
fi

if rg -n '^// CHECK(?:(?:-[A-Z]+)?)?: scf\.for$' test/phase2/tabs_abs_loop_shape.mlir >/dev/null; then
  echo "error: tabs_abs_loop_shape.mlir still checks bare scf.for nesting without vec-scope carrier details" >&2
  exit 1
fi

echo "phase2 check: tload_copy_family_shape.mlir"
"${ptoas_bin}" --pto-backend=vpto --emit-vpto test/phase2/tload_copy_family_shape.mlir -o - 2>/dev/null | \
  "${filecheck_bin}" test/phase2/tload_copy_family_shape.mlir

echo "phase2 check: tabs_abs_loop_shape.mlir"
"${ptoas_bin}" --pto-backend=vpto --emit-vpto test/phase2/tabs_abs_loop_shape.mlir -o - 2>/dev/null | \
  "${filecheck_bin}" test/phase2/tabs_abs_loop_shape.mlir

echo "phase2 check: tabs_precheck_a5.mlir"
{ "${ptoas_bin}" --pto-backend=vpto test/phase2/tabs_precheck_a5.mlir -o /dev/null 2>&1 || true; } | \
  "${filecheck_bin}" test/phase2/tabs_precheck_a5.mlir

echo "phase2 check: tstore_copy_family_shape.mlir"
"${ptoas_bin}" --pto-backend=vpto --emit-vpto test/phase2/tstore_copy_family_shape.mlir -o - 2>/dev/null | \
  "${filecheck_bin}" test/phase2/tstore_copy_family_shape.mlir

echo "phase2 check: copy_dynamic_transfer_operands.mlir"
"${ptoas_bin}" --pto-backend=vpto --emit-vpto test/phase2/copy_dynamic_transfer_operands.mlir -o - 2>/dev/null | \
  "${filecheck_bin}" test/phase2/copy_dynamic_transfer_operands.mlir

echo "phase2 check: copy_dynamic_transfer_operands.mlir HIVM names"
"${ptoas_bin}" --pto-arch=a5 --pto-backend=vpto --vpto-emit-hivm-llvm test/phase2/copy_dynamic_transfer_operands.mlir -o - 2>/dev/null | \
  "${filecheck_bin}" --check-prefix=CHECK-HIVM test/phase2/copy_dynamic_transfer_operands.mlir

echo "phase2 check: vpto_multi_aivector_scope_metadata.mlir"
"${ptoas_bin}" --pto-arch=a5 --pto-backend=vpto --vpto-emit-hivm-llvm test/phase2/vpto_multi_aivector_scope_metadata.mlir -o - 2>/dev/null | \
  "${filecheck_bin}" test/phase2/vpto_multi_aivector_scope_metadata.mlir

echo "phase2 check: vpto_vcvt_emit_hivm_llvm.mlir"
"${ptoas_bin}" --pto-arch=a5 --pto-backend=vpto --vpto-emit-hivm-llvm test/phase2/vpto_vcvt_emit_hivm_llvm.mlir -o - 2>/dev/null | \
  "${filecheck_bin}" test/phase2/vpto_vcvt_emit_hivm_llvm.mlir

echo "phase2 check: tstore_domain_todos.mlir"
{ "${ptoas_bin}" --pto-backend=vpto --emit-vpto test/phase2/tstore_domain_todos.mlir -o - 2>&1 || true; } | \
  "${filecheck_bin}" test/phase2/tstore_domain_todos.mlir

echo "phase2 check: pto_backend_vpto_wiring.mlir"
"${ptoas_bin}" --pto-backend=vpto --emit-vpto test/phase2/pto_backend_vpto_wiring.mlir -o - 2>/dev/null | \
  "${filecheck_bin}" --check-prefix=VPTO test/phase2/pto_backend_vpto_wiring.mlir

echo "phase2 check: pto_backend_vpto_wiring.mlir EmitC smoke"
"${ptoas_bin}" --pto-arch=a5 --pto-backend=emitc test/phase2/pto_backend_vpto_wiring.mlir -o - 2>/dev/null | \
  "${filecheck_bin}" --check-prefix=EMITC test/phase2/pto_backend_vpto_wiring.mlir

echo "phase2 check: vpto_fusion_aivector_scope_loop_preserved_after_canonicalize.mlir"
"${ptoas_bin}" test/samples/PyPTOIRParser/paged_attention_example_kernel_online_update.pto --enable-op-fusion --pto-arch=a5 --pto-backend=vpto --print-ir-after-all --print-ir-after-all-func-filter=kernel_online_update -o /dev/null > /tmp/vpto_fusion_aivector_scope_loop_preserved.out 2>&1
awk '/IR Dump After PTOLowLevelLoopFusion/{seen_low=1; next} seen_low && /IR Dump After Canonicalizer/{found=1} found{if (found > 1 && /IR Dump After /) exit; print; found=2}' /tmp/vpto_fusion_aivector_scope_loop_preserved.out | \
  "${filecheck_bin}" test/phase2/vpto_fusion_aivector_scope_loop_preserved_after_canonicalize.mlir

echo "phase2 check: vpto_fusion_pipeline_order.mlir"
"${ptoas_bin}" test/samples/PyPTOIRParser/paged_attention_example_kernel_online_update.pto --enable-op-fusion --pto-arch=a5 --pto-backend=vpto --print-ir-after-all --print-ir-after-all-func-filter=kernel_online_update -o /dev/null > /tmp/vpto_fusion_pipeline_order.out 2>&1
"${filecheck_bin}" test/phase2/vpto_fusion_pipeline_order.mlir < /tmp/vpto_fusion_pipeline_order.out
! rg 'IR Dump After (PTOValidateSimdIR|PTOInstantiateAndLowerToLibCall|PTOInlineLibCall)' /tmp/vpto_fusion_pipeline_order.out

vpto_fusion_region_lifecycle_out="/tmp/vpto_fusion_region_lifecycle.out"
"${ptoas_bin}" test/samples/PyPTOIRParser/paged_attention_example_kernel_online_update.pto --enable-op-fusion --pto-arch=a5 --pto-backend=vpto --print-ir-after-all --print-ir-after-all-func-filter=kernel_online_update -o /dev/null > "${vpto_fusion_region_lifecycle_out}" 2>&1

echo "phase2 check: vpto_fusion_region_lifecycle.mlir (low-level)"
awk '/IR Dump After PTOLowLevelLoopFusion/{found=1} found{if ($0 ~ /^\/\/ -----\/\/ IR Dump After / && $0 !~ /PTOLowLevelLoopFusion/) exit; print}' "${vpto_fusion_region_lifecycle_out}" | \
  "${filecheck_bin}" --check-prefix=LOW test/phase2/vpto_fusion_region_lifecycle.mlir

echo "phase2 check: vpto_fusion_region_lifecycle.mlir (flatten)"
awk '/IR Dump After PTOFlattenFusionRegion/{found=1} found{if ($0 ~ /^\/\/ -----\/\/ IR Dump After / && $0 !~ /PTOFlattenFusionRegion/) exit; print}' "${vpto_fusion_region_lifecycle_out}" | \
  "${filecheck_bin}" --check-prefix=FLAT test/phase2/vpto_fusion_region_lifecycle.mlir

echo "phase2 check: ctest"
ctest --test-dir build --output-on-failure
