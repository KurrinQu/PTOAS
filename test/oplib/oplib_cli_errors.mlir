// RUN: ! ptoas %S/softmax_chain.pto --pto-arch=a5 -o %t.no_dir.cpp > %t.no_dir.log 2>&1
// RUN: FileCheck %s --check-prefix=NO-DIR < %t.no_dir.log
// RUN: ! ptoas %S/softmax_chain.pto --enable-op-fusion --disable-oplib-lowering --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 -o %t.dep.cpp > %t.dep.log 2>&1
// RUN: FileCheck %s --check-prefix=REMOVED < %t.dep.log
// RUN: rm -rf %t.empty && mkdir -p %t.empty
// RUN: cp %S/resources/bad_empty_simd_template.txt %t.empty/bad.mlir
// RUN: ! ptoas %S/softmax_chain.pto --op-lib-dir=%t.empty --pto-arch=a5 -o %t.empty.cpp > %t.empty.log 2>&1
// RUN: FileCheck %s --check-prefix=EMPTY-SIMD < %t.empty.log
// RUN: rm -rf %t.core && mkdir -p %t.core
// RUN: cp %S/resources/bad_simd_core_slot_template.txt %t.core/bad.mlir
// RUN: ! ptoas %S/softmax_chain.pto --op-lib-dir=%t.core --pto-arch=a5 -o %t.core.cpp > %t.core.log 2>&1
// RUN: FileCheck %s --check-prefix=BAD-CORE < %t.core.log
// RUN: rm -rf %t.lanes && mkdir -p %t.lanes
// RUN: cp %S/resources/bad_simd_lanes_template.txt %t.lanes/bad.mlir
// RUN: ! ptoas %S/softmax_chain.pto --op-lib-dir=%t.lanes --pto-arch=a5 -o %t.lanes.cpp > %t.lanes.log 2>&1
// RUN: FileCheck %s --check-prefix=BAD-LANES < %t.lanes.log
// RUN: rm -rf %t.simd_attr && mkdir -p %t.simd_attr
// RUN: cp %S/resources/bad_simd_missing_attrs_template.txt %t.simd_attr/bad.mlir
// RUN: ! ptoas %S/softmax_chain.pto --op-lib-dir=%t.simd_attr --pto-arch=a5 -o %t.simd_attr.cpp > %t.simd_attr.log 2>&1
// RUN: FileCheck %s --check-prefix=BAD-SIMD-ATTR < %t.simd_attr.log
// RUN: rm -rf %t.disallowed && mkdir -p %t.disallowed
// RUN: cp %S/resources/bad_disallowed_ir_template.txt %t.disallowed/bad.mlir
// RUN: ! ptoas %S/softmax_chain.pto --op-lib-dir=%t.disallowed --pto-arch=a5 -o %t.disallowed.cpp > %t.disallowed.log 2>&1
// RUN: FileCheck %s --check-prefix=BAD-IR < %t.disallowed.log
// RUN: rm -rf %t.bad_vec && mkdir -p %t.bad_vec
// RUN: cp %S/resources/bad_vector_unsupported_template.txt %t.bad_vec/bad.mlir
// RUN: ! ptoas %S/softmax_chain.pto --op-lib-dir=%t.bad_vec --pto-arch=a5 -o %t.bad_vec.cpp > %t.bad_vec.log 2>&1
// RUN: FileCheck %s --check-prefix=BAD-VEC < %t.bad_vec.log
// RUN: rm -rf %t.legacy && mkdir -p %t.legacy
// RUN: cp %S/resources/bad_legacy_unrealized_cast_template.txt %t.legacy/bad.mlir
// RUN: ! ptoas %S/softmax_chain.pto --op-lib-dir=%t.legacy --pto-arch=a5 -o %t.legacy.cpp > %t.legacy.log 2>&1
// RUN: FileCheck %s --check-prefix=BAD-LEGACY-CAST < %t.legacy.log
// RUN: rm -rf %t.missing_knobs && mkdir -p %t.missing_knobs
// RUN: cp %S/resources/bad_vector_missing_knobs_template.txt %t.missing_knobs/bad.mlir
// RUN: ! ptoas %S/softmax_chain.pto --op-lib-dir=%t.missing_knobs --pto-arch=a5 -o %t.missing_knobs.cpp > %t.missing_knobs.log 2>&1
// RUN: FileCheck %s --check-prefix=BAD-MISSING-KNOBS < %t.missing_knobs.log
// RUN: rm -rf %t.bad_prefix && mkdir -p %t.bad_prefix
// RUN: cp %S/resources/bad_vector_bad_prefix_template.txt %t.bad_prefix/bad.mlir
// RUN: ! ptoas %S/softmax_chain.pto --op-lib-dir=%t.bad_prefix --pto-arch=a5 -o %t.bad_prefix.cpp > %t.bad_prefix.log 2>&1
// RUN: FileCheck %s --check-prefix=BAD-PREFIX < %t.bad_prefix.log
// RUN: rm -rf %t.missing_vld && mkdir -p %t.missing_vld
// RUN: cp %S/resources/bad_vector_missing_vld_template.txt %t.missing_vld/bad.mlir
// RUN: ! ptoas %S/softmax_chain.pto --op-lib-dir=%t.missing_vld --pto-arch=a5 -o %t.missing_vld.cpp > %t.missing_vld.log 2>&1
// RUN: FileCheck %s --check-prefix=BAD-MISSING-VLD < %t.missing_vld.log
// RUN: rm -rf %t.missing_vst && mkdir -p %t.missing_vst
// RUN: cp %S/resources/bad_vector_missing_vst_template.txt %t.missing_vst/bad.mlir
// RUN: ! ptoas %S/softmax_chain.pto --op-lib-dir=%t.missing_vst --pto-arch=a5 -o %t.missing_vst.cpp > %t.missing_vst.log 2>&1
// RUN: FileCheck %s --check-prefix=BAD-MISSING-VST < %t.missing_vst.log
// RUN: rm -rf %t.bad_vst_prefix && mkdir -p %t.bad_vst_prefix
// RUN: cp %S/resources/bad_vector_bad_vst_prefix_template.txt %t.bad_vst_prefix/bad.mlir
// RUN: ! ptoas %S/softmax_chain.pto --op-lib-dir=%t.bad_vst_prefix --pto-arch=a5 -o %t.bad_vst_prefix.cpp > %t.bad_vst_prefix.log 2>&1
// RUN: FileCheck %s --check-prefix=BAD-VST-PREFIX < %t.bad_vst_prefix.log
// RUN: rm -rf %t.bad_sig && mkdir -p %t.bad_sig
// RUN: cp %S/resources/bad_family_signature_template.txt %t.bad_sig/bad.mlir
// RUN: ! ptoas %S/softmax_chain.pto --op-lib-dir=%t.bad_sig --pto-arch=a5 -o %t.bad_sig.cpp > %t.bad_sig.log 2>&1
// RUN: FileCheck %s --check-prefix=BAD-SIG < %t.bad_sig.log
// RUN: rm -rf %t.bad_arg_match && mkdir -p %t.bad_arg_match
// RUN: cp %S/resources/bad_missing_arg_match_template.txt %t.bad_arg_match/bad.mlir
// RUN: ! ptoas %S/softmax_chain.pto --op-lib-dir=%t.bad_arg_match --pto-arch=a5 -o %t.bad_arg_match.cpp > %t.bad_arg_match.log 2>&1
// RUN: FileCheck %s --check-prefix=BAD-ARG-MATCH < %t.bad_arg_match.log
// RUN: rm -rf %t.bad_scalar_pos && mkdir -p %t.bad_scalar_pos
// RUN: cp %S/resources/bad_invalid_scalar_pos_template.txt %t.bad_scalar_pos/bad.mlir
// RUN: ! ptoas %S/softmax_chain.pto --op-lib-dir=%t.bad_scalar_pos --pto-arch=a5 -o %t.bad_scalar_pos.cpp > %t.bad_scalar_pos.log 2>&1
// RUN: FileCheck %s --check-prefix=BAD-SCALAR-POS < %t.bad_scalar_pos.log
// RUN: rm -rf %t.bad_is_binary && mkdir -p %t.bad_is_binary
// RUN: cp %S/resources/bad_missing_is_binary_template.txt %t.bad_is_binary/bad.mlir
// RUN: ! ptoas %S/softmax_chain.pto --op-lib-dir=%t.bad_is_binary --pto-arch=a5 -o %t.bad_is_binary.cpp > %t.bad_is_binary.log 2>&1
// RUN: FileCheck %s --check-prefix=BAD-IS-BINARY < %t.bad_is_binary.log
// RUN: rm -rf %t.bad_math && mkdir -p %t.bad_math
// RUN: cp %S/resources/bad_disallowed_math_template.txt %t.bad_math/bad.mlir
// RUN: ! ptoas %S/softmax_chain.pto --op-lib-dir=%t.bad_math --pto-arch=a5 -o %t.bad_math.cpp > %t.bad_math.log 2>&1
// RUN: FileCheck %s --check-prefix=BAD-MATH < %t.bad_math.log
// RUN: rm -rf %t.bad_vec_int && mkdir -p %t.bad_vec_int
// RUN: cp %S/resources/bad_vector_int_unsupported_template.txt %t.bad_vec_int/bad.mlir
// RUN: ! ptoas %S/softmax_chain.pto --op-lib-dir=%t.bad_vec_int --pto-arch=a5 -o %t.bad_vec_int.cpp > %t.bad_vec_int.log 2>&1
// RUN: FileCheck %s --check-prefix=BAD-VEC-INT < %t.bad_vec_int.log
// NO-DIR: Error: --op-lib-dir is required.
// REMOVED: Unknown command line argument '--disable-oplib-lowering'
// EMPTY-SIMD: E_OPLIB_EMPTY_BODY_FOR_SIMD
// BAD-CORE: E_OPLIB_SIMD_INVALID_CORE_SLOT
// BAD-LANES: E_OPLIB_SIMD_LANES_MISMATCH
// BAD-LANES: requires pto.simd.lanes = 64, got 32
// BAD-SIMD-ATTR: E_OPLIB_SIMD_ATTR_REQUIRED
// BAD-IR: E_OPLIB_BODY_DISALLOWED_IR
// BAD-VEC: E_OPLIB_SIMD_LANES_MISMATCH
// BAD-VEC: requires f32 vectors to use 64 lanes
// BAD-LEGACY-CAST: E_OPLIB_BODY_DISALLOWED_IR
// BAD-MISSING-KNOBS: E_OPLIB_SIMD_ATTR_REQUIRED
// BAD-MISSING-KNOBS: pto.simd.exec_mode
// BAD-PREFIX: E_OPLIB_SIMD_ATTR_REQUIRED
// BAD-PREFIX: must start with 'MODE_'
// BAD-MISSING-VLD: E_OPLIB_SIMD_ATTR_REQUIRED
// BAD-MISSING-VLD: pto.simd.vld_dist
// BAD-MISSING-VST: E_OPLIB_SIMD_ATTR_REQUIRED
// BAD-MISSING-VST: pto.simd.vst_dist
// BAD-VST-PREFIX: E_OPLIB_SIMD_ATTR_REQUIRED
// BAD-VST-PREFIX: must start with 'DIST_'
// BAD-SIG: invalid OP-Lib signature for kind=l3_float_unary_template
// BAD-ARG-MATCH: missing or invalid arg1 match attrs
// BAD-SCALAR-POS: invalid pto.oplib.match.scalar_pos
// BAD-IS-BINARY: missing required attr: pto.oplib.match.is_binary
// BAD-MATH: E_OPLIB_BODY_DISALLOWED_IR
// BAD-MATH: math.sin
// BAD-VEC-INT: E_OPLIB_SIMD_LANES_MISMATCH
// BAD-VEC-INT: requires i32 vectors to use 64 lanes
