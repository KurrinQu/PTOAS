// RUN: ! ptoas %S/resources/bad_family_signature_template.txt --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 -o %t.bad_sig.cpp > %t.bad_sig.log 2>&1
// RUN: FileCheck %s --check-prefix=BAD-SIG < %t.bad_sig.log
// RUN: ! ptoas %S/resources/bad_missing_arg_match_template.txt --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 -o %t.bad_arg.cpp > %t.bad_arg.log 2>&1
// RUN: FileCheck %s --check-prefix=BAD-ARG < %t.bad_arg.log
// RUN: ! ptoas %S/resources/bad_invalid_scalar_pos_template.txt --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 -o %t.bad_scalar.cpp > %t.bad_scalar.log 2>&1
// RUN: FileCheck %s --check-prefix=BAD-SCALAR < %t.bad_scalar.log
// RUN: ! ptoas %S/resources/bad_missing_cmp_mode_module.mlir --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 -o %t.bad_cmp.cpp > %t.bad_cmp.log 2>&1
// RUN: FileCheck %s --check-prefix=BAD-CMP < %t.bad_cmp.log
// RUN: ! ptoas %S/resources/bad_missing_is_binary_template.txt --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 -o %t.bad_binary.cpp > %t.bad_binary.log 2>&1
// RUN: FileCheck %s --check-prefix=BAD-IS-BINARY < %t.bad_binary.log
// RUN: ! ptoas %S/resources/bad_vector_unsupported_template.txt --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 -o %t.bad_vec.cpp > %t.bad_vec.log 2>&1
// RUN: FileCheck %s --check-prefix=BAD-VEC < %t.bad_vec.log

// BAD-SIG: invalid OP-Lib signature for kind=l3_float_unary_template
// BAD-ARG: missing or invalid arg1 match attrs
// BAD-SCALAR: invalid pto.oplib.match.scalar_pos
// BAD-CMP: missing required attr: pto.oplib.match.cmp_mode
// BAD-IS-BINARY: missing required attr: pto.oplib.match.is_binary
// BAD-VEC: E_OPLIB_SIMD_LANES_MISMATCH
// BAD-VEC: requires f32 vectors to use 64 lanes
