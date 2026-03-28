// RUN: ! python3 %S/../../oplib/level3/family_dsl.py --family-dsl=%S/resources/bad_family_dsl_empty_patterns.json > %t.family_dsl.log 2>&1
// RUN: FileCheck %s --check-prefix=BAD-FAMILY-DSL < %t.family_dsl.log
// RUN: rm -rf %t.bad_manifest && mkdir -p %t.bad_manifest/families
// RUN: cp %S/../../oplib/level3/cmp_tile_tile_templates.mlir %t.bad_manifest/
// RUN: cp %S/resources/bad_manifest_schema.yaml %t.bad_manifest/families/a5_oplib_v1_manifest.yaml
// RUN: ! ptoas %S/compare_family.pto --enable-op-fusion --op-lib-dir=%t.bad_manifest --pto-arch=a5 -o %t.bad_manifest.cpp > %t.bad_manifest.log 2>&1
// RUN: FileCheck %s --check-prefix=BAD-MANIFEST < %t.bad_manifest.log
// RUN: rm -rf %t.bad_template && mkdir -p %t.bad_template/families
// XFAIL: *
// RUN: cp %S/resources/bad_family_signature_template.txt %t.bad_template/bad.mlir
// RUN: cp %S/../../oplib/level3/families/a5_oplib_v1_manifest.yaml %t.bad_template/families/
// RUN: ! ptoas %S/compare_family.pto --enable-op-fusion --op-lib-dir=%t.bad_template --pto-arch=a5 -o %t.bad_template.cpp > %t.bad_template.log 2>&1
// RUN: FileCheck %s --check-prefix=BAD-TEMPLATE-IMPORT < %t.bad_template.log
// RUN: rm -rf %t.no_candidate && mkdir -p %t.no_candidate/families
// RUN: cp %S/../../oplib/level3/select_mask_templates.mlir %t.no_candidate/
// RUN: cp %S/../../oplib/level3/families/a5_oplib_v1_manifest.yaml %t.no_candidate/families/
// RUN: ! ptoas %S/compare_family.pto --enable-op-fusion --op-lib-dir=%t.no_candidate --pto-arch=a5 -o %t.no_candidate.cpp > %t.no_candidate.log 2>&1
// RUN: FileCheck %s --check-prefix=BAD-NO-CANDIDATE < %t.no_candidate.log
// RUN: rm -rf %t.bad_rewrite && mkdir -p %t.bad_rewrite/families
// RUN: cp %S/../../oplib/level3/float_unary_templates.mlir %t.bad_rewrite/
// RUN: cp %S/resources/bad_manifest_trecip_deferred.json %t.bad_rewrite/families/a5_oplib_v1_manifest.yaml
// RUN: ! ptoas %S/compare_family.pto --enable-op-fusion --op-lib-dir=%t.bad_rewrite --pto-arch=a5 -o %t.bad_rewrite.cpp > %t.bad_rewrite.log 2>&1
// RUN: FileCheck %s --check-prefix=BAD-REWRITE-STATE < %t.bad_rewrite.log

// BAD-FAMILY-DSL: family DSL requires a non-empty 'patterns' list
// BAD-MANIFEST: A5 OpLib V1 manifest
// BAD-MANIFEST: has unexpected schema_version 'broken_manifest/v0'
// BAD-TEMPLATE-IMPORT: invalid OP-Lib signature for kind=l3_float_unary_template
// BAD-NO-CANDIDATE: manifest-implemented op='tcmp' family=compare_tile_tile classification=native_a5_impl has no OP-Lib candidate for op=tcmp dtype=f32
// BAD-REWRITE-STATE: keeps approved public_api_rewrite op 'trecip' deferred; expected implemented

module {
}
