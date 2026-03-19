// RUN: ! python3 %S/check_implemented_op_alignment.py --manifest=%S/resources/bad_implemented_op_alignment_missing_dtype.json --template-dir=%S/../../oplib/level3 --test-dir=%S > %t.log 2>&1
// RUN: FileCheck %s --check-prefix=BAD-DTYPE < %t.log

// BAD-DTYPE: implemented ops missing concrete template dtype coverage:
// BAD-DTYPE-NEXT:   - trecip: missing dtypes bf16
// BAD-DTYPE: implemented ops missing lowering dtype coverage:
// BAD-DTYPE-NEXT:   - trecip: missing dtypes bf16

module {
}
