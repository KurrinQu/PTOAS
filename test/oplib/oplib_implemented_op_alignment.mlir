// RUN: python3 %S/check_implemented_op_alignment.py --manifest=%S/resources/implemented_op_alignment_subset.json --template-dir=%S/../../oplib/level3 --test-dir=%S

// This lit entry keeps the dtype-granular alignment checker wired against a
// subset manifest whose coverage is intentionally maintained in-tree.
module {
}
