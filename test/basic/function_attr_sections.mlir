// RUN: ptoas %s 2>&1 | FileCheck %s --check-prefix=IR
// RUN: ptoas %s | FileCheck %s --check-prefix=CPP

module {
  func.func @cube_marked() attributes {pto.kernel_kind = #pto.kernel_kind<cube>} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %sum = arith.addi %c0, %c1 : i32
    return
  }

  func.func @vector_marked() attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %sum = arith.addi %c2, %c3 : i32
    return
  }
}

// IR-LABEL: func.func @cube_marked() attributes {pto.kernel_kind = #pto.kernel_kind<cube>}
// IR: pto.section.cube {

// IR-LABEL: func.func @vector_marked() attributes {pto.kernel_kind = #pto.kernel_kind<vector>}
// IR: pto.section.vector {

// CPP-LABEL: __global__ AICORE void cube_marked()
// CPP: #if defined(__DAV_CUBE__)
// CPP: #endif // __DAV_CUBE__

// CPP-LABEL: __global__ AICORE void vector_marked()
// CPP: #if defined(__DAV_VEC__)
// CPP: set_mask_norm();
// CPP: set_vector_mask(-1, -1);
// CPP: #endif // __DAV_VEC__
