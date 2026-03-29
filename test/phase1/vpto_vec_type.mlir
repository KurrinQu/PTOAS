// RUN: ./build/tools/ptoas/ptoas %s 2>&1 | FileCheck %s

// The corrected Phase 1 contract keeps the normalized vector spelling while
// enforcing the 256-byte VPTO vector width contract.
// CHECK: error: '!pto.vec<32xf32>' expected exactly 256 bytes
module {
  func.func @illegal_f32_width(%arg0: !pto.vec<32xf32>) -> !pto.vec<32xf32> {
    return %arg0 : !pto.vec<32xf32>
  }
}
