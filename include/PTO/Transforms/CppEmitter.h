//===- CppEmitter.h - PTO-aware C++ emission helpers ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Local wrapper around the upstream EmitC-to-C++ translator. This preserves
// PTO-specific structured ops such as pto.simd.vec_scope until final printing
// instead of flattening them into verbatim brace markers.
//
//===----------------------------------------------------------------------===//

#ifndef PTO_TRANSFORMS_CPPEMITTER_H
#define PTO_TRANSFORMS_CPPEMITTER_H

#include "mlir/Support/LLVM.h"

namespace mlir {
class Operation;
namespace pto {

LogicalResult translateToCpp(Operation *op, raw_ostream &os,
                             bool declareVariablesAtTop = false);

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CPPEMITTER_H
