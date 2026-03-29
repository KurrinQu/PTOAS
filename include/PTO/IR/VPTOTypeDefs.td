//===- VPTOTypeDefs.td ---------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_VPTOTYPEDEFS
#define MLIR_DIALECT_PTO_IR_VPTOTYPEDEFS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

def VecType : TypeDef<PTO_Dialect, "Vec"> {
  let mnemonic = "vec";
  let summary = "A 256-byte PTO low-level vector";

  let parameters = (ins
    "int64_t":$elementCount,
    "Type":$elementType
  );

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def MaskType : TypeDef<PTO_Dialect, "Mask"> {
  let mnemonic = "mask";
  let summary = "A PTO low-level predicate/mask register";
}

def AlignType : TypeDef<PTO_Dialect, "Align"> {
  let mnemonic = "align";
  let summary = "A PTO low-level vector_align carrier";
}

#endif // MLIR_DIALECT_PTO_IR_VPTOTYPEDEFS
